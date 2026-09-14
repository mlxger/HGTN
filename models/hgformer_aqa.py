from __future__ import annotations
from re import I

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from models.uncertain_regressor import DAE
from models.transformer import Transformer


def build_incidence_window(num_nodes: int, window: int) -> torch.Tensor:
    """Construct a temporal sliding-window incidence matrix H of shape (T, E).
    Each hyperedge covers a contiguous window of length `window`.
    E = T - window + 1.
    """
    assert window >= 2 and window <= num_nodes, "window must be in [2, T]"
    num_edges = num_nodes - window + 1
    H = torch.zeros(num_nodes, num_edges)
    for e in range(num_edges):
        H[e : e + window, e] = 1.0
    return H


def build_incidence_topk(x: torch.Tensor, topk: int) -> torch.Tensor:
    """Build a feature-similarity Top-K incidence matrix H (T, E=T).
    For each node i, create a hyperedge e_i that includes i and its top-k most similar nodes.
    Similarity is computed by cosine similarity over feature channels.

    Args:
        x: (B, T, C) or (T, C). If batch given, we take the batch mean for stability.
        topk: number of neighbors to include besides self.
    Returns:
        H: (T, T) with binary membership for each hyperedge.
    """
    if x.dim() == 3:
        # average across batch for robust graph (B, T, C) -> (T, C)
        x_repr = x.mean(dim=0)
    else:
        x_repr = x
    t, c = x_repr.shape
    # cosine similarity (T, T)
    x_norm = F.normalize(x_repr, p=2, dim=-1)
    sim = x_norm @ x_norm.t()
    # exclude self by temporarily removing then add back
    idx = torch.arange(t, device=sim.device)
    sim[range(t), range(t)] = -1e9
    # topk neighbors
    k = min(max(1, topk), t - 1)
    _, nn_idx = torch.topk(sim, k=k, dim=-1)
    H = torch.zeros(t, t, device=x.device if x.is_cuda else None)
    # include self and its top-k
    H[idx, idx] = 1.0
    H[idx.unsqueeze(1).expand(-1, k), nn_idx] = 1.0
    return H


def detect_action_phases(x: torch.Tensor, num_phases: int = 3) -> torch.Tensor:
    """Detect action phases based on feature dynamics.
    
    Args:
        x: (B, T, C) or (T, C) feature tensor
        num_phases: number of phases to divide into
    
    Returns:
        phase_labels: (T,) tensor with phase assignments
    """
    if x.dim() == 3:
        x_repr = x.mean(dim=0)  # (T, C)
    else:
        x_repr = x
    
    t, c = x_repr.shape
    
    # Compute feature change magnitude over time
    x_diff = torch.diff(x_repr, dim=0)  # (T-1, C)
    change_magnitude = torch.norm(x_diff, dim=1)  # (T-1,)
    
    # Pad to match original length
    change_magnitude = F.pad(change_magnitude, (0, 1), value=change_magnitude[-1])
    
    # Smooth the change magnitude
    kernel_size = min(5, t // 4)
    if kernel_size > 1:
        kernel = torch.ones(kernel_size, device=x.device) / kernel_size
        change_smooth = F.conv1d(
            change_magnitude.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2
        ).squeeze()
    else:
        change_smooth = change_magnitude
    
    # Divide into phases based on change patterns
    phase_boundaries = torch.quantile(change_smooth, torch.linspace(0, 1, num_phases + 1, device=x.device))
    phase_labels = torch.zeros(t, dtype=torch.long, device=x.device)
    
    for i in range(num_phases):
        if i == num_phases - 1:
            mask = change_smooth >= phase_boundaries[i]
        else:
            mask = (change_smooth >= phase_boundaries[i]) & (change_smooth < phase_boundaries[i + 1])
        phase_labels[mask] = i
    
    return phase_labels


def build_hierarchical_incidence(x: torch.Tensor, fine_windows: List[int], coarse_windows: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build hierarchical hypergraph incidence matrices.
    
    Args:
        x: (B, T, C) or (T, C) feature tensor
        fine_windows: window sizes for fine-grained temporal patterns
        coarse_windows: window sizes for coarse-grained global patterns
    
    Returns:
        H_fine: fine-grained incidence matrix
        H_coarse: coarse-grained incidence matrix
    """
    if x.dim() == 3:
        t = x.shape[1]
    else:
        t = x.shape[0]
    
    device = x.device
    
    # Fine-grained layer: local temporal patterns
    H_fine_list = []
    for window in fine_windows:
        if window <= t:
            H_w = build_incidence_window(t, window).to(device)
            H_fine_list.append(H_w)
    
    if H_fine_list:
        H_fine = torch.cat(H_fine_list, dim=1)  # concatenate along edge dimension
    else:
        H_fine = torch.zeros(t, 1, device=device)
    
    # Coarse-grained layer: global action structure
    H_coarse_list = []
    
    # Long-range temporal windows
    for window in coarse_windows:
        if window <= t:
            H_w = build_incidence_window(t, window).to(device)
            H_coarse_list.append(H_w)
    
    # Add global connectivity hyperedges (every k-th frame)
    stride = max(1, t // 8)  # connect every 1/8 of the sequence
    global_edges = []
    for start in range(0, t, stride):
        edge = torch.zeros(t, device=device)
        indices = list(range(start, t, stride))
        if len(indices) >= 2:  # need at least 2 nodes for a hyperedge
            edge[indices] = 1.0
            global_edges.append(edge.unsqueeze(1))
    
    if global_edges:
        H_global = torch.cat(global_edges, dim=1)
        H_coarse_list.append(H_global)
    
    if H_coarse_list:
        H_coarse = torch.cat(H_coarse_list, dim=1)
    else:
        H_coarse = torch.zeros(t, 1, device=device)
    
    return H_fine, H_coarse


def build_phase_aware_incidence(x: torch.Tensor, num_phases: int = 3, intra_phase_window: int = 3) -> torch.Tensor:
    """Build phase-aware hypergraph incidence matrix.
    
    Args:
        x: (B, T, C) or (T, C) feature tensor
        num_phases: number of action phases
        intra_phase_window: window size within each phase
    
    Returns:
        H_phase: phase-aware incidence matrix
    """
    if x.dim() == 3:
        t = x.shape[1]
    else:
        t = x.shape[0]
    
    device = x.device
    phase_labels = detect_action_phases(x, num_phases)
    
    H_phase_list = []
    
    # Intra-phase connections (dense within each phase)
    for phase_id in range(num_phases):
        phase_mask = (phase_labels == phase_id)
        phase_indices = torch.where(phase_mask)[0]
        
        if len(phase_indices) >= 2:
            # Create sliding windows within this phase
            phase_length = len(phase_indices)
            window_size = min(intra_phase_window, phase_length)
            
            for i in range(phase_length - window_size + 1):
                edge = torch.zeros(t, device=device)
                edge[phase_indices[i:i + window_size]] = 1.0
                H_phase_list.append(edge.unsqueeze(1))
    
    # Inter-phase connections (sparse connections between adjacent phases)
    for phase_id in range(num_phases - 1):
        current_phase_mask = (phase_labels == phase_id)
        next_phase_mask = (phase_labels == phase_id + 1)
        
        current_indices = torch.where(current_phase_mask)[0]
        next_indices = torch.where(next_phase_mask)[0]
        
        if len(current_indices) > 0 and len(next_indices) > 0:
            # Connect end of current phase with beginning of next phase
            edge = torch.zeros(t, device=device)
            # Take last few frames of current phase and first few frames of next phase
            transition_size = min(2, len(current_indices), len(next_indices))
            edge[current_indices[-transition_size:]] = 1.0
            edge[next_indices[:transition_size]] = 1.0
            H_phase_list.append(edge.unsqueeze(1))
    
    # Key frame connections (connect frames with high feature change)
    if x.dim() == 3:
        x_repr = x.mean(dim=0)
    else:
        x_repr = x
    
    x_diff = torch.diff(x_repr, dim=0)
    change_magnitude = torch.norm(x_diff, dim=1)
    change_magnitude = F.pad(change_magnitude, (0, 1), value=change_magnitude[-1])
    
    # Select top-k frames with highest change
    k_key = min(5, t // 4)
    if k_key >= 2:
        _, key_indices = torch.topk(change_magnitude, k_key)
        edge = torch.zeros(t, device=device)
        edge[key_indices] = 1.0
        H_phase_list.append(edge.unsqueeze(1))
    
    if H_phase_list:
        H_phase = torch.cat(H_phase_list, dim=1)
    else:
        H_phase = torch.zeros(t, 1, device=device)
    
    return H_phase


def hypergraph_attention_bias(H: torch.Tensor, eps: float = 1e-6, use_de_inv: bool = True) -> torch.Tensor:
    """Compute additive attention bias B from incidence H.
    B = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}  (if use_de_inv)
        or Dv^{-1/2} H De^{-1/2} H^T Dv^{-1/2}
    Returns (T, T) tensor.
    """
    Dv = H.sum(dim=1).clamp_min(eps)
    De = H.sum(dim=0).clamp_min(eps)
    inv_sqrt_Dv = Dv.pow(-0.5)
    if use_de_inv:
        De_term = De.pow(-1.0)
    else:
        De_term = De.pow(-0.5)
    tmp = H * De_term.unsqueeze(0)
    A = tmp @ H.t()
    B = inv_sqrt_Dv.unsqueeze(1) * A * inv_sqrt_Dv.unsqueeze(0)
    return B


class GraphConvolutionLayer_t(nn.Module):
    def __init__(self, in_channels, out_channels, frames, dropout_rate=0.4):
        super(GraphConvolutionLayer_t, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels, out_channels)*(1/in_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.adjweight = nn.Parameter(torch.ones(frames))


    def forward(self, x, adj_matrix):

        batch_size, frames, in_channels = x.size()
        adj_weight = adj_matrix * self.adjweight
        adj_weight = F.softmax(adj_weight,dim=1)
        featuremap_adj_t = adj_weight
        x = x.permute(0, 2, 1)
        x = torch.bmm(x, adj_weight)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        weight = self.weight.unsqueeze(0).repeat(batch_size,1,1)
        attention_weights = F.softmax(weight, dim=1)
        attention_weights_t = attention_weights

        output = torch.bmm(x, attention_weights)

        return output, featuremap_adj_t, attention_weights_t


class GraphConvolutionalNetwork_t(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork_t, self).__init__()
        #self.gc1_t = GraphConvolutionLayer_t(in_channels, hidden_channels,frames=68)
        #self.gc2_t = GraphConvolutionLayer_t(hidden_channels, out_channels,frames=68)
        '''Fis V'''
        self.gc1_t = GraphConvolutionLayer_t(in_channels, hidden_channels,frames=124)
        self.gc2_t = GraphConvolutionLayer_t(hidden_channels, out_channels,frames=124)

    def forward(self, x, adj_matrix):


        result, featuremap_adj_t1,attention_weights_t1 = self.gc1_t(x, adj_matrix)
        x = F.elu(result,alpha=1.0)
        result2, featuremap_adj_t2, attention_weights_t2 = self.gc2_t(x, adj_matrix)
        x = F.elu(result2, alpha=1.0)
        return x,featuremap_adj_t2, attention_weights_t2

channels_t = 1024
hidden_channels_t = 512
output_channels_t = 1024
def construct_adjacency_matrix_t(image_size):

    adj_matrix = torch.zeros((image_size, image_size), dtype=torch.float)

    # 遍历所有数据点
    for i in range(image_size):
        # 每个点与自己相连
        adj_matrix[i, i] = 1

        # 如果不是第一个点，则与前一个点相连
        if i > 0:
            adj_matrix[i, i - 1] = 1
            adj_matrix[i - 1, i] = 1

        # 如果不是最后一个点，则与后一个点相连
        if i < image_size-1:
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1


    return adj_matrix

class GraphConvolutionLayer_s(nn.Module):
    def __init__(self, in_channels, out_channels, frames, dropout_rate=0.4):
        super(GraphConvolutionLayer_s, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels, out_channels)*(1/in_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.adjweight = nn.Parameter(torch.ones(frames))



    def forward(self, x, adj_matrix):

        batch_size, frames, in_channels = x.size()
        adj_weight = adj_matrix * self.adjweight
        adj_weight = F.softmax(adj_weight,dim=1)
        featuremap_adj_s = adj_weight
        x = torch.bmm(x, adj_weight)
        x = self.dropout(x)


        x = x.permute(0, 2, 1)
        weight = self.weight.unsqueeze(0).repeat(batch_size,1,1)
        attention_weights = F.softmax(weight, dim=1)
        attention_weights_s = attention_weights
        output = torch.bmm(x, attention_weights)
        output = output.permute(0, 2, 1)
        return output, featuremap_adj_s, attention_weights_s
class GraphConvolutionalNetwork_s(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork_s, self).__init__()
        self.gc1_s = GraphConvolutionLayer_s(in_channels, hidden_channels,frames=1024)
        self.gc2_s = GraphConvolutionLayer_s(hidden_channels, out_channels,frames=1024)

    def forward(self, x, adj_matrix):
        result, featuremap_adj_s1, attention_weights_s1 = self.gc1_s(x, adj_matrix)
        x = F.elu(result,alpha=1.0)
        result2, featuremap_adj_s2, attention_weights_s2 = self.gc2_s(x, adj_matrix)
        x = F.elu(result2,alpha=1.0)
        return x, featuremap_adj_s2, attention_weights_s2
#channels_s = 68
#hidden_channels_s = 136
#output_channels_s = 68



channels_s = 124
hidden_channels_s = 248
output_channels_s = 124
def construct_adjacency_matrix_s(image_size):
    num_pixels = image_size * image_size
    # 28*28= 784
    adjacency_matrix_s = torch.zeros((num_pixels, num_pixels), dtype=torch.float)

    # Iterate over all pixels
    for i in range(num_pixels):
        # Get row and column indices of current pixel
        row = i // image_size
        col = i % image_size

        # Check if neighboring pixels are within the image boundaries
        if row > 0:
            adjacency_matrix_s[i, i - image_size] = 1.0  # Connect to pixel above
        if row < image_size - 1:
            adjacency_matrix_s[i, i + image_size] = 1.0  # Connect to pixel below
        if col > 0:
            adjacency_matrix_s[i, i - 1] = 1.0  # Connect to pixel on the left
        if col < image_size - 1:
            adjacency_matrix_s[i, i + 1] = 1.0  # Connect to pixel on the right

        # Connect to diagonal pixels
        if row > 0 and col > 0:
            adjacency_matrix_s[i, i - image_size - 1] = 1.0  # Connect to pixel on the top-left diagonal
        if row > 0 and col < image_size - 1:
            adjacency_matrix_s[i, i - image_size + 1] = 1.0  # Connect to pixel on the top-right diagonal
        if row < image_size - 1 and col > 0:
            adjacency_matrix_s[i, i + image_size - 1] = 1.0  # Connect to pixel on the bottom-left diagonal
        if row < image_size - 1 and col < image_size - 1:
            adjacency_matrix_s[i, i + image_size + 1] = 1.0  # Connect to pixel on the bottom-right diagonal

        # Connect to self
        adjacency_matrix_s[i, i] = 1.0

    return adjacency_matrix_s


class HyperAQAFormer(nn.Module):
    """Hypergraph-augmented Transformer for Action Quality Assessment.

    Input: x of shape (B, T, C)
    - Build multiple temporal hypergraphs (multi-scale sliding windows, hierarchical structure, phase-aware)
    - Convert to additive attention bias and inject into encoder self-attention
    - Decoder uses learnable query prototypes and a regressor head (compatible with existing training code)
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        n_head: int = 1,
        n_encoder: int = 1,
        n_decoder: int = 1,
        n_query: int = 1,
        dropout: float = 0.0,
        windows: Optional[List[int]] = None,
        use_topk: bool = False,
        topk: int = 4,
        de_use_inv: bool = True,
        # New parameters for enhanced hypergraph
        use_hierarchical: bool = False,
        fine_windows: Optional[List[int]] = None,
        coarse_windows: Optional[List[int]] = None,
        use_phase_aware: bool = False,                
        num_phases: int = 3,
        intra_phase_window: int = 3,
    ) -> None:
        super().__init__()

        self.windows = windows if windows is not None else [3]
        self.use_topk = use_topk
        self.topk = topk
        self.de_use_inv = de_use_inv
        
        # Enhanced hypergraph parameters
        self.use_hierarchical = use_hierarchical
        self.fine_windows = fine_windows if fine_windows is not None else [2, 3]
        self.coarse_windows = coarse_windows if coarse_windows is not None else [5, 8]
        self.use_phase_aware = use_phase_aware
        self.num_phases = num_phases
        self.intra_phase_window = intra_phase_window


        # Calculate total number of bias components
        num_components = len(self.windows)
        if self.use_topk:
            num_components += 1
        if self.use_hierarchical:
            num_components += 2  # fine and coarse
        if self.use_phase_aware:
            num_components += 1

        # Learnable scaling for each bias component
        self.bias_scales = nn.Parameter(torch.ones(num_components))

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.1)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2.data.fill_(0.1)
        self.gcn_t = GraphConvolutionalNetwork_t(channels_t, hidden_channels_t, output_channels_t)
        self.gcn_s = GraphConvolutionalNetwork_s(channels_s, hidden_channels_s, output_channels_s)

        # input projection (T,C) -> (T,hidden)
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        #self.regressor = DAE()
        self.register_buffer("weight", torch.linspace(0, 1, n_query, requires_grad=False))

         # 缓存与可视化相关的状态
        #self._force_zero_bias = False
        #self.last_bias_components: List[Tuple[str, torch.Tensor]] = []
        #self.last_bias_sum: Optional[torch.Tensor] = None
        #self.last_attn_bias: Optional[torch.Tensor] = None


    @staticmethod
    def _stack_bias(biases: List[torch.Tensor], scales: torch.Tensor) -> torch.Tensor:
        """Weighted sum of bias components.
        biases: list of (T,T) tensors on same device; scales: (K,) learnable.
        """
        stacked = torch.stack(biases, dim=0)  # (K, T, T)
        scaled = scales.view(-1, 1, 1) * stacked
        return scaled.sum(dim=0)  # (T, T)

    def _build_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Build additive attention bias (T,T) from multi-source hypergraphs."""
        b, t, c = x.shape
        device = x.device
        bias_components: List[torch.Tensor] = []
        #component_labels: List[str] = []

        # Original multi-scale sliding windows
        for w in self.windows:
            if w <= t:
                H = build_incidence_window(t, w).to(device)
                B = hypergraph_attention_bias(H, use_de_inv=self.de_use_inv).to(device)
                bias_components.append(B)

        # Optional top-k hyperedges
        if self.use_topk:
            Hk = build_incidence_topk(x.detach(), topk=self.topk)  # (T,T)
            Bk = hypergraph_attention_bias(Hk, use_de_inv=self.de_use_inv).to(device)
            bias_components.append(Bk)

        # Hierarchical hypergraph structure
        if self.use_hierarchical:
            H_fine, H_coarse = build_hierarchical_incidence(x.detach(), self.fine_windows, self.coarse_windows)
            
            B_fine = hypergraph_attention_bias(H_fine, use_de_inv=self.de_use_inv).to(device)
            B_coarse = hypergraph_attention_bias(H_coarse, use_de_inv=self.de_use_inv).to(device)
            
            bias_components.append(B_fine)
            bias_components.append(B_coarse)

        # Phase-aware hypergraph
        if self.use_phase_aware:
            H_phase = build_phase_aware_incidence(x.detach(), self.num_phases, self.intra_phase_window)
            B_phase = hypergraph_attention_bias(H_phase, use_de_inv=self.de_use_inv).to(device)
            bias_components.append(B_phase)

        # Sum with learnable scales
        if bias_components:
            scales = self.bias_scales[:len(bias_components)]
            B_sum = self._stack_bias(bias_components, scales)
        else:
            B_sum = torch.zeros(t, t, device=device)

        """
        # 缓存（搬运到 CPU，避免显存占用）
        self.last_bias_components = [
            (label, B.detach().cpu()) for label, B in zip(component_labels, bias_components)
        ]
        self.last_bias_sum = B_sum.detach().cpu()

        if getattr(self, "_force_zero_bias", False):
            return torch.zeros_like(B_sum)
        """
        
        return B_sum

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        b, t, c = x.shape

        adj_matrix_t = construct_adjacency_matrix_t(t)
        adj_matrix_t.unsqueeze(0)
        adj_matrix_t = adj_matrix_t.repeat(b,1,1).cuda()
        x1, featuremap_adj_t2, attention_weights_t2  = self.gcn_t(x, adj_matrix_t)

        adj_matrix_s = construct_adjacency_matrix_s(32)
        adj_matrix_s.unsqueeze(0)
        adj_matrix_s = adj_matrix_s.repeat(b,1,1).cuda()
        x2, featuremap_adj_s2, attention_weights_s2 = self.gcn_s(x, adj_matrix_s)

        x = x * 0.8 + x1 * self.w1 + x2 * self.w2

         # hypergraph-aware attention bias
        attn_bias = self._build_bias(x)  # (T, T)
        #self.last_attn_bias = attn_bias.detach().cpu()

        # projection
        x_proj = self.in_proj(x.transpose(1, 2)).transpose(1, 2)  # (B, T, hidden)

        # transformer encoder with additive bias
        encode_x = self.transformer.encoder(x_proj, mask=attn_bias)

         # decoder with learnable queries
        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        q1, _ = self.transformer.decoder(q, encode_x)
        # scoring head
        s = self.regressor(q1)  # (B, n_query, n_query)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (B, n_query)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        return {"output": out, "embed": q1}


"""
    model = HyperAQAFormer(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_head=args.n_head,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        n_query=args.n_query,
        dropout=args.dropout,
        
        # 原有参数 - 保持基础多尺度功能
        windows=[3, 5],           # 基础滑动窗口
        use_topk=True,            # 保持特征相似性连接
        topk=4,
        de_use_inv=True,
        
        # 新增层次化超图参数
        use_hierarchical=True,    # 启用层次化结构
        fine_windows=[2, 3, 4],   # 细粒度层：更密集的小窗口捕获局部细节
        coarse_windows=[6, 8, 12], # 粗粒度层：更大窗口捕获全局结构
        
        # 新增动作阶段感知参数
        use_phase_aware=True,     # 启用阶段感知
        num_phases=3,             # 动作分为3个阶段（准备-执行-结束）
        intra_phase_window=4,     # 阶段内连接窗口，稍大以捕获完整子动作
    ).to(device)
"""