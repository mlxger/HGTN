from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.transformer import Transformer
from models.hgformer_aqa import (
    build_incidence_window, build_incidence_topk, 
    build_hierarchical_incidence, build_phase_aware_incidence,
    hypergraph_attention_bias
)


class AttentionGuidedPrototypeAlignment(nn.Module):
    
    
    def __init__(self, hidden_dim: int, n_query: int, n_head: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_query = n_query
        self.n_head = n_head
        
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_head,
            dropout=0.1,
            batch_first=True
        )
        
        
        self.prototype_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
       
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        
        self.temporal_importance = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, prototype_queries: torch.Tensor, encoded_features: torch.Tensor):
        """
        Args:
            prototype_queries: (B, n_query, hidden_dim) 
            encoded_features: (B, T, hidden_dim) 
        Returns:
            enhanced_prototypes: (B, n_query, hidden_dim) 
            attention_weights: (B, n_query, T) 
        """
        B, n_query, hidden_dim = prototype_queries.shape
        B, T, _ = encoded_features.shape
        
        #
        enhanced_prototypes, attention_weights = self.multihead_attn(
            query=prototype_queries,  # (B, n_query, hidden_dim)
            key=encoded_features,     # (B, T, hidden_dim)  
            value=encoded_features    # (B, T, hidden_dim)
        )
        
        
        enhanced_prototypes = self.prototype_enhancer(enhanced_prototypes)
        
        
        quality_gates = self.quality_gate(enhanced_prototypes)
        enhanced_prototypes = enhanced_prototypes * quality_gates
        
        
        enhanced_prototypes = enhanced_prototypes + prototype_queries
        
        
        temporal_importance = self.temporal_importance(encoded_features)  # (B, T, 1)
        
        return enhanced_prototypes, attention_weights, temporal_importance


class EnhancedHyperAQAFormer(nn.Module):
    
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        n_head: int = 1,
        n_encoder: int = 1,
        n_decoder: int = 1,
        n_query: int = 4,
        dropout: float = 0.0,
        windows: Optional[List[int]] = None,
        use_topk: bool = False,
        topk: int = 4,
        de_use_inv: bool = True,
        # Enhanced prototype parameters
        use_prototype_alignment: bool = True,
        alignment_heads: int = 4,
        use_adaptive_weighting: bool = True,
        use_temporal_importance: bool = True,
    ) -> None:
        super().__init__()

        self.windows = windows if windows is not None else [3, 5]
        self.use_topk = use_topk
        self.topk = topk
        self.de_use_inv = de_use_inv
        self.n_query = n_query
        self.use_prototype_alignment = use_prototype_alignment
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_temporal_importance = use_temporal_importance

        # Calculate total number of bias components
        num_components = len(self.windows)
        if self.use_topk:
            num_components += 1

        # Learnable scaling for each bias component
        self.bias_scales = nn.Parameter(torch.ones(num_components))

        # input projection
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

        #
        self.prototype = nn.Embedding(n_query, hidden_dim)
        
        
        if self.use_prototype_alignment:
            self.prototype_alignment = AttentionGuidedPrototypeAlignment(
                hidden_dim=hidden_dim,
                n_query=n_query,
                n_head=alignment_heads
            )
        
        
        if self.use_adaptive_weighting:
            self.adaptive_weighter = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_query),
                nn.Softmax(dim=-1)
            )
        
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        
        self.register_buffer("fixed_weight", torch.linspace(0, 1, n_query, requires_grad=False))

    @staticmethod
    def _stack_bias(biases: List[torch.Tensor], scales: torch.Tensor) -> torch.Tensor:
        """Weighted sum of bias components."""
        stacked = torch.stack(biases, dim=0)  # (K, T, T)
        scaled = scales.view(-1, 1, 1) * stacked
        return scaled.sum(dim=0)  # (T, T)

    def _build_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Build additive attention bias (T,T) from multi-source hypergraphs."""
        b, t, c = x.shape
        device = x.device
        bias_components: List[torch.Tensor] = []

        # Multi-scale sliding windows
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

        # Sum with learnable scales
        if bias_components:
            scales = self.bias_scales[:len(bias_components)]
            B_sum = self._stack_bias(bias_components, scales)
        else:
            B_sum = torch.zeros(t, t, device=device)
        
        return B_sum

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        b, t, c = x.shape

        # hypergraph-aware attention bias
        attn_bias = self._build_bias(x)  # (T, T)

        # projection
        x_proj = self.in_proj(x.transpose(1, 2)).transpose(1, 2)  # (B, T, hidden)

        # transformer encoder with additive bias
        encode_x = self.transformer.encoder(x_proj, mask=attn_bias)

        
        prototype_queries = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)  # (B, n_query, hidden)
        
        
        if self.use_prototype_alignment:
            enhanced_prototypes, attn_weights, temporal_importance = self.prototype_alignment(
                prototype_queries, encode_x
            )
        else:
           
            enhanced_prototypes, attn_weights = self.transformer.decoder(prototype_queries, encode_x)
            temporal_importance = None
        
        
        prototype_scores = self.regressor(enhanced_prototypes).squeeze(-1)  # (B, n_query)
        
        
        if self.use_adaptive_weighting:
            
            global_feature = encode_x.mean(dim=1)  # (B, hidden)
            adaptive_weights = self.adaptive_weighter(global_feature)  # (B, n_query)
        else:
            adaptive_weights = self.fixed_weight.unsqueeze(0).repeat(b, 1)
        
        
        final_scores = torch.sum(adaptive_weights * F.softmax(prototype_scores, dim=-1), dim=1)
        
        return {
            "output": final_scores,
            "embed": enhanced_prototypes,
            "prototype_scores": prototype_scores,
            "adaptive_weights": adaptive_weights,
            "attention_weights": attn_weights,
            "temporal_importance": temporal_importance
        }
