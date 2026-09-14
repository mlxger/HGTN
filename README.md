# Adaptive Hypergraph Transformer for Action Quality Assessment


## Environment
```bash
  conda create --name hgtn python==3.10
  conda activate hgtn
  pip install -r requirment.txt
```

# Datasets

The extracted VST features and label files of the Rhythmic Gymnastics and Fis-V datasets can be downloaded from the [GDLT](https://github.com/xuangch/CVPR22_GDLT) repository.

The extracted VST features, label files, and original videos of the FineFS dataset can be downloaded from the [FineFS-dataset](https://github.com/yanliji/FineFS-dataset) repository.

The original videos of the Rhythmic Gymnastics dataset can be downloaded from the [ACTION-NET](https://github.com/qinghuannn/ACTION-NET) repository.

The original videos of the Fis-V dataset can be downloaded from the [MS_LSTM](https://github.com/chmxu/MS_LSTM) repository.

# Running
```bash
  python main.py 
```

# Testing
```bash
  python main.py --ckpt {pkl file here} --test
```
