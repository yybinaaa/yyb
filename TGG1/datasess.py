# test_time_feature.py
from dataset import MyDataset
import torch

dataset = MyDataset("data/", args)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

for batch in loader:
    _, _, _, _, _, _, seq_feats, _, _ = batch
    print("Time features example:")
    print("hour (200):", [f['200'] for f in seq_feats[0] if '200' in f])
    print("log_gap (203):", [f['203'] for f in seq_feats[0] if '203' in f])
    break
