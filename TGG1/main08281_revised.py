
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model08281 import SASRecWithTime  # 确保是你修改后的模型类
from dataset08281 import RecDataset
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args(args=[])
    return args

def collate_fn(batch):
    seq_ids, sparse_feats, mm_feats, time_feats, pos_seqs = zip(*batch)

    seq_ids = torch.stack(seq_ids, dim=0)
    sparse_feats = torch.stack(sparse_feats, dim=0)
    time_feats = torch.stack(time_feats, dim=0)
    pos_seqs = torch.stack(pos_seqs, dim=0)
    B, L = pos_seqs.shape
    device = pos_seqs.device

    mm_batch = {}
    for fid in mm_feats[0].keys():
        mm_batch[fid] = torch.stack([x[fid] for x in mm_feats], dim=0)

    pool = pos_seqs.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)  # [B, L, B]
    all_idx = torch.arange(B, device=device)
    all_idx3 = all_idx.view(1, 1, B).expand(B, L, B)
    self_idx3 = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, B)
    mask = all_idx3 != self_idx3
    neg_indices = all_idx3[mask].view(B, L, B - 1)
    neg_items_bt = torch.gather(pool, dim=2, index=neg_indices)  # [B, L, B-1]

    return seq_ids, sparse_feats, mm_batch, time_feats, pos_seqs, neg_items_bt

def main():
    args = get_args()
    dataset = RecDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SASRecWithTime(dataset.user_num, dataset.item_num, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            seq_ids, sparse_feats, mm_batch, time_feats, pos_seqs, neg_items_bt = batch
            seq_ids = seq_ids.to(args.device)
            sparse_feats = sparse_feats.to(args.device)
            time_feats = time_feats.to(args.device)
            pos_seqs = pos_seqs.to(args.device)
            neg_items_bt = neg_items_bt.to(args.device)
            for k in mm_batch:
                mm_batch[k] = mm_batch[k].to(args.device)

            loss_mask = (pos_seqs != 0)
            optimizer.zero_grad()
            seq_embs, pos_embs, neg_embs = model(seq_ids, pos_seqs, neg_items_bt, time_feats, sparse_feats, mm_batch)
            loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    main()
