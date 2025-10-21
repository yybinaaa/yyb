# in prepare_embeddings.py (完整版)
import torch
import numpy as np
import argparse
from pathlib import Path

from dataset_v2 import OneRecV2Dataset # 注意这里的导入
from model_v2 import OneRecV2Model, ModelConfig

def get_args():
    parser = argparse.ArgumentParser(description="Extract item embeddings.")
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--model_checkpoint', required=True, type=str)
    parser.add_argument('--output_file', default='item_embeddings.npy', type=str)
    # 添加必要的参数以初始化 Dataset 和 Model
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str)
    parser.add_argument('--codebook_size', default=8192, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cpu')
    
    # 使用 semantic_id 来初始化，虽然我们只提取 item_id embedding，但需要保持模型结构一致
    dataset = OneRecV2Dataset(args.data_dir, args, codebook_size=args.codebook_size)
    config = ModelConfig()
    model = OneRecV2Model(config, dataset).to(device)
    
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.eval()
    
    # 提取 item_id embedding，它在 ContextProcessor 里
    embedding_layer = model.context_processor.embedding_tables['item_id']
    item_embeddings_np = embedding_layer.weight.detach().cpu().numpy()
    
    print(f"Extracted embedding matrix with shape: {item_embeddings_np.shape}")
    
    np.save(args.output_file, item_embeddings_np)
    print(f"Embeddings saved to: {args.output_file}")

if __name__ == '__main__':
    main()