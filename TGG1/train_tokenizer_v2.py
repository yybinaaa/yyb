import torch
import numpy as np
import argparse
from pathlib import Path

# 从我们刚刚创建的文件中导入 Tokenizer 类
from tokenizer import RQ_KMeans

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train an RQ-KMeans tokenizer using PyTorch and Faiss-GPU.")
    
    parser.add_argument('--embedding_file', default='item_embeddings.npy', type=str,
                        help="Path to the .npy file containing item embeddings.")
                        
    parser.add_argument('--output_file', default='tokenizer/rq_kmeans_tokenizer.pth', type=str,
                        help="Path to save the trained tokenizer model (.pth).")
                        
    parser.add_argument('--num_layers', default=3, type=int,
                        help="Number of residual quantization layers.")
                        
    parser.add_argument('--codebook_size', default=8192, type=int,
                        help="Number of clusters (K) for each K-Means layer.")

    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="Device to use for training ('cuda' or 'cpu').")
                        
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training on CPU.")
        device = torch.device('cpu')

    # --- 1. 加载数据并移动到指定设备 ---
    print(f"Loading embeddings from: {args.embedding_file}")
    embeddings_np = np.load(args.embedding_file)
    
    # 忽略 padding embedding
    if np.all(embeddings_np[0] == 0):
        print("Ignoring the first embedding vector (assumed to be padding).")
        embeddings_np = embeddings_np[1:]
        
    # 将 NumPy 数组转换为 PyTorch 张量并移动到 GPU (如果可用)
    embeddings_tensor = torch.from_numpy(embeddings_np).to(device)
    
    # --- 2. 初始化并训练 Tokenizer ---
    tokenizer = RQ_KMeans(
        num_layers=args.num_layers,
        codebook_size=args.codebook_size
    )
    # 将 tokenizer 模型本身也移动到 GPU，以便在 GPU 上计算残差
    tokenizer.to(device)
    
    # 调用训练方法
    tokenizer.train_tokenizer(embeddings_tensor, use_gpu=(device.type == 'cuda'), seed=args.seed)

    # --- 3. 保存训练好的模型 ---
    tokenizer.save(args.output_file)

    # --- 4. (可选) 测试一下 tokenizer ---
    print("\n--- Testing the trained tokenizer ---")
    tokenizer.eval() # 设置为评估模式
    
    # 随机取10个样本进行测试
    sample_indices = np.random.choice(len(embeddings_tensor), 10, replace=False)
    sample_vectors = embeddings_tensor[sample_indices]
    
    # 调用 tokenize 方法
    semantic_ids = tokenizer.tokenize(sample_vectors)
    
    print("Sample original vectors (first 5 dims):")
    print(sample_vectors[:, :5])
    print("\nGenerated Semantic IDs:")
    print(semantic_ids)
    print("\nTest complete.")

if __name__ == '__main__':
    main()