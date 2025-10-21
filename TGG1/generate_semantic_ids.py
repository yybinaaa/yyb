import torch
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# 从我们创建的文件中导入 Tokenizer 类
from tokenizer import RQ_KMeans

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Generate and save semantic IDs for all items using a trained tokenizer.")
    
    parser.add_argument('--embedding_file', default='item_embeddings.npy', type=str,
                        help="Path to the .npy file containing item embeddings.")
                        
    parser.add_argument('--tokenizer_path', default='tokenizer/rq_kmeans_tokenizer.pth', type=str,
                        help="Path to the trained tokenizer model (.pth).")
                        
    parser.add_argument('--output_file', default='semantic_id_map.json', type=str,
                        help="Path to save the final item_reid to semantic_ids mapping JSON file.")

    parser.add_argument('--batch_size', default=4096, type=int,
                        help="Batch size for tokenization inference to manage memory.")
                        
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="Device to use for inference ('cuda' or 'cpu').")
                        
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Running on CPU.")
        device = torch.device('cpu')

    # --- 1. 加载训练好的 Tokenizer ---
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    # 我们需要知道 tokenizer 的参数来初始化它
    # 假设参数与训练时相同，更健壮的做法是将参数保存在tokenizer文件里
    # 但为了简单，我们这里直接使用默认值
    from model_v2 import ModelConfig 
    config = ModelConfig()
    tokenizer = RQ_KMeans(num_layers=3, codebook_size=8192, dim=config.hidden_dim) 
    tokenizer.load(args.tokenizer_path, device=device)
    tokenizer.eval() # 设置为评估模式
    print("Tokenizer loaded successfully.")

    # --- 2. 加载完整的 Item Embeddings ---
    print(f"Loading embeddings from: {args.embedding_file}")
    embeddings_np = np.load(args.embedding_file)
    
    # [关键] 我们需要处理 padding token (索引为0)
    # 我们先将它从计算中移除，最后再给它一个特殊的语义ID
    has_padding = np.all(embeddings_np[0] == 0)
    if has_padding:
        print("Separating padding embedding (at index 0).")
        embeddings_tensor = torch.from_numpy(embeddings_np[1:]).to(device)
    else:
        embeddings_tensor = torch.from_numpy(embeddings_np).to(device)
        
    num_items = len(embeddings_tensor)
    print(f"Loaded {num_items} non-padding item embeddings.")

    # --- 3. 分批生成所有语义ID ---
    print("Generating semantic IDs for all items...")
    all_semantic_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_items, args.batch_size), desc="Tokenizing"):
            batch_embeddings = embeddings_tensor[i : i + args.batch_size]
            batch_semantic_ids = tokenizer.tokenize(batch_embeddings)
            all_semantic_ids.append(batch_semantic_ids.cpu())
            
    # 合并所有批次的结果
    final_semantic_ids = torch.cat(all_semantic_ids, dim=0).numpy()

    # --- 4. 构建并保存最终的映射文件 ---
    print("Building the final item ID to semantic ID map...")
    
    # 创建一个从 reid -> [s1, s2, s3] 的字典
    # 注意：reid 从 1 开始，因为我们跳过了 padding
    id_map = {
        # reid 是 int, semantic_ids 是 list of int
        int(reid): [int(s) for s in semantic_ids]
        for reid, semantic_ids in enumerate(final_semantic_ids, start=1)
    }
    
    # 如果原始数据有 padding，为 reid=0 添加一个特殊的映射
    if has_padding:
        id_map[0] = [0, 0, 0] # 或者其他你选择的 padding ID

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving semantic ID map to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(id_map, f, indent=4)
        
    print("\nGeneration complete! ✨")
    print(f"A total of {len(id_map)} item mappings have been saved.")
    print("We are now ready to upgrade our main model training!")

if __name__ == '__main__':
    main()