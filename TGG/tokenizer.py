import torch
import numpy as np
import faiss
import pickle
from pathlib import Path

class RQ_KMeans(torch.nn.Module):
    """
    一个 PyTorch 风格的 RQ-KMeans Tokenizer。
    [修正] 修复了加载 state_dict 时 Unexpected key 的问题。
    """
    def __init__(self, num_layers: int, codebook_size: int, dim: int = 256): # [修正] 需要知道维度来预创建参数
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self._is_trained = False

        # [核心修正] 在 __init__ 中就根据配置创建好参数的“骨架”
        # 这样做，load_state_dict 就能找到对应的键 "codebooks.0", "codebooks.1", ...
        self.codebooks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(codebook_size, dim), requires_grad=False) for _ in range(num_layers)]
        )

    def train_tokenizer(self, data: torch.Tensor, use_gpu: bool = True, seed: int = 42):
        """
        使用 Faiss-GPU 训练 Tokenizer 的核心方法。
        """
        assert data.is_cuda if use_gpu else not data.is_cuda, "Data tensor must be on the correct device."
        
        print("Starting RQ-KMeans training...")
        residuals = data.clone()
        
        # 检查维度是否匹配
        if data.shape[1] != self.dim:
            raise ValueError(f"Input data dimension ({data.shape[1]}) does not match model dimension ({self.dim}).")

        for layer_idx in range(self.num_layers):
            print(f"\n--- Training Layer {layer_idx + 1}/{self.num_layers} ---")
            
            kmeans = faiss.Kmeans(d=self.dim, k=self.codebook_size, niter=20, verbose=True, seed=seed + layer_idx)
            data_for_faiss = residuals.cpu().numpy()
            
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(self.dim))
                kmeans.train(data_for_faiss, index=gpu_index)
            else:
                kmeans.train(data_for_faiss)
            
            # [核心修正] 不再是 append，而是直接给预先创建好的参数赋值
            codebook_tensor = torch.from_numpy(kmeans.centroids).to(data.device)
            self.codebooks[layer_idx].data = codebook_tensor
            
            if layer_idx < self.num_layers - 1:
                print("  Calculating residuals for the next layer...")
                indices = torch.cdist(residuals, codebook_tensor).argmin(dim=1)
                quantized_vectors = codebook_tensor[indices]
                residuals = residuals - quantized_vectors
        
        self._is_trained = True
        print("\nTokenizer training complete! ✨")

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Tokenizer has not been trained yet. Call train_tokenizer() first.")
        
        residuals = x.clone()
        semantic_ids = []

        for layer_idx in range(self.num_layers):
            codebook = self.codebooks[layer_idx]
            indices = torch.cdist(residuals, codebook).argmin(dim=1)
            semantic_ids.append(indices)
            
            quantized_vectors = codebook[indices]
            residuals = residuals - quantized_vectors
            
        return torch.stack(semantic_ids, dim=1)

    def save(self, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)
        print(f"Tokenizer saved to {file_path}")

    def load(self, file_path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(file_path, map_location=device))
        self._is_trained = True
        self.to(device)
        print(f"Tokenizer loaded from {file_path}")