import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import time


class RecommenderFAISS:
    """
    专门为推荐系统优化的GPU FAISS集成
    用于加速InfoNCE loss计算和负样本采样
    """
    
    def __init__(self, dimension: int, device: str = 'cuda'):
        """
        初始化推荐系统FAISS
        
        Args:
            dimension: 嵌入维度
            device: 设备类型
        """
        self.dimension = dimension
        self.device = device
        self.vectors = None
        self.index_type = None
        self.is_built = False
        self.vector_ids = None  # 存储向量对应的item_id
        
    def add_vectors(self, vectors: torch.Tensor, vector_ids: Optional[torch.Tensor] = None):
        """
        添加向量到索引
        
        Args:
            vectors: 嵌入向量 [num_vectors, dimension]
            vector_ids: 向量对应的item_id [num_vectors]
        """
        if vectors.dim() != 2 or vectors.size(1) != self.dimension:
            raise ValueError(f"Vectors must be 2D with shape [num_vectors, {self.dimension}]")
        
        if self.vectors is None:
            self.vectors = vectors.to(self.device)
            if vector_ids is not None:
                self.vector_ids = vector_ids.to(self.device)
        else:
            self.vectors = torch.cat([self.vectors, vectors.to(self.device)], dim=0)
            if vector_ids is not None:
                self.vector_ids = torch.cat([self.vector_ids, vector_ids.to(self.device)], dim=0)
    
    def build_index(self, index_type: str = 'flat', **kwargs):
        """
        构建索引
        
        Args:
            index_type: 索引类型 ('flat', 'ivf')
            **kwargs: 索引特定参数
        """
        if self.vectors is None:
            raise ValueError("No vectors added to index")
        
        self.index_type = index_type
        
        if index_type == 'flat':
            # 简单暴力搜索，不需要预处理
            self.is_built = True
            
        elif index_type == 'ivf':
            # 倒排文件索引
            self._build_ivf_index(**kwargs)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def _build_ivf_index(self, n_clusters: int = 100):
        """
        构建倒排文件索引
        """
        print(f"Building IVF index with {n_clusters} clusters...")
        
        # 使用K-means聚类
        self.cluster_centers = self._kmeans_clustering(n_clusters)
        
        # 为每个向量分配最近的聚类中心
        self.cluster_assignments = self._assign_to_clusters()
        
        # 构建倒排列表
        self.inverted_lists = self._build_inverted_lists()
        
        self.is_built = True
        print("IVF index built successfully")
    
    def _kmeans_clustering(self, n_clusters: int, max_iter: int = 50) -> torch.Tensor:
        """
        K-means聚类（优化版本）
        """
        num_vectors = self.vectors.size(0)
        
        # 随机初始化聚类中心
        indices = torch.randperm(num_vectors)[:n_clusters]
        centers = self.vectors[indices].clone()
        
        for iteration in range(max_iter):
            # 计算每个向量到聚类中心的距离
            distances = torch.cdist(self.vectors, centers)
            assignments = torch.argmin(distances, dim=1)
            
            # 更新聚类中心
            new_centers = torch.zeros_like(centers)
            for i in range(n_clusters):
                cluster_mask = (assignments == i)
                if cluster_mask.sum() > 0:
                    new_centers[i] = self.vectors[cluster_mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]
            
            # 检查收敛
            if torch.allclose(centers, new_centers, atol=1e-6):
                break
                
            centers = new_centers
        
        return centers
    
    def _assign_to_clusters(self) -> torch.Tensor:
        """
        将向量分配到最近的聚类中心
        """
        distances = torch.cdist(self.vectors, self.cluster_centers)
        return torch.argmin(distances, dim=1)
    
    def _build_inverted_lists(self) -> List[torch.Tensor]:
        """
        构建倒排列表
        """
        inverted_lists = []
        for cluster_id in range(self.cluster_centers.size(0)):
            cluster_mask = (self.cluster_assignments == cluster_id)
            cluster_indices = torch.where(cluster_mask)[0]
            inverted_lists.append(cluster_indices)
        return inverted_lists
    
    def search_similar_vectors(self, queries: torch.Tensor, k: int, exclude_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        搜索相似向量（用于负样本采样）
        
        Args:
            queries: 查询向量 [num_queries, dimension]
            k: 返回的最近邻数量
            exclude_ids: 要排除的item_id [num_queries]
            
        Returns:
            distances: 距离矩阵 [num_queries, k]
            indices: 索引矩阵 [num_queries, k]
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        if self.index_type == 'flat':
            return self._flat_search_with_exclusion(queries, k, exclude_ids)
        elif self.index_type == 'ivf':
            return self._ivf_search_with_exclusion(queries, k, exclude_ids)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _flat_search_with_exclusion(self, queries: torch.Tensor, k: int, exclude_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        暴力搜索（带排除）
        """
        queries = queries.to(self.device)
        
        # 计算所有距离
        distances = torch.cdist(queries, self.vectors)
        
        # 如果有排除的ID，将对应距离设为很大值
        if exclude_ids is not None and self.vector_ids is not None:
            for i, exclude_id in enumerate(exclude_ids):
                if exclude_id != 0:  # 0通常表示padding
                    mask = (self.vector_ids == exclude_id)
                    distances[i, mask] = float('inf')
        
        # 找到最近的k个
        distances, indices = torch.topk(distances, k, dim=1, largest=False)
        
        return distances, indices
    
    def _ivf_search_with_exclusion(self, queries: torch.Tensor, k: int, exclude_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IVF索引搜索（带排除）
        """
        queries = queries.to(self.device)
        num_queries = queries.size(0)
        
        all_distances = []
        all_indices = []
        
        for i, query in enumerate(queries):
            # 找到最近的聚类中心
            cluster_distances = torch.cdist(query.unsqueeze(0), self.cluster_centers)
            nearest_cluster = torch.argmin(cluster_distances)
            
            # 在该聚类中搜索
            cluster_indices = self.inverted_lists[nearest_cluster]
            if len(cluster_indices) == 0:
                # 如果聚类为空，返回随机结果
                random_indices = torch.randperm(self.vectors.size(0))[:k]
                distances = torch.cdist(query.unsqueeze(0), self.vectors[random_indices])
                all_distances.append(distances.squeeze(0))
                all_indices.append(random_indices)
            else:
                cluster_vectors = self.vectors[cluster_indices]
                distances = torch.cdist(query.unsqueeze(0), cluster_vectors)
                
                # 如果有排除的ID，处理排除逻辑
                if exclude_ids is not None and self.vector_ids is not None:
                    exclude_id = exclude_ids[i]
                    if exclude_id != 0:
                        exclude_mask = (self.vector_ids[cluster_indices] == exclude_id)
                        distances[0, exclude_mask] = float('inf')
                
                # 找到最近的k个
                if len(cluster_indices) >= k:
                    distances, local_indices = torch.topk(distances, k, dim=1, largest=False)
                    global_indices = cluster_indices[local_indices.squeeze(0)]
                else:
                    # 如果聚类中的向量不够，用随机向量填充
                    remaining = k - len(cluster_indices)
                    random_indices = torch.randperm(self.vectors.size(0))[:remaining]
                    random_distances = torch.cdist(query.unsqueeze(0), self.vectors[random_indices])
                    
                    distances = torch.cat([distances.squeeze(0), random_distances.squeeze(0)])
                    global_indices = torch.cat([cluster_indices, random_indices])
                
                all_distances.append(distances)
                all_indices.append(global_indices)
        
        return torch.stack(all_distances), torch.stack(all_indices)
    
    def compute_similarity_matrix(self, queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        计算相似度矩阵（用于InfoNCE loss）
        
        Args:
            queries: 查询向量 [batch_size, seq_len, dimension]
            candidates: 候选向量 [batch_size, seq_len, dimension]
            
        Returns:
            similarity_matrix: 相似度矩阵 [batch_size, seq_len, batch_size * seq_len]
        """
        batch_size, seq_len, dim = queries.shape
        
        # 重塑为2D
        queries_2d = queries.view(-1, dim)  # [batch_size * seq_len, dim]
        candidates_2d = candidates.view(-1, dim)  # [batch_size * seq_len, dim]
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            queries_2d.unsqueeze(1),  # [batch_size * seq_len, 1, dim]
            candidates_2d.unsqueeze(0),  # [1, batch_size * seq_len, dim]
            dim=-1, eps=1e-8
        )  # [batch_size * seq_len, batch_size * seq_len]
        
        # 重塑回原始维度
        similarity = similarity.view(batch_size, seq_len, batch_size * seq_len)
        
        return similarity
    
    def get_negative_samples(self, pos_embeddings: torch.Tensor, pos_ids: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        获取负样本（用于Inbatch负采样优化）
        
        Args:
            pos_embeddings: 正样本嵌入 [batch_size, seq_len, dimension]
            pos_ids: 正样本ID [batch_size, seq_len]
            k: 每个位置返回的负样本数量
            
        Returns:
            neg_embeddings: 负样本嵌入 [batch_size, seq_len, k, dimension]
        """
        batch_size, seq_len, dim = pos_embeddings.shape
        
        # 搜索相似向量作为负样本
        pos_embeddings_2d = pos_embeddings.view(-1, dim)
        pos_ids_2d = pos_ids.view(-1)
        
        distances, indices = self.search_similar_vectors(pos_embeddings_2d, k, pos_ids_2d)
        
        # 获取负样本嵌入
        neg_embeddings = self.vectors[indices]  # [batch_size * seq_len, k, dimension]
        neg_embeddings = neg_embeddings.view(batch_size, seq_len, k, dim)
        
        return neg_embeddings


def create_recommender_faiss_index(model, dataset, device='cuda'):
    """
    为推荐系统创建FAISS索引
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        device: 设备类型
        
    Returns:
        faiss_index: FAISS索引
    """
    print("Creating FAISS index for recommendation system...")
    
    # 获取所有item的嵌入
    model.eval()
    all_embeddings = []
    all_item_ids = []
    
    with torch.no_grad():
        # 这里需要根据你的数据集结构来获取所有item
        # 假设我们有一个item到embedding的映射
        for item_id in range(1, dataset.itemnum + 1):
            # 创建虚拟的item特征（根据你的特征结构调整）
            item_feat = dataset.fill_missing_feat(dataset.item_feat_dict.get(str(item_id), {}), item_id)
            
            # 获取item嵌入（这里需要根据你的模型结构调整）
            # 假设有一个获取item嵌入的方法
            item_emb = model.get_item_embedding(item_feat)
            
            all_embeddings.append(item_emb)
            all_item_ids.append(item_id)
    
    # 转换为tensor
    all_embeddings = torch.stack(all_embeddings)
    all_item_ids = torch.tensor(all_item_ids)
    
    # 创建FAISS索引
    faiss_index = RecommenderFAISS(all_embeddings.size(-1), device)
    faiss_index.add_vectors(all_embeddings, all_item_ids)
    faiss_index.build_index('ivf', n_clusters=min(100, len(all_embeddings) // 10))
    
    print(f"FAISS index created with {len(all_embeddings)} items")
    return faiss_index


# 使用示例
if __name__ == "__main__":
    # 测试推荐系统FAISS
    print("=== Recommender FAISS Test ===")
    
    dimension = 128
    num_items = 1000
    batch_size = 32
    seq_len = 20
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模拟数据
    item_embeddings = torch.randn(num_items, dimension, device=device)
    item_ids = torch.arange(1, num_items + 1, device=device)
    
    # 创建FAISS索引
    faiss_index = RecommenderFAISS(dimension, device)
    faiss_index.add_vectors(item_embeddings, item_ids)
    faiss_index.build_index('ivf', n_clusters=50)
    
    # 测试负样本采样
    pos_embeddings = torch.randn(batch_size, seq_len, dimension, device=device)
    pos_ids = torch.randint(1, num_items + 1, (batch_size, seq_len), device=device)
    
    start_time = time.time()
    neg_embeddings = faiss_index.get_negative_samples(pos_embeddings, pos_ids, k=5)
    end_time = time.time()
    
    print(f"Negative sampling time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"Negative embeddings shape: {neg_embeddings.shape}")
    
    # 测试相似度计算
    start_time = time.time()
    similarity = faiss_index.compute_similarity_matrix(pos_embeddings, pos_embeddings)
    end_time = time.time()
    
    print(f"Similarity computation time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"Similarity matrix shape: {similarity.shape}")
