class GBDTClassifier:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_split=2):
        self.n_estimators = n_estimators           # 弱学习器数量
        self.max_depth = max_depth                 # 树的最大深度
        self.learning_rate = learning_rate         # 学习率（步长）
        self.min_samples_split = min_samples_split # 分裂所需的最小样本数
        self.trees = []                            # 存储每棵树
        self.init_pred = 0                         # 初始预测值（log odds）

    def _log_loss_gradient(self, y_true, y_pred):
        """计算 log loss 的负梯度（即 pseudo-residuals）"""
        return y_true - 1 / (1 + np.exp(-y_pred))

    def _fit_tree(self, X, residuals, depth=0, min_samples_split=2):
        """递归构造一棵回归树来拟合残差"""
        if len(X) < min_samples_split or depth >= self.max_depth or np.var(residuals) < 1e-6:
            return {'value': np.mean(residuals), 'is_leaf': True}

        best_gain = -np.inf
        best_split = None
        best_left_mask = None
        m, n = X.shape

        for feat_idx in range(n):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue

                gain = self._compute_gain(residuals, left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat_idx, thresh)
                    best_left_mask = left_mask

        if best_gain == -np.inf:
            return {'value': np.mean(residuals), 'is_leaf': True}

        feat_idx, thresh = best_split
        left_mask = best_left_mask

        left_tree = self._fit_tree(
            X[left_mask], residuals[left_mask], 
            depth + 1, min_samples_split
        )
        right_tree = self._fit_tree(
            X[~left_mask], residuals[~left_mask], 
            depth + 1, min_samples_split
        )

        return {
            'is_leaf': False,
            'feature': feat_idx,
            'threshold': thresh,
            'left': left_tree,
            'right': right_tree
        }

    def _compute_gain(self, residuals, left_mask):
        """计算分裂增益（基于方差减少）"""
        var_full = np.var(residuals) * len(residuals)
        var_left = np.var(residuals[left_mask]) * np.sum(left_mask)
        var_right = np.var(residuals[~left_mask]) * np.sum(~left_mask)
        return var_full - var_left - var_right

    def _predict_tree(self, tree, x):
        """单棵树预测"""
        if tree['is_leaf']:
            return tree['value']
        feat_idx = tree['feature']
        thresh = tree['threshold']
        if x[feat_idx] <= thresh:
            return self._predict_tree(tree['left'], x)
        else:
            return self._predict_tree(tree['right'], x)

    def fit(self, X, y):
        """训练 GBDT 模型"""
        X = np.array(X)
        y = np.array(y)
        
        # 初始化为 log(odds)，若无先验可设为 0
        self.init_pred = 0.0
        F = np.full(len(y), self.init_pred)

        self.trees = []

        for _ in range(self.n_estimators):
            # 1. 计算负梯度（伪残差）
            residuals = self._log_loss_gradient(y, F)

            # 2. 训练一棵回归树来拟合残差
            tree = self._fit_tree(X, residuals, depth=0, min_samples_split=self.min_samples_split)
            self.trees.append(tree)

            # 3. 预测所有样本的残差修正值
            updates = np.array([self._predict_tree(tree, x) for x in X])

            # 4. 更新当前模型输出（带学习率）
            F += self.learning_rate * updates

    def predict_proba(self, X):
        """预测购买概率"""
        X = np.array(X)
        # 初始预测 + 所有树的累加更新
        F = np.full(len(X), self.init_pred)
        for tree in self.trees:
            updates = np.array([self._predict_tree(tree, x) for x in X])
            F += self.learning_rate * updates
        # 转换为概率
        prob = 1 / (1 + np.exp(-F))
        return np.column_stack([1 - prob, prob])  # sklearn format

    def predict(self, X):
        """预测类别（0/1）"""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, classification_report

# Step 1: 生成模拟数据（连续特征 + purchase label）
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    flip_y=0.01,
    class_sep=1.0,
    random_state=42
)

# 特征全部是 float，符合“连续特征”要求
print(f"X shape: {X.shape}, y mean: {y.mean():.3f}")

# Step 2: 训练自定义 GBDT 模型
gbdt = GBDTClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    min_samples_split=10
)

gbdt.fit(X, y)

# Step 3: 预测
y_prob = gbdt.predict_proba(X)[:, 1]
y_pred = gbdt.predict(X)

# Step 4: 评估
auc = roc_auc_score(y, y_prob)
print(f"\n✅ 自定义 GBDT 模型训练完成！")
print(f"AUC: {auc:.4f}")
print("\n📋 分类报告:")
print(classification_report(y, y_pred, target_names=['未购买', '购买']))
