import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import glob
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from typing import List, Dict, Tuple, Optional
from mpl_toolkits.mplot3d import Axes3D  # 用于3D可视化
import pickle

# 设置字体显示（保留英文和中文字体以确保兼容性）
plt.rcParams["font.family"] = ["Arial", "SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 特征工程函数 - 从task1_energy_baseline.py整合

def pairwise_dist_sorted(coords: np.ndarray) -> np.ndarray:
    """计算并排序所有原子对之间的欧氏距离。"""
    # 计算所有原子对之间的距离
    dists = pdist(coords)
    # 排序
    dists_sorted = np.sort(dists)
    return dists_sorted

def coulomb_matrix(coords: np.ndarray, atomic_nums: np.ndarray) -> np.ndarray:
    """用原子坐标和原子序数构造库仑矩阵。"""
    n_atoms = len(coords)
    cm = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        # 对角线元素
        cm[i, i] = 0.5 * atomic_nums[i] ** 2.4
        # 非对角线元素
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > 0:
                cm[i, j] = (atomic_nums[i] * atomic_nums[j]) / dist
                cm[j, i] = cm[i, j]
    
    return cm

def coulomb_eigvals(cm: np.ndarray, n_atoms: int = 20) -> np.ndarray:
    """取库仑矩阵的特征值作为不变特征。"""
    # 计算特征值并排序（降序）
    eigvals = eigh(cm, eigvals_only=True)
    # 确保特征值数量为n_atoms（20）
    if len(eigvals) < n_atoms:
        # 填充零以达到固定长度
        padded_eigvals = np.zeros(n_atoms)
        padded_eigvals[:len(eigvals)] = np.sort(eigvals)[::-1]  # 降序排列
        return padded_eigvals
    else:
        # 降序排列并取前n_atoms个
        return np.sort(eigvals)[::-1][:n_atoms]

def featurize(coords: np.ndarray, atomic_nums: np.ndarray) -> np.ndarray:
    """组合原子间距离和库仑矩阵特征值形成特征向量。"""
    # 计算排序后的原子间距离（190维）
    dist_features = pairwise_dist_sorted(coords)
    
    # 计算库仑矩阵及其特征值（20维）
    cm = coulomb_matrix(coords, atomic_nums)
    cm_eigvals = coulomb_eigvals(cm)
    
    # 组合特征
    features = np.concatenate([dist_features, cm_eigvals])
    
    return features

class CyberCGCNDataProcessor:
    """用于处理GCN数据的处理器类"""
    def __init__(self, data_dir: str, distance_cutoff: float = 3.5):
        """初始化数据处理器
        
        Args:
            data_dir: 包含XYZ文件的目录路径
            distance_cutoff: 用于构建邻接矩阵的距离阈值（Å）
        """
        self.data_dir = data_dir
        self.distance_cutoff = distance_cutoff
        self.raw_data = []  # 存储原始数据
        self.train_data = []  # 训练集
        self.val_data = []  # 验证集
        self.test_data = []  # 测试集
        self.y_scaler = None  # 目标值标准化器
        
    def load_and_process_data(self):
        """加载并处理数据"""
        # 获取所有XYZ文件
        xyz_files = glob.glob(os.path.join(self.data_dir, '*.xyz'))
        
        for file_path in tqdm(xyz_files, desc="处理数据文件"):
            # 解析XYZ文件
            atoms, coords, energy = self._parse_xyz_file(file_path)
            
            # 提取原子序数（假设都是Au原子，原子序数为79）
            atomic_nums = np.array([79 for _ in atoms])
            
            # 构建图数据
            X = self._build_node_features(coords, atomic_nums)
            A = self._build_adjacency_matrix(coords)
            y = energy  # 目标值为能量
            
            # 添加到原始数据列表
            self.raw_data.append({'X': X, 'A': A, 'y': y})
        
    def _parse_xyz_file(self, file_path: str) -> Tuple[List[str], np.ndarray, float]:
        """解析XYZ文件，提取原子类型、坐标和能量
        
        Args:
            file_path: XYZ文件路径
        
        Returns:
            原子类型列表、坐标数组和能量值
        """
        atoms = []
        coords = []
        energy = 0.0
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # 第二行通常包含能量信息
            if len(lines) > 1:
                energy_line = lines[1].strip()
                try:
                    # 尝试从第二行提取能量值
                    energy = float(energy_line.split()[0])
                except (ValueError, IndexError):
                    # 如果无法提取，使用0.0
                    energy = 0.0
            
            # 读取原子和坐标信息
            for line in lines[2:]:  # 跳过前两行
                parts = line.strip().split()
                if len(parts) >= 4:
                    atom_type = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append(atom_type)
                    coords.append([x, y, z])
        
        return atoms, np.array(coords), energy
    
    def _build_node_features(self, coords: np.ndarray, atomic_nums: np.ndarray) -> np.ndarray:
        """构建节点特征矩阵
        
        Args:
            coords: 原子坐标数组，形状为 [num_nodes, 3]
            atomic_nums: 原子序数数组，形状为 [num_nodes]
        
        Returns:
            节点特征矩阵，形状为 [num_nodes, in_features]
        """
        num_nodes = len(coords)
        
        # 计算全局特征（原子间距离和库仑矩阵特征值）
        global_features = featurize(coords, atomic_nums)
        
        # 为每个节点创建特征向量
        # 基础特征：坐标 + 原子序数（扩展为3维）
        node_features = []
        for i in range(num_nodes):
            # 节点的坐标特征
            coord_feature = coords[i]
            # 节点的原子序数特征（扩展为3维）
            atomic_num_feature = np.array([atomic_nums[i], atomic_nums[i], atomic_nums[i]])
            # 组合特征
            node_feature = np.concatenate([coord_feature, atomic_num_feature])
            node_features.append(node_feature)
        
        # 添加全局特征作为额外的节点特征
        node_features_array = np.array(node_features)
        
        return node_features_array
    
    def _build_adjacency_matrix(self, coords: np.ndarray) -> np.ndarray:
        """构建邻接矩阵
        
        Args:
            coords: 原子坐标数组，形状为 [num_nodes, 3]
        
        Returns:
            邻接矩阵，形状为 [num_nodes, num_nodes]
        """
        num_nodes = len(coords)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # 计算所有原子对之间的距离
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(coords[i] - coords[j])
                # 如果距离小于阈值，添加边
                if distance < self.distance_cutoff:
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
        
        # 添加自环
        np.fill_diagonal(adj_matrix, 1.0)
        
        return adj_matrix
    
    def split_dataset(self, train_size: int = 700, val_size: int = 150):
        """划分数据集
        
        Args:
            train_size: 训练集大小
            val_size: 验证集大小
        """
        # 随机打乱数据
        np.random.shuffle(self.raw_data)
        
        # 划分训练集、验证集和测试集
        self.train_data = self.raw_data[:train_size]
        self.val_data = self.raw_data[train_size:train_size + val_size]
        self.test_data = self.raw_data[train_size + val_size:]
        
    def standardize_data(self):
        """标准化目标值"""
        # 提取所有目标值
        all_targets = np.array([item['y'] for item in self.raw_data])
        
        # 创建并拟合标准化器
        self.y_scaler = StandardScaler()
        self.y_scaler.fit(all_targets.reshape(-1, 1))
        
        # 标准化训练集、验证集和测试集的目标值
        for data_split in [self.train_data, self.val_data, self.test_data]:
            for item in data_split:
                item['y'] = self.y_scaler.transform(np.array([[item['y']]]))[0][0]
    
    def save_processed_data(self, output_dir: str):
        """保存处理后的数据
        
        Args:
            output_dir: 输出目录路径
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存训练集、验证集和测试集
        np.savez(os.path.join(output_dir, 'train_data.npz'), data=self.train_data)
        np.savez(os.path.join(output_dir, 'val_data.npz'), data=self.val_data)
        np.savez(os.path.join(output_dir, 'test_data.npz'), data=self.test_data)
        
        # 保存标准化器
        if self.y_scaler is not None:
            np.save(os.path.join(output_dir, 'y_scaler_mean.npy'), self.y_scaler.mean_)
            np.save(os.path.join(output_dir, 'y_scaler_scale.npy'), self.y_scaler.scale_)

class GCNLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_features: int, out_features: int):
        """初始化图卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
        """
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 节点特征矩阵，形状为 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵，形状为 [batch_size, num_nodes, num_nodes]
        
        Returns:
            输出特征矩阵，形状为 [batch_size, num_nodes, out_features]
        """
        # 计算图卷积
        # 先对邻接矩阵进行归一化
        batch_size, num_nodes, _ = adj.size()
        
        # 添加自环（已经在构建邻接矩阵时添加了，这里可以省略）
        # adj = adj + torch.eye(num_nodes).to(adj.device)
        
        # 计算度矩阵的逆平方根
        degree = torch.sum(adj, dim=2)  # 计算每个节点的度
        degree_inv_sqrt = torch.pow(degree, -0.5)  # 度矩阵的逆平方根
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 处理零除情况
        degree_matrix = torch.diag_embed(degree_inv_sqrt)  # 转换为对角矩阵
        
        # 计算归一化的邻接矩阵：D^(-1/2) * A * D^(-1/2)
        adj_normalized = torch.bmm(torch.bmm(degree_matrix, adj), degree_matrix)
        
        # 应用图卷积：A * X * W
        support = self.linear(x)
        output = torch.bmm(adj_normalized, support)
        
        return output

class GCNRegressionModel(nn.Module):
    """用于回归任务的图卷积网络模型 - 支持多层GCN"""
    def __init__(self, in_features: int = 6, hidden_dims: list = None, 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        """初始化GCN回归模型
        
        Args:
            in_features: 输入特征维度
            hidden_dims: 各隐藏层的特征维度列表，默认为[64, 32]（2层GCN）
            dropout_rate: Dropout层的丢弃率
            use_batch_norm: 是否使用批归一化
        """
        super(GCNRegressionModel, self).__init__()
        
        # 默认隐藏层配置改为更简单的2层GCN，且维度更小
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        # 创建GCN层
        self.gcn_layers = nn.ModuleList()
        # 第一层GCN
        self.gcn_layers.append(GCNLayer(in_features, hidden_dims[0]))
        
        # 添加更多GCN层
        for i in range(1, len(hidden_dims)):
            self.gcn_layers.append(GCNLayer(hidden_dims[i-1], hidden_dims[i]))
        
        # 创建BatchNorm层（如果启用）
        self.bn_layers = None
        if use_batch_norm:
            self.bn_layers = nn.ModuleList()
            for dim in hidden_dims:
                # 添加track_running_stats=False以提高小样本训练稳定性
                self.bn_layers.append(nn.BatchNorm1d(dim, track_running_stats=False))
        
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接回归层 - 添加更多的隐藏单元
        self.regression_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 使用normal而非uniform，可能更稳定
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 节点特征矩阵，形状为 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵，形状为 [batch_size, num_nodes, num_nodes]
        
        Returns:
            预测的能量值，形状为 [batch_size, 1]
        """
        batch_size, num_nodes, _ = x.size()
        
        # 逐层通过GCN
        for i, gcn_layer in enumerate(self.gcn_layers):
            # 图卷积操作
            x = gcn_layer(x, adj)
            
            # 应用BatchNorm（如果启用）
            if self.use_batch_norm:
                # 调整形状以适应BatchNorm1d: [batch_size * num_nodes, hidden_dim]
                x_flat = x.view(-1, self.hidden_dims[i])
                x_flat = self.bn_layers[i](x_flat)
                # 恢复原始形状
                x = x_flat.view(batch_size, num_nodes, self.hidden_dims[i])
            
            # 除最后一层外，应用ReLU激活和Dropout
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # 对所有节点取平均
        
        # 全连接回归层输出预测能量
        x = self.regression_layer(x)
        
        return x

class GraphDataset(torch.utils.data.Dataset):
    """用于加载和处理图数据的Dataset类"""
    def __init__(self, data_list: List[Dict]):
        """初始化数据集
        
        Args:
            data_list: 包含图数据的列表，每个元素是一个字典，包含'X', 'A', 'y'
        """
        self.data_list = data_list
    
    def __len__(self):
        """返回数据集的大小"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取指定索引的数据样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含节点特征、邻接矩阵和目标值的元组
        """
        graph = self.data_list[idx]
        
        # 将numpy数组转换为PyTorch张量
        X = torch.FloatTensor(graph['X'])
        A = torch.FloatTensor(graph['A'])
        y = torch.FloatTensor([graph['y']])
        
        return X, A, y

class AugmentedGraphDataset(torch.utils.data.Dataset):
    """包含数据增强功能的图数据集类"""
    def __init__(self, data_list: List[Dict], augment_prob: float = 0.5):
        """初始化增强数据集
        
        Args:
            data_list: 包含图数据的列表
            augment_prob: 应用数据增强的概率
        """
        self.data_list = data_list
        self.augment_prob = augment_prob
    
    def __len__(self):
        """返回数据集的大小"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取指定索引的数据样本并应用数据增强
        
        Args:
            idx: 样本索引
        
        Returns:
            包含节点特征、邻接矩阵和目标值的元组
        """
        data = self.data_list[idx]
        X, A, y = data['X'], data['A'], data['y']
        
        # 随机旋转（数据增强）
        if np.random.random() < self.augment_prob:
            # 随机生成旋转角度
            theta = np.random.uniform(0, 2*np.pi)
            # 构建绕z轴的旋转矩阵
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            # 对坐标特征应用旋转
            X_rotated = X.copy()
            # 假设前3列是坐标
            X_rotated[:, :3] = X_rotated[:, :3].dot(rotation_matrix)
            X = X_rotated
        
        return torch.tensor(X, dtype=torch.float32), \
               torch.tensor(A, dtype=torch.float32), \
               torch.tensor([y], dtype=torch.float32)

class EarlyStopping:
    """早停机制实现，用于防止模型过拟合"""
    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0, path: str = 'checkpoint.pt'):
        """初始化早停器
        
        Args:
            patience: 验证损失不再改善时等待的epochs数
            verbose: 是否打印详细信息
            delta: 损失改善的最小幅度
            path: 保存最佳模型的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss: float, model: nn.Module):
        """每个epoch结束时调用此方法
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """保存模型检查点
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 修改train_model函数，添加L1正则化项
def train_model(model: nn.Module, train_loader: Data.DataLoader, val_loader: Data.DataLoader, 
                criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
                num_epochs: int, patience: int, checkpoint_path: str, y_scaler: StandardScaler, 
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, l1_lambda: float = 0.0001):
    """训练GCN模型
    
    Args:
        model: GCN回归模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备 (CPU或GPU)
        num_epochs: 最大训练轮数
        patience: 早停耐心值
        checkpoint_path: 最佳模型保存路径
        y_scaler: 目标值标准化器，用于反标准化以便于监控训练过程
        scheduler: 学习率调度器
        l1_lambda: L1正则化系数
    
    Returns:
        训练好的模型和损失曲线
    """
    # 初始化早停器
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    
    # 开始训练循环
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        # 遍历训练数据
        for X, A, y in train_loader:
            # 将数据移至指定设备
            X, A, y = X.to(device), A.to(device), y.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(X, A)
            
            # 计算损失
            mse_loss = criterion(outputs, y.view(-1, 1))
            
            # 添加L1正则化
            l1_reg = torch.tensor(0., requires_grad=True, device=device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            
            loss = mse_loss + l1_lambda * l1_reg
            
            # 反向传播
            loss.backward()
            
            # 添加梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 累计训练损失
            train_loss += loss.item() * X.size(0)
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        # 禁用梯度计算以加速验证过程
        with torch.no_grad():
            for X, A, y in val_loader:
                # 将数据移至指定设备
                X, A, y = X.to(device), A.to(device), y.to(device)
                
                # 前向传播
                outputs = model(X, A)
                
                # 计算损失
                loss = criterion(outputs, y.view(-1, 1))
                
                # 累计验证损失
                val_loss += loss.item() * X.size(0)
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 反标准化损失以便于解释（如果提供了标准化器）
        if y_scaler is not None:
            # 计算反标准化后的训练损失和验证损失
            # 注意：这里我们使用近似值，因为MSE在标准化空间和原始空间的转换不是线性的
            train_loss_original = train_loss * (y_scaler.scale_[0] ** 2)
            val_loss_original = val_loss * (y_scaler.scale_[0] ** 2)
            # 修复：将numpy数组转换为标量值后再格式化输出
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_original:.6f}, Val Loss: {val_loss_original:.6f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model)
        
        # 如果触发早停，结束训练
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # 加载最佳模型权重
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model, train_losses, val_losses

def evaluate_model(model: nn.Module, test_loader: Data.DataLoader, 
                   device: torch.device, y_scaler: StandardScaler) -> Dict:
    """评估模型在测试集上的性能
    
    Args:
        model: 训练好的GCN回归模型
        test_loader: 测试数据加载器
        device: 评估设备 (CPU或GPU)
        y_scaler: 目标值标准化器，用于反标准化预测值和真实值
    
    Returns:
        包含评估指标的字典，预测值和真实值
    """
    model.eval()
    
    # 存储所有预测值和真实值
    all_predictions = []
    all_true_values = []
    
    # 禁用梯度计算以加速评估过程
    with torch.no_grad():
        for X, A, y in test_loader:
            # 将数据移至指定设备
            X, A, y = X.to(device), A.to(device), y.to(device)
            
            # 前向传播
            outputs = model(X, A)
            
            # 将预测值和真实值添加到列表中
            all_predictions.extend(outputs.cpu().numpy())
            all_true_values.extend(y.cpu().numpy())
    
    # 将列表转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_true_values = np.array(all_true_values)
    
    # 反标准化预测值和真实值
    all_predictions_original = y_scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    all_true_values_original = y_scaler.inverse_transform(all_true_values.reshape(-1, 1)).flatten()
    
    # 计算评估指标
    mse = np.mean((all_true_values_original - all_predictions_original) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true_values_original, all_predictions_original)
    r2 = r2_score(all_true_values_original, all_predictions_original)
    
    # 计算Bias和一致性界限(LOA)
    differences = all_predictions_original - all_true_values_original
    bias = np.mean(differences)
    std_diff = np.std(differences)
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff
    
    # 打印评估指标
    print(f"评估指标:")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"决定系数 (R² Score): {r2:.6f}")
    print(f"偏差 (Bias): {bias:.6f}")
    print(f"一致性界限 (LOA): {loa_lower:.6f} 到 {loa_upper:.6f}")
    print(f"Bias±LOA: {bias:.6f} ± {1.96 * std_diff:.6f}")
    
    # 返回评估指标和预测值、真实值
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
        'predictions': all_predictions_original,
        'true_values': all_true_values_original
    }

# 添加可视化函数
def plot_loss_curves(train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None, smooth_window: int = 5):
    """绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 图像保存路径，None表示不保存
        smooth_window: 平滑窗口大小
    """
    # 添加损失平滑
    def smooth_curve(points, factor=0.8):
        smoothed = []
        for point in points:
            if smoothed:
                previous = smoothed[-1]
                smoothed.append(previous * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed
    
    # 平滑损失曲线
    if len(train_losses) > smooth_window:
        train_losses_smooth = smooth_curve(train_losses)
        val_losses_smooth = smooth_curve(val_losses)
    else:
        train_losses_smooth = train_losses
        val_losses_smooth = val_losses
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Original Training Loss', alpha=0.3)
    plt.plot(val_losses, label='Original Validation Loss', alpha=0.3)
    plt.plot(train_losses_smooth, label='Smoothed Training Loss')
    plt.plot(val_losses_smooth, label='Smoothed Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_vs_true(predictions: np.ndarray, true_values: np.ndarray, 
                             metrics: Dict, save_path: Optional[str] = None):
    """绘制预测值vs真实值散点图
    
    Args:
        predictions: 模型预测值
        true_values: 真实值
        metrics: 评估指标字典
        save_path: 图像保存路径，None表示不保存
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    plt.scatter(true_values, predictions, alpha=0.6, edgecolors='w', s=60)
    
    # 绘制理想线 y=x
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal Line')
    
    # 添加评估指标文本
    metrics_text = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}\nMAE = {metrics['mae']:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title('Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(predictions: np.ndarray, true_values: np.ndarray, save_path: Optional[str] = None):
    """绘制残差图
    
    Args:
        predictions: 模型预测值
        true_values: 真实值
        save_path: 图像保存路径，None表示不保存
    """
    residuals = true_values - predictions
    
    plt.figure(figsize=(10, 6))
    
    # 绘制残差散点图
    plt.scatter(true_values, residuals, alpha=0.6, edgecolors='w', s=60)
    
    # 绘制水平线 y=0
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    
    # 添加统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.text(0.05, 0.95, f"Mean Residual = {mean_residual:.4f}\nResidual Std = {std_residual:.4f}", 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title('Residual Plot')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_distribution(predictions: np.ndarray, true_values: np.ndarray, save_path: Optional[str] = None):
    """绘制残差分布图
    
    Args:
        predictions: 模型预测值
        true_values: 真实值
        save_path: 图像保存路径，None表示不保存
    """
    residuals = true_values - predictions
    
    plt.figure(figsize=(10, 6))
    
    # 绘制残差直方图和核密度估计
    sns.histplot(residuals, kde=True, bins=30, alpha=0.7)
    
    # 添加统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.text(0.05, 0.95, f"Mean Residual = {mean_residual:.4f}\nResidual Std = {std_residual:.4f}", 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bland_altman(predictions: np.ndarray, true_values: np.ndarray, save_path: Optional[str] = None):
    """绘制Bland-Altman图，显示预测值和真实值之间的偏差和一致性界限
    
    Args:
        predictions: 模型预测值
        true_values: 真实值
        save_path: 图像保存路径，None表示不保存
    """
    # 计算平均值和差值
    mean_values = (predictions + true_values) / 2
    differences = predictions - true_values
    
    # 计算偏差(Bias)和一致性界限(LOA)
    bias = np.mean(differences)
    std_diff = np.std(differences)
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff
    
    # 创建Bland-Altman图
    plt.figure(figsize=(10, 8))
    plt.scatter(mean_values, differences, alpha=0.6, edgecolors='w', s=60)
    
    # 绘制偏差线和一致性界限
    plt.axhline(y=bias, color='r', linestyle='-', label=f'Bias: {bias:.4f}')
    plt.axhline(y=loa_upper, color='g', linestyle='--', label=f'LOA Upper: {loa_upper:.4f}')
    plt.axhline(y=loa_lower, color='g', linestyle='--', label=f'LOA Lower: {loa_lower:.4f}')
    
    # 添加统计信息文本框
    stats_text = f"Bias = {bias:.4f}\n" + \
                f"SD of differences = {std_diff:.4f}\n" + \
                f"95% LOA = Bias ± 1.96×SD\n" + \
                f"          = {loa_lower:.4f} to {loa_upper:.4f}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title('Bland-Altman Plot')
    plt.xlabel('Average of Predictions and True Values')
    plt.ylabel('Difference (Predictions - True Values)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 添加数据输入可视化函数
class DataVisualizer:
    """用于可视化数据输入的类"""
    def __init__(self, processor, output_dir):
        """初始化可视化器
        
        Args:
            processor: CyberCGCNDataProcessor实例
            output_dir: 图像保存目录
        """
        self.processor = processor
        self.output_dir = output_dir
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def visualize_data_distribution(self):
        """可视化能量值的分布情况"""
        # 提取所有能量值
        all_energies = np.array([item['y'] for item in self.processor.raw_data])
        
        plt.figure(figsize=(12, 6))
        
        # 绘制直方图
        plt.subplot(1, 2, 1)
        sns.histplot(all_energies, kde=True, bins=30, alpha=0.7)
        plt.title('Energy Distribution Histogram')
        plt.xlabel('Energy Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制箱线图
        plt.subplot(1, 2, 2)
        sns.boxplot(y=all_energies)
        plt.title('Energy Distribution Boxplot')
        plt.ylabel('Energy Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'energy_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Energy distribution visualization saved to: {save_path}")
    
    def visualize_atomic_structure(self, sample_idx=0, show_edges=True):
        """可视化原子的3D结构
        
        Args:
            sample_idx: 要可视化的样本索引
            show_edges: 是否显示原子间连接
        """
        if sample_idx >= len(self.processor.raw_data):
            print(f"Sample index out of range, there are {len(self.processor.raw_data)} samples")
            return
        
        # 获取样本数据
        sample = self.processor.raw_data[sample_idx]
        X = sample['X']
        A = sample['A']
        y = sample['y']
        
        # 提取坐标（前3个特征是坐标）
        coords = X[:, :3]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原子
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                            s=500, c='gold', alpha=0.8, edgecolors='k')
        
        # 绘制连接边
        if show_edges:
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    if A[i, j] > 0:
                        ax.plot([coords[i, 0], coords[j, 0]], 
                                [coords[i, 1], coords[j, 1]], 
                                [coords[i, 2], coords[j, 2]], 
                                'k-', alpha=0.5, linewidth=1)
        
        # 添加标签和标题
        ax.set_title(f'Atomic Structure Visualization (Sample {sample_idx}, Energy: {y:.4f})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        
        # 设置坐标轴范围相同，使图形比例一致
        max_range = np.array([coords[:, 0].max()-coords[:, 0].min(), 
                             coords[:, 1].max()-coords[:, 1].min(), 
                             coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
        
        mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
        mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
        mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        save_path = os.path.join(self.output_dir, f'atomic_structure_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Atomic structure visualization saved to: {save_path}")
    
    def visualize_feature_distribution(self, sample_idx=0):
        """可视化节点特征的分布情况
        
        Args:
            sample_idx: 要可视化的样本索引
        """
        if sample_idx >= len(self.processor.raw_data):
            print(f"Sample index out of range, there are {len(self.processor.raw_data)} samples")
            return
        
        # 获取样本数据
        sample = self.processor.raw_data[sample_idx]
        X = sample['X']
        
        # 创建特征分布图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Node Feature Distribution', fontsize=16)
        
        # 坐标特征 (前3个特征)和原子序数特征
        features = ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 'Atomic Number 1', 'Atomic Number 2', 'Atomic Number 3']
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        for i, (feature, color) in enumerate(zip(features, colors)):
            row = i // 3
            col = i % 3
            
            sns.histplot(X[:, i], ax=axes[row, col], color=color, bins=20, kde=True)
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.output_dir, f'feature_distribution_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature distribution visualization saved to: {save_path}")
    
    def visualize_dataset_split(self):
        """可视化数据集划分情况"""
        # 获取各数据集的能量值
        train_energies = np.array([item['y'] for item in self.processor.train_data])
        val_energies = np.array([item['y'] for item in self.processor.val_data])
        test_energies = np.array([item['y'] for item in self.processor.test_data])
        
        # 创建数据集划分可视化
        plt.figure(figsize=(15, 8))
        
        # 绘制小提琴图
        data = [train_energies, val_energies, test_energies]
        labels = ['Training Set', 'Validation Set', 'Test Set']
        
        sns.violinplot(data=data)
        plt.xticks(range(len(labels)), labels)
        plt.title('Dataset Split Visualization')
        plt.ylabel('Energy Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加样本数量信息
        for i, (dataset, label) in enumerate(zip(data, labels)):
            plt.text(i, np.max(dataset) * 1.05, f'n={len(dataset)}', 
                     horizontalalignment='center', fontsize=12)
        
        save_path = os.path.join(self.output_dir, 'dataset_split.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dataset split visualization saved to: {save_path}")
    
    def visualize_all(self, num_samples=3):
        """可视化所有数据相关内容
        
        Args:
            num_samples: 要可视化的样本数量
        """
        # 可视化数据分布
        self.visualize_data_distribution()
        
        # 可视化数据集划分
        self.visualize_dataset_split()
        
        # 可视化多个样本的原子结构和特征分布
        for i in range(min(num_samples, len(self.processor.raw_data))):
            self.visualize_atomic_structure(sample_idx=i)
            self.visualize_feature_distribution(sample_idx=i)

# 主函数，整合数据预处理、模型创建、训练和评估流程
def main():
    # 设置数据目录和输出目录
    # 修复：使用原始字符串表示路径，避免Unicode转义错误
    data_dir = r"c:\Users\hp\Desktop\CyberC\data (1)\data\Au20_OPT_1000"
    output_dir = r"c:\Users\hp\Desktop\CyberC\processed_data"
    model_dir = r"c:\Users\hp\Desktop\CyberC\models"
    plots_dir = r"c:\Users\hp\Desktop\CyberC\plots"
    data_viz_dir = r"c:\Users\hp\Desktop\CyberC\data_visualization"  # 新增数据可视化目录
    
    # 创建目录
    for directory in [output_dir, model_dir, plots_dir, data_viz_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 创建数据处理器实例
    processor = CyberCGCNDataProcessor(
        data_dir=data_dir,
        distance_cutoff=3.5  # 使用3.5 Å作为距离阈值
    )
    
    # 加载并处理数据
    processor.load_and_process_data()
    
    # 划分数据集
    processor.split_dataset(train_size=700, val_size=150)
    
    # 标准化数据
    processor.standardize_data()
    
    # 保存处理后的数据
    processor.save_processed_data(output_dir)
    
    print("数据预处理完成！")
    
    # 添加数据输入可视化
    print("开始数据输入可视化...")
    visualizer = DataVisualizer(processor, data_viz_dir)
    visualizer.visualize_all(num_samples=3)  # 可视化前3个样本
    print("数据输入可视化完成！")
    
    # 创建PyTorch数据集和数据加载器
    
    # 修改数据增强策略，降低增强概率
    train_dataset = AugmentedGraphDataset(processor.train_data, augment_prob=0.3)  # 降低增强概率
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,  # 增大批次大小
        shuffle=True
    )
    
    # 验证数据集和数据加载器
    val_dataset = GraphDataset(processor.val_data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,  # 增大批次大小
        shuffle=False
    )
    
    # 测试数据集和数据加载器
    test_dataset = GraphDataset(processor.test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,  # 增大批次大小
        shuffle=False
    )
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建更稳定的GCN模型
    model = GCNRegressionModel(
        in_features=6,
        hidden_dims=[64, 32],  # 更小的隐藏层维度
        dropout_rate=0.1,      # 进一步降低Dropout率
        use_batch_norm=True
    ).to(device)
    
    # 使用SGD+动量代替Adam，有时候更稳定
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-3)  # 更小的学习率，更强的正则化
    
    # 改进学习率调度器配置
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=10,   # 进一步增加耐心值
        verbose=True
    )
    
    # 调整训练参数
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=300,      # 增加训练轮数以补偿更小的学习率
        patience=25,         # 增加早停耐心值
        checkpoint_path=os.path.join(model_dir, 'best_model.pt'),
        y_scaler=processor.y_scaler,
        scheduler=scheduler
    )
    
    # 生成训练和验证损失曲线
    plot_loss_curves(
        train_losses, 
        val_losses, 
        save_path=os.path.join(plots_dir, 'loss_curves.png'),
        smooth_window=5
    )
    
    # 评估模型
    print("\n评估模型在测试集上的性能...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        y_scaler=processor.y_scaler
    )
    
    # 生成预测值vs真实值散点图
    plot_predictions_vs_true(
        metrics['predictions'], 
        metrics['true_values'],
        metrics, 
        save_path=os.path.join(plots_dir, 'predictions_vs_true.png')
    )
    
    # 生成残差图
    plot_residuals(
        metrics['predictions'], 
        metrics['true_values'], 
        save_path=os.path.join(plots_dir, 'residuals.png')
    )
    
    # 生成残差分布图
    plot_residual_distribution(
        metrics['predictions'], 
        metrics['true_values'], 
        save_path=os.path.join(plots_dir, 'residual_distribution.png')
    )

if __name__ == "__main__":
    main()