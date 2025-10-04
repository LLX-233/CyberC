import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import glob
import os  # 添加os模块导入
from scipy.spatial.distance import pdist
from scipy import stats  # 导入scipy.stats模块用于t检验

# 设置字体，优先使用英文，保留中文字体以确保兼容性
plt.rcParams["font.family"] = ["Arial", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# -----------------------------#
# 模型定义（从CyberC_2.0.py复制）
# -----------------------------#
class GCNLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adj.size()
        degree = torch.sum(adj, dim=2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_matrix = torch.diag_embed(degree_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(degree_matrix, adj), degree_matrix)
        support = self.linear(x)
        output = torch.bmm(adj_normalized, support)
        return output

class GCNRegressionModel(nn.Module):
    """用于回归任务的图卷积网络模型"""
    def __init__(self, in_features: int = 6, hidden_dims: list = None, 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super(GCNRegressionModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(in_features, hidden_dims[0]))
        
        for i in range(1, len(hidden_dims)):
            self.gcn_layers.append(GCNLayer(hidden_dims[i-1], hidden_dims[i]))
        
        self.bn_layers = None
        if use_batch_norm:
            self.bn_layers = nn.ModuleList()
            for dim in hidden_dims:
                self.bn_layers.append(nn.BatchNorm1d(dim, track_running_stats=False))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.regression_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.size()
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            
            if self.use_batch_norm:
                x_flat = x.view(-1, self.hidden_dims[i])
                x_flat = self.bn_layers[i](x_flat)
                x = x_flat.view(batch_size, num_nodes, self.hidden_dims[i])
            
            if i < len(self.gcn_layers) - 1:
                x = nn.functional.relu(x)
                x = self.dropout(x)
        
        x = torch.mean(x, dim=1)
        x = self.regression_layer(x)
        
        return x

# -----------------------------#
# 工具函数
# -----------------------------#
def load_xyz(file_path: str) -> tuple:
    """加载单个.xyz文件"""
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].strip())
    try:
        energy = float(lines[1].strip())
    except Exception:
        toks = lines[1].split()
        energy = float(toks[-1])
    coords = []
    for i in range(2, 2 + n):
        parts = lines[i].split()
        if len(parts) == 3:
            x, y, z = map(float, parts)
        else:
            x, y, z = map(float, parts[-3:])
        coords.append([x, y, z])
    arr = np.array(coords, dtype=float)
    return energy, arr

def center_coords(coords: np.ndarray) -> np.ndarray:
    """将坐标中心化到质心"""
    cen = coords.mean(axis=0, keepdims=True)
    return coords - cen

def build_node_features(coords: np.ndarray) -> np.ndarray:
    """构建节点特征矩阵"""
    num_nodes = len(coords)
    node_features = []
    atomic_num = 79.0  # Au的原子序数
    
    for i in range(num_nodes):
        coord_feature = coords[i]
        atomic_num_feature = np.array([atomic_num, atomic_num, atomic_num])
        node_feature = np.concatenate([coord_feature, atomic_num_feature])
        node_features.append(node_feature)
    
    return np.array(node_features)

def build_adjacency_matrix(coords: np.ndarray, distance_cutoff: float = 3.5) -> np.ndarray:
    """构建邻接矩阵"""
    num_nodes = len(coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance < distance_cutoff:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
    
    np.fill_diagonal(adj_matrix, 1.0)  # 添加自环
    return adj_matrix

def apply_perturbation(coords: np.ndarray, perturbation_std: float, num_atoms_to_perturb: int = 3) -> tuple:
    """对指定数量的原子应用随机扰动"""
    perturbed_coords = coords.copy()
    
    # 随机选择要扰动的原子
    n_atoms = coords.shape[0]
    atom_indices = np.random.choice(n_atoms, size=min(num_atoms_to_perturb, n_atoms), replace=False)
    
    # 应用高斯随机扰动
    perturbations = np.random.normal(0, perturbation_std, size=(len(atom_indices), 3))
    perturbed_coords[atom_indices] += perturbations
    
    # 计算扰动大小（被扰动原子的平均位移）
    perturbation_magnitude = np.mean(np.linalg.norm(perturbations, axis=1))
    
    return perturbed_coords, perturbation_magnitude

def calculate_stability_metric(original_energy, perturbed_energies, perturbation_magnitudes):
    """计算结构稳定性指标"""
    # 计算每个扰动的能量变化
    energy_changes = np.abs(perturbed_energies - original_energy)
    
    # 稳定性指标：能量变化除以扰动大小（平均）
    stability_metrics = energy_changes / (perturbation_magnitudes + 1e-10)  # 添加小值避免除零
    
    # 整体稳定性指标：平均稳定性指标的倒数（值越大表示越稳定）
    overall_stability = np.mean(1.0 / (stability_metrics + 1e-10))
    
    return stability_metrics, overall_stability

def visualize_structure_with_perturbation(original_coords, perturbed_coords, perturbation_indices, title, save_path):
    """可视化原始结构和扰动结构"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    
    original_centered = center_coords(original_coords)
    perturbed_centered = center_coords(perturbed_coords)
    
    fig = plt.figure(figsize=(12, 6))
    
    # 原始结构
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_centered[:,0], original_centered[:,1], original_centered[:,2], 
               s=40, c="blue", alpha=0.9, edgecolor="k")
    ax1.set_title("Original Structure")
    
    # 扰动结构
    ax2 = fig.add_subplot(122, projection='3d')
    colors = ["orange" if i in perturbation_indices else "blue" for i in range(len(perturbed_centered))]
    ax2.scatter(perturbed_centered[:,0], perturbed_centered[:,1], perturbed_centered[:,2], 
               s=40, c=colors, alpha=0.9, edgecolor="k")
    ax2.set_title("Perturbed Structure (orange = perturbed atoms)")
    
    # 设置等比例
    for ax in [ax1, ax2]:
        max_range = 0
        for axis in 'xyz':
            max_range = max(max_range, getattr(ax, f"get_{axis}lim")()[1] - getattr(ax, f"get_{axis}lim")()[0])
        mid_range = (getattr(ax, "get_xlim")()[0] + getattr(ax, "get_xlim")()[1]) / 2
        for axis in 'xyz':
            getattr(ax, f"set_{axis}lim")(mid_range - max_range/2, mid_range + max_range/2)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()

# -----------------------------#
# 主函数
# -----------------------------#
def main():
    # 设置文件路径
    project_root = r"c:\Users\hp\Desktop\CyberC"  # 修改为原始字符串
    data_dir = os.path.join(project_root, "data (1)", "data", "Au20_OPT_1000")
    model_path = os.path.join(project_root, "models", "best_model.pt")
    scalers_path = os.path.join(project_root, "processed_data", "scalers.pkl")
    output_dir = os.path.join(project_root, "reports_task3")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载最低能量结构（从任务二结果）
    # 首先找到能量最低的xyz文件
    xyz_files = glob.glob(os.path.join(data_dir, "*.xyz"))
    energies = []
    coords_map = {}
    
    for file_path in xyz_files:
        try:
            energy, coords = load_xyz(file_path)
            energies.append((file_path, energy))
            coords_map[file_path] = coords
        except Exception as e:
            print(f"[Warning] Skipping {file_path}: {e}")
    
    # 按能量排序，找到最低能量结构
    energies.sort(key=lambda x: x[1])
    best_file, best_energy = energies[0]
    best_coords = coords_map[best_file]
    
    print(f"Lowest energy structure: {os.path.basename(best_file)}, energy={best_energy:.6f}")
    
    # 2. 加载训练好的模型和标准化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNRegressionModel(
        in_features=6,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        use_batch_norm=True
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载标准化器 - 修改为使用pickle加载y_scaler对象
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    y_scaler = scalers['y_scaler']
    
    # 3. 生成扰动结构并进行预测
    # 定义不同的扰动标准差（控制扰动大小）
    perturbation_std_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # 单位：Å
    num_perturbations_per_level = 20  # 每个扰动级别生成的结构数量
    num_atoms_to_perturb = 3  # 每次扰动的原子数量
    
    results = []
    
    # 计算原始结构的预测能量
    original_features = build_node_features(best_coords)
    original_adj = build_adjacency_matrix(best_coords)
    
    with torch.no_grad():
        X = torch.FloatTensor(original_features).unsqueeze(0).to(device)
        A = torch.FloatTensor(original_adj).unsqueeze(0).to(device)
        pred_energy_original = model(X, A).cpu().numpy()[0, 0]
    
    # 反标准化预测能量 - 使用y_scaler.inverse_transform方法
    pred_energy_original_denorm = y_scaler.inverse_transform([[pred_energy_original]])[0, 0]
    
    print(f"Predicted energy of original structure: {pred_energy_original_denorm:.6f}")
    
    # 可视化一些扰动结构的示例
    example_dir = os.path.join(output_dir, "examples")
    os.makedirs(example_dir, exist_ok=True)
    
    # 对每个扰动级别生成多个结构
    for std in perturbation_std_values:
        print(f"Processing perturbation level: std={std}Å")
        
        for i in range(num_perturbations_per_level):
            # 应用扰动
            perturbed_coords, perturbation_magnitude = apply_perturbation(
                best_coords, std, num_atoms_to_perturb)
            
            # 构建特征和邻接矩阵
            perturbed_features = build_node_features(perturbed_coords)
            perturbed_adj = build_adjacency_matrix(perturbed_coords)
            
            # 预测能量
            with torch.no_grad():
                X = torch.FloatTensor(perturbed_features).unsqueeze(0).to(device)
                A = torch.FloatTensor(perturbed_adj).unsqueeze(0).to(device)
                pred_energy = model(X, A).cpu().numpy()[0, 0]
            
            # 反标准化预测能量 - 使用y_scaler.inverse_transform方法
            pred_energy_denorm = y_scaler.inverse_transform([[pred_energy]])[0, 0]
            
            # 计算能量变化
            energy_change = pred_energy_denorm - pred_energy_original_denorm
            
            # 保存结果
            results.append({
                'perturbation_std': std,
                'perturbation_magnitude': perturbation_magnitude,
                'pred_energy': pred_energy_denorm,
                'energy_change': energy_change
            })
            
            # 为每个扰动级别保存一个示例可视化
            if i == 0:
                # 找出被扰动的原子（位移最大的前num_atoms_to_perturb个原子）
                displacements = np.linalg.norm(perturbed_coords - best_coords, axis=1)
                perturbation_indices = np.argsort(displacements)[-num_atoms_to_perturb:]
                
                # 可视化
                save_path = os.path.join(example_dir, f"perturbation_std_{std}_example.png")
                visualize_structure_with_perturbation(
                    best_coords, perturbed_coords, perturbation_indices,
                    title=f"Perturbation size: std={std}Å",
                    save_path=save_path
                )
    
    # 4. 分析结果
    results_df = pd.DataFrame(results)
    
    # 计算稳定性指标
    stability_metrics, overall_stability = calculate_stability_metric(
        pred_energy_original_denorm,
        results_df['pred_energy'].values,
        results_df['perturbation_magnitude'].values
    )
    
    results_df['stability_metric'] = stability_metrics
    
    print(f"Overall structure stability metric: {overall_stability:.6f}")
    
    # 计算每个扰动级别的MAE和RMSE
    metrics_by_level = []
    
    for std in perturbation_std_values:
        level_data = results_df[results_df['perturbation_std'] == std]
        
        # 这里我们假设原始结构的预测能量是"真实值"
        # 计算扰动结构预测能量与原始结构预测能量之间的差异统计
        mae = mean_absolute_error([pred_energy_original_denorm] * len(level_data), level_data['pred_energy'])
        rmse = np.sqrt(mean_squared_error([pred_energy_original_denorm] * len(level_data), level_data['pred_energy']))
        
        # 执行配对t检验：比较原始结构能量和扰动结构能量
        # 由于每个扰动结构都与同一个原始结构比较，我们构造配对样本
        original_energy_array = np.full(len(level_data), pred_energy_original_denorm)
        t_stat, p_value = stats.ttest_rel(original_energy_array, level_data['pred_energy'])
        
        metrics_by_level.append({
            'perturbation_std': std,
            'mae': mae,
            'rmse': rmse,
            'avg_energy_change': level_data['energy_change'].mean(),
            'std_energy_change': level_data['energy_change'].std(),
            'avg_stability_metric': level_data['stability_metric'].mean(),
            # 配对t检验：比较原始结构与扰动结构的能量差异
            't_statistic': t_stat,
            'p_value': p_value
        })
    metrics_df = pd.DataFrame(metrics_by_level)
    
    # 保存结果
    results_df.to_csv(os.path.join(output_dir, "perturbation_results.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "perturbation_metrics.csv"), index=False)
    
    print("\nStatistics by perturbation level:")
    print(metrics_df.to_string(index=False))
    
    # 打印t检验结果摘要
    print("\nPaired t-test results:")
    print("{:<15} {:<15} {:<15}".format("Perturbation Std", "t-statistic", "p-value"))
    print("-" * 45)
    for _, row in metrics_df.iterrows():
        print("{:<15.2f} {:<15.4f} {:<15.4e}".format(row['perturbation_std'], row['t_statistic'], row['p_value']))
    
    # 判断统计显著性
    alpha = 0.05
    significant_levels = metrics_df[metrics_df['p_value'] < alpha]['perturbation_std'].tolist()
    if significant_levels:
        print("\nStatistically significant differences found at perturbation levels:", significant_levels)
    else:
        print("\nNo statistically significant differences found at any perturbation level.")
    
    # 5. 可视化结果
    
    # 扰动大小与能量变化的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='perturbation_magnitude', y='energy_change', hue='perturbation_std', 
                   palette='viridis', data=results_df, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Perturbation Magnitude (Å)')
    plt.ylabel('Energy Change (eV)')
    plt.title('Relationship between Perturbation Magnitude and Energy Change')
    plt.legend(title='Perturbation Std (Å)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "perturbation_vs_energy_change.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # MAE和RMSE随扰动标准差的变化
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='perturbation_std', y='mae', data=metrics_df, marker='o', label='MAE')
    sns.lineplot(x='perturbation_std', y='rmse', data=metrics_df, marker='s', label='RMSE')
    plt.xlabel('Perturbation Std (Å)')
    plt.ylabel('Error (eV)')
    plt.title('MAE and RMSE vs. Perturbation Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_vs_perturbation_std.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # 稳定性指标随扰动标准差的变化
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='perturbation_std', y='avg_stability_metric', data=metrics_df, marker='^')
    plt.xlabel('Perturbation Std (Å)')
    plt.ylabel('Average Stability Metric')
    plt.title('Stability Metric vs. Perturbation Standard Deviation')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "stability_vs_perturbation_std.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # 能量变化分布
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['energy_change'], kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Energy Change (eV)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Energy Changes')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "energy_change_distribution.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # 配对t检验结果可视化 - p值分布
    plt.figure(figsize=(10, 6))
    sns.barplot(x='perturbation_std', y='p_value', data=metrics_df)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (α=0.05)')
    plt.axhline(y=0.01, color='r', linestyle=':', label='Significance level (α=0.01)')
    plt.xlabel('Perturbation Std (Å)')
    plt.ylabel('p-value')
    plt.title('P-values from Paired t-test Across Perturbation Levels')
    plt.yscale('log')  # 使用对数刻度更清晰地显示小p值
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "t_test_p_values.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # 配对t检验结果可视化 - t统计量
    plt.figure(figsize=(10, 6))
    sns.barplot(x='perturbation_std', y='t_statistic', data=metrics_df)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    # 计算并显示临界t值（双侧检验，自由度=样本量-1）
    df = num_perturbations_per_level - 1
    critical_t = stats.t.ppf(1 - 0.025, df)  # 双侧检验，α=0.05
    plt.axhline(y=critical_t, color='r', linestyle='--', label=f'Critical t-value (α=0.05, df={df})')
    plt.axhline(y=-critical_t, color='r', linestyle='--')
    plt.xlabel('Perturbation Std (Å)')
    plt.ylabel('t-statistic')
    plt.title('t-statistics from Paired t-test Across Perturbation Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "t_test_statistics.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\nAnalysis complete! Results saved to {output_dir}")

    # 添加分组条形图展示不同扰动级别下的比较结果（类似用户提供的示例图）
    plt.figure(figsize=(12, 7))
    
    # 准备数据 - 这里假设我们想比较不同扰动级别下的原始结构能量预测与扰动结构能量预测
    comparison_data = []
    for perturbation_std in perturbation_stds:
        # 筛选当前扰动级别的数据
        current_data = results_df[results_df['perturbation_std'] == perturbation_std]
        
        # 为每组添加原始结构和扰动结构的平均能量预测
        original_mean_energy = current_data['original_energy'].mean()
        perturbed_mean_energy = current_data['perturbed_energy'].mean()
        
        # 计算标准误差（用于误差条）
        original_se = current_data['original_energy'].sem()
        perturbed_se = current_data['perturbed_energy'].sem()
        
        comparison_data.append({
            'perturbation_std': perturbation_std,
            'group': 'Original',
            'mean_energy': original_mean_energy,
            'sem': original_se
        })
        comparison_data.append({
            'perturbation_std': perturbation_std,
            'group': 'Perturbed',
            'mean_energy': perturbed_mean_energy,
            'sem': perturbed_se
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 绘制分组条形图
    sns.barplot(x='perturbation_std', y='mean_energy', hue='group', data=comparison_df,
                palette=['#FF9999', '#66B3FF'], capsize=0.1)
    
    # 添加误差条
    # 注意：sns.barplot默认会添加误差条，但这里我们可以自定义
    
    # 添加显著性标记
    # 遍历每个扰动级别，标记有显著差异的比较
    for i, perturbation_std in enumerate(perturbation_stds):
        # 获取当前扰动级别的p值
        p_value = metrics_df[metrics_df['perturbation_std'] == perturbation_std]['p_value'].values[0]
        
        # 根据p值添加不同的显著性标记
        if p_value < 0.001:
            sig_marker = '***'
        elif p_value < 0.01:
            sig_marker = '**'
        elif p_value < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'
        
        # 在条形图上方添加显著性标记
        plt.text(i, max(comparison_df[comparison_df['perturbation_std'] == perturbation_std]['mean_energy']) + 0.1,
                 sig_marker, ha='center', fontsize=12, fontweight='bold')
    
    plt.xlabel('Perturbation Std (Å)')
    plt.ylabel('Mean Energy (eV)')
    plt.title('Comparison of Mean Energy between Original and Perturbed Structures')
    plt.legend(title='Structure Type')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grouped_mean_energy_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # 配对t检验结果可视化 - 分组条形图（类似示例图）
    plt.figure(figsize=(10, 7))
    
    # 准备数据 - 比较原始结构和扰动结构的平均能量预测
    comparison_data = []
    for perturbation_std in perturbation_stds:
        # 筛选当前扰动级别的数据
        current_data = results_df[results_df['perturbation_std'] == perturbation_std]
        
        # 计算原始结构和扰动结构的平均能量预测
        original_mean_energy = current_data['original_energy'].mean()
        perturbed_mean_energy = current_data['perturbed_energy'].mean()
        
        # 计算标准误差（用于误差条）
        original_se = current_data['original_energy'].sem()
        perturbed_se = current_data['perturbed_energy'].sem()
        
        comparison_data.append({
            'group': f'Original ({perturbation_std}Å)',
            'mean_energy': original_mean_energy,
            'sem': original_se
        })
        comparison_data.append({
            'group': f'Perturbed ({perturbation_std}Å)',
            'mean_energy': perturbed_mean_energy,
            'sem': perturbed_se
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 绘制分组条形图
    palette = ['#FF9999', '#66B3FF']  # 两组的颜色
    ax = sns.barplot(x='group', y='mean_energy', data=comparison_df,
                    palette=palette, capsize=0.1)
    
    # 添加误差条
    # 注意：seaborn的barplot默认会添加误差条（基于置信区间），但这里我们使用自己计算的标准误差
    
    # 计算每组的位置，用于添加显著性标记
    groups = comparison_df['group'].unique()
    group_positions = {group: i for i, group in enumerate(groups)}
    
    # 添加显著性标记
    for i in range(0, len(perturbation_stds)):
        # 获取当前扰动级别的p值
        perturbation_std = perturbation_stds[i]
        p_value = metrics_df[metrics_df['perturbation_std'] == perturbation_std]['p_value'].values[0]
        
        # 确定当前组的位置
        original_group = f'Original ({perturbation_std}Å)'
        perturbed_group = f'Perturbed ({perturbation_std}Å)'
        
        # 计算两组之间的高度，用于放置连接线和显著性标记
        original_height = comparison_df[comparison_df['group'] == original_group]['mean_energy'].values[0]
        perturbed_height = comparison_df[comparison_df['group'] == perturbed_group]['mean_energy'].values[0]
        max_height = max(original_height, perturbed_height)
        
        # 根据p值添加不同的显著性标记
        if p_value < 0.001:
            sig_marker = '***'
        elif p_value < 0.01:
            sig_marker = '**'
        elif p_value < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'
        
        # 在两组之间添加连接线
        x1, x2 = group_positions[original_group], group_positions[perturbed_group]
        y, h, col = max_height, 0.1, 'k'
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        
        # 添加显著性标记文本
        ax.text((x1+x2)/2, y+h, sig_marker, ha='center', va='bottom', color=col, fontsize=12, fontweight='bold')
    
    plt.xlabel('Structure Type')
    plt.ylabel('Mean Energy (eV)')
    plt.title('Comparison of Mean Energy: Original vs Perturbed Structures')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paired_t_test_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()
    
    # 配对t检验结果可视化 - 每个扰动级别独立比较
    plt.figure(figsize=(12, 7))
    
    # 为每个扰动级别创建一个子图
    for i, perturbation_std in enumerate(perturbation_stds, 1):
        ax = plt.subplot(1, len(perturbation_stds), i)
        
        # 筛选当前扰动级别的数据
        current_data = results_df[results_df['perturbation_std'] == perturbation_std]
        
        # 准备数据
        group_data = [
            {'group': 'Original', 'energy': current_data['original_energy'].mean(), 'sem': current_data['original_energy'].sem()},
            {'group': 'Perturbed', 'energy': current_data['perturbed_energy'].mean(), 'sem': current_data['perturbed_energy'].sem()}
        ]
        
        group_df = pd.DataFrame(group_data)
        
        # 绘制条形图
        palette = ['#FF9999', '#66B3FF']
        sns.barplot(x='group', y='energy', data=group_df, palette=palette, ax=ax, capsize=0.1)
        
        # 添加误差条
        for j, (_, row) in enumerate(group_df.iterrows()):
            ax.errorbar(j, row['energy'], yerr=row['sem'], fmt='none', c='black', capsize=5)
        
        # 获取当前扰动级别的p值
        p_value = metrics_df[metrics_df['perturbation_std'] == perturbation_std]['p_value'].values[0]
        
        # 根据p值添加显著性标记
        if p_value < 0.001:
            sig_marker = '***'
        elif p_value < 0.01:
            sig_marker = '**'
        elif p_value < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'
        
        # 添加显著性标记
        max_energy = group_df['energy'].max() + 0.2  # 稍微高于最高条形图
        ax.text(0.5, max_energy, sig_marker, ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        # 添加连接线
        x1, x2 = 0, 1
        y = max_energy - 0.1
        ax.plot([x1, x2], [y, y], 'k-', lw=1.5)
        
        # 设置子图标题和标签
        ax.set_title(f'Std: {perturbation_std}Å')
        ax.set_ylabel('Mean Energy (eV)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Paired t-test Results: Original vs Perturbed Structures')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为suptitle留出空间
    plt.savefig(os.path.join(output_dir, "paired_t_test_subplots.png"), dpi=160, bbox_inches="tight")
    plt.close()