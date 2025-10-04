import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from scipy.stats import skew
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体，优先使用英文，保留中文字体以确保兼容性
plt.rcParams["font.family"] = ["Arial", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# -----------------------------#
# 用户配置
# -----------------------------#
SHOW_FIG = True
SAVE_FIG = True
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 使用绝对路径直接指向正确的位置
# 使用原始字符串避免转义问题
data_dir = r"c:\Users\hp\Desktop\CyberC\data (1)\data\Au20_OPT_1000"
out_dir = os.path.join(PROJECT_ROOT, "reports_task2")
os.makedirs(out_dir, exist_ok=True)

# -----------------------------#
# IO和辅助函数
# -----------------------------#
def load_xyz(file_path: str) -> Tuple[float, np.ndarray]:
    """
    加载单个.xyz文件：
      第1行：原子数（20）
      第2行：能量（浮点数或文本+浮点数）
      接下来20行："Au x y z"或"x y z"
    返回：(能量, 坐标[N,3])
    """
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].strip())
    # 稳健的能量解析：尝试浮点数；否则取最后一个标记
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
    assert arr.shape == (n, 3), f"Invalid coords shape in {file_path}: {arr.shape}"
    return energy, arr

@dataclass
class Record:
    file: str
    energy: float
    coords: np.ndarray

def load_all(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    加载所有.xyz文件并返回：
      - df: 列 = ['file','energy']，索引为0..N-1
      - coords_map: 文件 -> 坐标数组
    """
    pattern = os.path.join(data_dir, "*.[xX][yY][zZ]")
    files = sorted(glob.glob(pattern))
    print("CWD =", os.getcwd())
    print("DATA_DIR =", os.path.abspath(data_dir))
    print("匹配的.xyz文件数量 =", len(files))
    if len(files) == 0:
        raise FileNotFoundError(f"在{os.path.abspath(data_dir)}中未找到.xyz文件")
    recs = []
    coords_map = {}
    for fp in files:
        try:
            e, c = load_xyz(fp)
            recs.append((os.path.basename(fp), e))
            coords_map[os.path.basename(fp)] = c
        except Exception as e:
            print(f"[警告] 跳过{fp}: {e}")
    df = pd.DataFrame(recs, columns=["file", "energy"])
    # 将能量强制转换为数值并删除无效行
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df = df.dropna(subset=["energy"]).reset_index(drop=True)
    return df, coords_map

# -----------------------------#
# 几何工具
# -----------------------------#
def center_coords(coords: np.ndarray) -> np.ndarray:
    """将坐标中心化到质心。"""
    cen = coords.mean(axis=0, keepdims=True)
    return coords - cen

def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """返回所有两两距离r_ij，以扁平数组形式。"""
    return pdist(coords)  # 长度 = C(N,2)

def radius_of_gyration(coords: np.ndarray) -> float:
    """Rg = sqrt( mean_i ||r_i - r_cm||^2 )。"""
    centered = center_coords(coords)
    return float(np.sqrt((centered**2).sum(axis=1).mean()))

def max_diameter(coords: np.ndarray) -> float:
    """Dmax = max_ij r_ij。"""
    d = pairwise_distances(coords)
    return float(d.max()) if len(d) > 0 else 0.0

def rdf_all_pairs(all_dists: np.ndarray, r_max: float, bins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    从数据集中的所有配对距离计算类似RDF的直方图。
    返回 (r_centers, normalized_counts)
    """
    if all_dists.size == 0 or r_max <= 0:
        return np.array([]), np.array([])
    hist, edges = np.histogram(all_dists, bins=bins, range=(0.0, r_max), density=False)
    # 通过总计数和箱宽归一化 -> 简单概率密度
    dr = edges[1] - edges[0]
    pdf = hist / (hist.sum() * dr + 1e-12)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, pdf

def find_r_cut_from_rdf(r: np.ndarray, g: np.ndarray) -> Optional[float]:
    """
    启发式地找到第一个峰后的第一个最小值。
    返回r_cut或None（如果失败）。
    """
    if len(r) < 5 or len(g) < 5:
        return None
    peak_idx = np.argmax(g[:max(5, len(g)//3)])  # 关注前1/3
    if peak_idx >= len(g) - 3:
        return None
    # 找到峰后的第一个局部最小值
    for i in range(peak_idx + 1, len(g) - 1):
        if g[i] < g[i-1] and g[i] <= g[i+1]:
            return float(r[i])
    return None

def coordination_number(coords: np.ndarray, r_cut: float) -> float:
    """
    在r_cut内的平均配位数（排除自身）。
    """
    if r_cut <= 0:
        return 0.0
    N = coords.shape[0]
    dmat = squareform(pdist(coords))
    cn = []
    for i in range(N):
        # 计算严格在r_cut内的邻居数量（排除d=0的自身）
        cnt = np.sum((dmat[i, :] < r_cut) & (dmat[i, :] > 1e-12))
        cn.append(cnt)
    return float(np.mean(cn)) if len(cn) > 0 else 0.0

def geom_summary(coords: np.ndarray, r_cut: float) -> Dict[str, Any]:
    """
    计算单个团簇的几何摘要。
    """
    c = center_coords(coords)
    d = pairwise_distances(c)
    return {
        "Rg": radius_of_gyration(c),
        "Dmax": float(d.max()) if len(d) > 0 else 0.0,
        "bond_mean": float(d.mean()) if len(d) > 0 else 0.0,
        "bond_min": float(d.min()) if len(d) > 0 else 0.0,
        "bond_max": float(d.max()) if len(d) > 0 else 0.0,
        "bond_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "CN": coordination_number(c, r_cut)
    }

# -----------------------------#
# 绘图函数
# -----------------------------#
def maybe_save_show(path: Optional[str]):
    if SAVE_FIG and path is not None:
        plt.savefig(path, dpi=160, bbox_inches="tight")
        print(f"[已保存] {path}")
    if SHOW_FIG:
        plt.show(block=False)
        plt.pause(0.15)
    else:
        plt.close()

def plot_energy_distribution(df: pd.DataFrame):
    """能量的直方图 + KDE + 箱线图。"""
    if df.empty:
        print("[警告] 能量数据框为空；跳过能量图。")
        return
    plt.figure(figsize=(6,4))
    sns.histplot(df["energy"], bins=40, kde=True, color="tab:blue")
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title("Energy Distribution (Histogram + KDE)")
    maybe_save_show(os.path.join(out_dir, "energy_hist_kde.png"))

    plt.figure(figsize=(5,4))
    sns.boxplot(y=df["energy"], color="lightgray")
    plt.title("Energy Box Plot")
    maybe_save_show(os.path.join(out_dir, "energy_boxplot.png"))

def visualize_structure(coords: np.ndarray, title: str, fname: str):
    """单个团簇的3D散点图。"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    c = center_coords(coords)
    fig = plt.figure(figsize=(5.2,5.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(c[:,0], c[:,1], c[:,2], s=40, c="orange", alpha=0.9, edgecolor="k")
    ax.set_title(title)
    # 设置等比例；处理零范围情况
    span = (c.max(axis=0) - c.min(axis=0))
    max_range = float(span.max())
    if max_range <= 1e-8:
        max_range = 1.0
    for axis in 'xyz':
        getattr(ax, f"set_{axis}lim")( -0.6*max_range, 0.6*max_range )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    maybe_save_show(os.path.join(out_dir, fname))

def plot_geom_compare(df_l: pd.DataFrame, df_h: pd.DataFrame, metrics: Tuple[str,...]):
    """低能量与高能量结构的平均指标柱状图比较（带误差线）。"""
    if df_l.empty or df_h.empty:
        print("[警告] 低/高能量集之一为空；跳过柱状图比较。")
        return
    means = []
    stds = []
    groups = []
    names = []
    for m in metrics:
        means += [df_l[m].mean(), df_h[m].mean()]
        stds  += [df_l[m].std(ddof=1), df_h[m].std(ddof=1)]
        groups += ["Low Energy","High Energy"]
        names  += [m, m]
    plot_df = pd.DataFrame({"metric": names, "group": groups, "mean": means, "std": stds})
    plt.figure(figsize=(8,4.5))
    sns.barplot(data=plot_df, x="metric", y="mean", hue="group", palette="Set2", capsize=.15, errorbar=None)
    # 手动添加误差线 - 修复索引错误
    ax = plt.gca()
    for i, bar in enumerate(ax.patches):
        if i < len(plot_df):  # 添加边界检查
            x = bar.get_x() + bar.get_width()/2.0
            y = bar.get_height()
            err = plot_df.iloc[i]["std"]
            ax.errorbar(x, y, yerr=err, ecolor="k", capsize=3, fmt="none")
    plt.title("Low Energy vs High Energy: Geometric Metrics (Mean ± Std)")
    plt.ylabel("Value")
    maybe_save_show(os.path.join(out_dir, "geom_compare_bars.png"))

def plot_scatter_energy_relations(df: pd.DataFrame):
    """散点图：能量 vs Rg / 平均键长 / CN。"""
    if df.empty:
        print("[警告] 几何数据框为空；跳过散点图。")
        return
    pairs = [("Rg","Energy vs Radius of Gyration"),
             ("bond_mean","Energy vs Mean Bond Length"),
             ("CN","Energy vs Coordination Number")]
    for m, ttl in pairs:
        plt.figure(figsize=(5.2,4.2))
        sns.scatterplot(x=df[m], y=df["energy"], s=20, alpha=0.7)
        plt.xlabel(m); plt.ylabel("Energy"); plt.title(ttl)
        maybe_save_show(os.path.join(out_dir, f"scatter_energy_{m}.png"))

def plot_rdf_compare(r_low: np.ndarray, g_low: np.ndarray, r_high: np.ndarray, g_high: np.ndarray):
    if len(r_low) == 0 or len(r_high) == 0:
        print("[警告] RDF数组为空；跳过RDF比较。")
        return
    plt.figure(figsize=(6,4))
    plt.plot(r_low, g_low, label="Low Energy", lw=2)
    plt.plot(r_high, g_high, label="High Energy", lw=2)
    plt.xlabel("r (Pair Distance)")
    plt.ylabel("Probability Density (Normalized)")
    plt.title("RDF-like (Pair Distance PDF) Comparison")
    plt.legend()
    maybe_save_show(os.path.join(out_dir, "rdf_compare.png"))

# -----------------------------#
# 添加必要的导入
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# -----------------------------#
# 原子特征处理和SHAP分析
# -----------------------------#
def featurize_atoms(coords: np.ndarray) -> np.ndarray:
    """
    为每个原子创建特征向量
    输入: 原子坐标 [20, 3]
    输出: 每个原子的特征向量 [20, 6]
    """
    # 原子坐标特征 + 原子序数特征（Au的原子序数为79）
    atomic_num = 79.0
    atom_features = []
    
    for i in range(len(coords)):
        # 原子坐标特征
        coord_feature = coords[i]
        # 原子序数特征（扩展为3维）
        atomic_feature = np.array([atomic_num, atomic_num, atomic_num])
        # 组合特征
        combined_feature = np.concatenate([coord_feature, atomic_feature])
        atom_features.append(combined_feature)
        
    return np.array(atom_features)

def feature_dimensionality_engineering(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对原子特征进行升维和降维处理
    输入: 原子坐标 [20, 3]
    输出: (降维后的特征 [20, 20], 权重 [20])
    """
    # 1. 创建初始原子特征
    atom_features = featurize_atoms(coords)
    
    # 2. 特征升维：通过随机森林回归创建高维特征
    # 假设我们想要预测每个原子的某种属性（这里使用与质心的距离作为目标）
    centroid = np.mean(coords, axis=0)
    distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    
    # 创建随机森林模型进行特征升维
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(atom_features, distances_to_centroid)
    
    # 提取树的路径作为高维特征（约200-300维）
    high_dim_features = rf.apply(atom_features)
    
    # 3. 使用PCA降维到20维
    pca = PCA(n_components=20, random_state=42)
    low_dim_features = pca.fit_transform(high_dim_features)
    
    # 4. 使用SHAP计算特征重要性权重
    # 创建一个简单模型用于SHAP分析
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(low_dim_features, distances_to_centroid)
    
    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(low_dim_features)
    
    # 计算每个原子的SHAP重要性权重（特征维度的绝对值平均值）
    atom_weights = np.abs(shap_values.values).mean(axis=1)
    
    # 归一化权重
    atom_weights = (atom_weights - atom_weights.min()) / (atom_weights.max() - atom_weights.min() + 1e-12)
    
    return low_dim_features, atom_weights

def visualize_atoms_with_weights(coords: np.ndarray, weights: np.ndarray, title: str, fname: str):
    """
    可视化原子结构，使用权重控制原子大小和连接线宽
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    c = center_coords(coords)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置点大小范围（200-1000）和线宽范围（1-5）
    point_sizes = 200 + weights * 800
    line_widths = 1 + weights * 4
    
    # 绘制原子点
    scatter = ax.scatter(
        c[:, 0], c[:, 1], c[:, 2], 
        s=point_sizes, c=weights, cmap='viridis', alpha=0.8, edgecolor='k'
    )
    
    # 添加颜色条表示权重
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Weight')
    
    # 绘制连接原子的线，线宽由权重决定
    N = len(c)
    for i in range(N):
        for j in range(i+1, N):
            # 只有当两个原子之间的距离小于一定阈值时才绘制连接线
            dist = np.linalg.norm(c[i] - c[j])
            if dist < 3.5:  # 合理的Au-Au键长阈值
                # 连接线宽取两个原子权重的平均值
                avg_weight = (line_widths[i] + line_widths[j]) / 2
                ax.plot(
                    [c[i, 0], c[j, 0]],
                    [c[i, 1], c[j, 1]],
                    [c[i, 2], c[j, 2]],
                    'k-', alpha=0.3, linewidth=avg_weight
                )
    
    # 设置图形属性
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    
    # 设置坐标轴范围相同，使图形比例一致
    span = (c.max(axis=0) - c.min(axis=0))
    max_range = float(span.max())
    if max_range <= 1e-8:
        max_range = 1.0
    for axis in 'xyz':
        getattr(ax, f"set_{axis}lim")(-0.6*max_range, 0.6*max_range)
    
    maybe_save_show(os.path.join(out_dir, fname))

# -----------------------------#
# 主函数
# -----------------------------#
def main():
    # 加载所有数据
    df, coords_map = load_all(data_dir)
    if df.empty:
        raise RuntimeError("加载后能量数据框为空。请检查data_dir和文件格式。")
    df = df.sort_values("energy").reset_index(drop=True)

    # 1) 能量分布统计
    e = df["energy"].values
    stats = {
        "N": int(len(e)),
        "mean": float(np.mean(e)),
        "std": float(np.std(e, ddof=1)) if len(e) > 1 else 0.0,
        "var": float(np.var(e, ddof=1)) if len(e) > 1 else 0.0,
        "min": float(np.min(e)),
        "max": float(np.max(e)),
        "skewness": float(skew(e, bias=False)) if len(e) > 2 else 0.0
    }
    print("能量统计：", stats)
    pd.Series(stats).to_csv(os.path.join(out_dir, "energy_stats.csv"))
    plot_energy_distribution(df)

    # 2) 识别并可视化最低能量结构
    if len(df) == 0:
        raise RuntimeError("数据框中没有行；无法可视化最低能量结构。")
    best_row = df.iloc[0]
    best_file = best_row["file"]
    best_energy = float(best_row["energy"])
    print(f"最低能量结构：{best_file}，能量={best_energy:.6f}")
    visualize_structure(coords_map[best_file],
                        title=f"最低能量结构\n{best_file}, E={best_energy:.3f}",
                        fname="最低能量结构.png")

    # 3) 从所有配对的RDF导出全局r_cut
    all_dists = []
    rmax_guess = 0.0
    for f in df["file"]:
        c = center_coords(coords_map[f])
        d = pairwise_distances(c)
        if len(d) > 0:
            all_dists.append(d)
            rmax_guess = max(rmax_guess, d.max())
    if len(all_dists) == 0:
        print("[警告] 未收集到配对距离；回退r_cut=3.3 Å")
        r_cut = 3.3
        r_max = 0.0
    else:
        all_dists = np.concatenate(all_dists, axis=0)
        r_max = float(min(np.percentile(all_dists, 99.9), rmax_guess))
        r_grid, g_pdf = rdf_all_pairs(all_dists, r_max=r_max, bins=70)
        r_cut = find_r_cut_from_rdf(r_grid, g_pdf)
        if r_cut is None:
            r_cut = 3.3
            print(f"[警告] 无法从RDF检测r_cut；回退r_cut={r_cut:.2f}")
        else:
            print(f"从RDF检测到r_cut：r_cut={r_cut:.3f}（第一个峰后的第一个最小值）")

    # 4) 计算所有结构的几何摘要
    geom_rows = []
    for f, E in zip(df["file"], df["energy"]):
        summary = geom_summary(coords_map[f], r_cut=r_cut)
        summary.update({"file": f, "energy": float(E)})
        geom_rows.append(summary)
    geom_df = pd.DataFrame(geom_rows)
    geom_df.to_csv(os.path.join(out_dir, "geom_summary_all.csv"), index=False)

    # 5) 定义低能量和高能量集（底部5% vs 顶部5%，至少1个）
    q = 0.05
    n = max(1, int(len(geom_df) * q))
    df_low = geom_df.nsmallest(n, "energy").copy()
    df_high = geom_df.nlargest(n, "energy").copy()
    print(f"低能量集大小={len(df_low)}，高能量集大小={len(df_high)}")

    # 6) 比较低能量和高能量集的RDF
    # 分别构建两个PDF：
    low_d_all = []; high_d_all = []
    for f in df_low["file"]:
        low_d = pairwise_distances(center_coords(coords_map[f]))
        if len(low_d) > 0:
            low_d_all.append(low_d)
    for f in df_high["file"]:
        high_d = pairwise_distances(center_coords(coords_map[f]))
        if len(high_d) > 0:
            high_d_all.append(high_d)
    if len(low_d_all) > 0 and len(high_d_all) > 0 and r_max > 0:
        low_d_all = np.concatenate(low_d_all); high_d_all = np.concatenate(high_d_all)
        r_low, g_low = rdf_all_pairs(low_d_all, r_max=r_max, bins=60)
        r_high, g_high = rdf_all_pairs(high_d_all, r_max=r_max, bins=60)
        plot_rdf_compare(r_low, g_low, r_high, g_high)
    else:
        print("[警告] 由于低/高距离为空或r_max=0，跳过RDF比较。")

    # 7) 几何指标的柱状图比较
    metrics_to_compare = ("Rg","Dmax","bond_mean","bond_std","CN")
    plot_geom_compare(df_low, df_high, metrics=metrics_to_compare)

    # 8) 散点关系：能量 vs Rg / 平均键长 / CN
    plot_scatter_energy_relations(geom_df)

    # 9) 保存关键CSV文件
    df_low.to_csv(os.path.join(out_dir, "low_energy_geom.csv"), index=False)
    df_high.to_csv(os.path.join(out_dir, "high_energy_geom.csv"), index=False)

    # 总结低能量配置的几何特征
    print("\n低能量配置的几何特征总结：")
    print("1. 转动半径 (Rg): 平均 = {:.3f}".format(df_low["Rg"].mean()))
    print("2. 最大直径 (Dmax): 平均 = {:.3f}".format(df_low["Dmax"].mean()))
    print("3. 平均键长: 平均 = {:.3f}".format(df_low["bond_mean"].mean()))
    print("4. 配位数 (CN): 平均 = {:.3f}".format(df_low["CN"].mean()))
    print("5. 键长标准差: 平均 = {:.3f}".format(df_low["bond_std"].mean()))
    
    # 保持图形打开
    if SHOW_FIG:
        plt.show()

    # 10) 原子特征的升维、降维和SHAP权重分析
    print("\n开始进行原子特征的升维、降维和SHAP权重分析...")
    
    # 处理最低能量结构
    best_file = df.iloc[0]["file"]
    best_coords = coords_map[best_file]
    best_energy = float(df.iloc[0]["energy"])
    
    try:
        # 执行特征工程
        low_dim_features, atom_weights = feature_dimensionality_engineering(best_coords)
        
        # 可视化结果
        visualize_atoms_with_weights(
            best_coords,
            atom_weights,
            title=f"SHAP Weight Visualization of Lowest Energy Structure\n{best_file}, E={best_energy:.3f}",
            fname="shap_atom_weights.png"
        )
        
        print("原子特征处理和SHAP权重分析完成！")
        print(f"归一化权重范围: [{atom_weights.min():.4f}, {atom_weights.max():.4f}]")
        
    except Exception as e:
        print(f"[警告] 特征工程过程中出错: {e}")
    
    # 保持图形打开
    if SHOW_FIG:
        plt.show()

if __name__ == "__main__":
    main()