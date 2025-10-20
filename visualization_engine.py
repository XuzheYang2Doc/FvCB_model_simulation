"""
visualization_engine.py - FvCB可视化引擎
基于模块3 sim_data 绘 S 曲线/热图/敏感性图，教科书风格。
教学导向：标注三阶段（Rubisco 弯头 ca_j=25 Pa），热图见 T 罚 A 红区。
作者：GPT-5-Thinking (基于Sharkey 2016 光合绘图协议，知识至2024.6)
依赖：matplotlib (pip install matplotlib), +模块3 simulation_engine.py
使用：viz.plot_scan(sim_data) 或 CLI python visualization_engine.py --plot_type scan
"""

import numpy as np
import matplotlib.pyplot as plt
from model_core import fvcb_core  # 模块1：单点 A_net
from simulation_engine import find_transition_points, sensitivity_analysis  # 模块3：交点/敏感

# 中文字体设置（SimHei 黑体，确保下标/中文稳；备选 DejaVu Sans）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常
plt.rcParams['mathtext.fontset'] = 'stix'  # **LaTeX 下标稳**：r'$A_c$' → A_c


def plot_a_ci_curve(params, C_i_range=(0, 100, 101), T=25.0, I=1000.0, save_path='a_ci_curve.pdf'):
    """
    绘 S 曲线：A_net vs C_i (教学：三阶段标注，Rubisco 凹 ~25 Pa)。
    C_i_range: (min, max, steps)；返回 fig for 扩展。
    """
    C_i = np.linspace(*C_i_range, C_i_range[2])
    results = fvcb_core(params, C_i, I, T)  # 批量 A_c/A_j/A_p/A_net
    A_net = results['A_net']

    # 计算交点 (模块3 桥接)
    transitions = find_transition_points(params, C_i_range, T=T)
    ca_j, j_p = transitions['ca_j'], transitions['j_p']

    fig, ax = plt.subplots(figsize=(10, 6))  # 教科书尺寸
    ax.plot(C_i, A_net, linewidth=3.0, color='blue', label=r'净光合速率 $A_{net}$')

    # **三阶段标注**：axvline + annotate (教学：Rubisco 限 0-ca_j)
    ax.axvline(x=ca_j, color='green', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Rubisco-RuBP ({ca_j:.0f} Pa)')
    ax.axvline(x=j_p, color='orange', linestyle='--', alpha=0.8, linewidth=1.5, label=f'RuBP-TPU ({j_p:.0f} Pa)')
    ax.annotate('Rubisco 限速区', xy=(ca_j / 2, A_net[int(len(A_net) / 4)]), xytext=(ca_j / 2, 20),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=12, ha='center')
    ax.annotate('RuBP 再生限速区', xy=((ca_j + j_p) / 2, A_net[int(len(A_net) / 2)]), xytext=(40, 30),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=12, ha='center')

    # 美化 (教科书风：粗体标签，网格淡)
    ax.set_xlabel('叶内 CO₂ 分压 ($C_i$, Pa)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'净光合速率 ($A_{net}$, $\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=14, fontweight='bold')
    ax.set_title('FvCB 模型：A-C_i 响应曲线\n(标准条件 T=25°C, I=1000 μmol m$^{-2}$ s$^{-1}$, 图 6.1)', fontsize=16,
                 fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, C_i_range[1])
    ax.set_ylim(0, np.max(A_net) * 1.1)

    # 正区阴影 (提升饱满)
    ax.fill_between(C_i, 0, A_net, alpha=0.2, color='lightblue', label='光合区')

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
    plt.show()
    print(f"📊 S 曲线保存至 {save_path} (max A_net={np.max(A_net):.1f})")
    return fig


def plot_temperature_response(params, T_range=(15, 35, 21), C_i=40.0, I=1000.0, save_path='temp_response.pdf'):
    """
    绘温度响应：A_net vs T (教学：峰值 ~25°C，35°C 降 15% 光呼吸罚)。
    T_range: (min, max, steps)；返回 fig。
    """
    T = np.linspace(*T_range, T_range[2])
    A_net = [fvcb_core(params, C_i, I, t)['A_net'][0].item() for t in T]  # 单点循环

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T, A_net, linewidth=3.0, color='red', marker='o', markersize=4, label=r'$A_{net}$ vs T')

    # 峰值标注 (教学：E_a 升温活化 vs Γ* Q10 罚)
    peak_t = T[np.argmax(A_net)]
    peak_a = np.max(A_net)
    ax.annotate(f'峰值 ({peak_t:.0f}°C, {peak_a:.1f})', xy=(peak_t, peak_a), xytext=(25, peak_a + 2),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=12)

    ax.set_xlabel('叶温 (T, °C)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'净光合速率 ($A_{net}$, $\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=14, fontweight='bold')
    ax.set_title('FvCB 模型：温度对光合的影响\n(C_i=40 Pa, I=1000 μmol m$^{-2}$ s$^{-1}$, 图 6.2)', fontsize=16,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T_range[0], T_range[1])
    ax.set_ylim(0, np.max(A_net) * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
    plt.show()
    print(f"📈 温度响应保存至 {save_path} (峰 A_net={peak_a:.1f} at {peak_t:.0f}°C)")
    return fig


def plot_heatmap(sim_data, save_path='a_net_heatmap.pdf'):
    """
    绘热图：A_net (T, C_i) 色块 (教学：30°C+ 红区 A 降，光呼吸显)。
    sim_data: 模块3 输出 {'data': [T,C_i], 'grid': {'T':, 'C_i':} }
    """
    if 'T' not in sim_data['grid'] or 'C_i' not in sim_data['grid']:
        raise ValueError("热图需 T x C_i 扫描！教学：用 scan_ranges={'T':..., 'C_i':...}")

    T_grid, C_i_grid = sim_data['grid']['T'], sim_data['grid']['C_i'][0]  # ij indexing
    A_data = sim_data['data']

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.contourf(T_grid, C_i_grid, A_data, levels=20, cmap='RdYlBu_r', extend='max')  # **冷热反转**：高A蓝
    plt.colorbar(im, ax=ax, label=r'$A_{net}$ ($\mu$mol m$^{-2}$ s$^{-1}$)')

    # 等高线 (教学：A=20 等值线弯曲，T↑ C_i* 升)
    cs = ax.contour(T_grid, C_i_grid, A_data, levels=[10, 20, 30], colors='black', alpha=0.6, linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%d')

    ax.set_xlabel('叶温 (T, °C)', fontsize=14, fontweight='bold')
    ax.set_ylabel('叶内 CO₂ 分压 ($C_i$, Pa)', fontsize=14, fontweight='bold')
    ax.set_title('FvCB 模型：A_net 的 T-C_i 响应热图\n(等高线标关键 A 值，图 6.3)', fontsize=16, fontweight='bold',
                 pad=20)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
    plt.show()
    print(f"🔥 热图保存至 {save_path} (max A_net={np.max(A_data):.1f})")
    return fig


def plot_sensitivity(sens_df, save_path='sensitivity_bar.pdf'):
    """
    绘敏感性条形：delta_% vs 参数 (教学：Vcmax 扰动 A 变 4%，J 稳)。
    sens_df: 模块3 pd.DataFrame (perturbation, delta_%)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    params = sens_df['perturbation'].tolist()
    deltas = sens_df['delta_%'].tolist()

    bars = ax.bar(params, deltas, color=['blue' if d > 0 else 'red' for d in deltas], alpha=0.7, width=0.6)
    ax.set_ylabel('A_net 相对变化 (%)', fontsize=14, fontweight='bold')
    ax.set_title('FvCB 参数敏感性分析\n(蒙特卡洛 ±10% 扰动 at C_i=40 Pa, 图 6.4)', fontsize=16, fontweight='bold',
                 pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 数值标注 (教学：精确 %)
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{delta:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', format='pdf')
    plt.show()
    print(f"📊 敏感性图保存至 {save_path}")
    return fig


# CLI 入口 (教学：python visualization_engine.py --preset c3_wheat --plot_type all)
if __name__ == "__main__":
    from parameter_manager import PRESETS  # 模块2
    from simulation_engine import scan_variables  # 模块3
    import argparse

    parser = argparse.ArgumentParser(description="FvCB可视化 - 绘 S/热图/敏感性")
    parser.add_argument('--preset', choices=list(PRESETS.keys()), default='c3_wheat')
    parser.add_argument('--plot_type', choices=['scan', 'temp', 'heatmap', 'sens', 'all'], default='all')
    parser.add_argument('--ci_range', nargs=3, type=float, default=[0, 100, 101])
    parser.add_argument('--temp_range', nargs=3, type=float, default=[15, 35, 21])
    args = parser.parse_args()

    params = PRESETS[args.preset]
    print(f"🎨 生成 {args.plot_type} 图 (预设: {args.preset})")

    if args.plot_type in ['scan', 'all']:
        plot_a_ci_curve(params, args.ci_range)

    if args.plot_type in ['temp', 'all']:
        plot_temperature_response(params, args.temp_range)

    if args.plot_type in ['heatmap', 'all']:
        scan_ranges = {'C_i': tuple(args.ci_range), 'T': tuple(args.temp_range)}
        sim_data = scan_variables(params, scan_ranges)
        plot_heatmap(sim_data)

    if args.plot_type in ['sens', 'all']:
        sens_df = sensitivity_analysis(params, {'Vcmax25': 0.1, 'Jmax25': 0.1}, num_runs=50)
        plot_sensitivity(sens_df)
