"""
simulation_engine.py - FvCB模拟引擎 (修复版 v2)
批量运行模型，扫描C_i/I/T，计算交点/敏感性。
教学导向：学生探索三阶段区间（e.g., 35°C下Rubisco区缩短20%）。
作者：GPT-5-Thinking (基于von Caemmerer 2000模拟协议，知识至2024.6)
依赖：numpy, scipy (pip install scipy), +模块1 model_core.py
修复v2：**['A_net'][0].item()** 转标量，防 ndarray 格式歧义；fsolve 更稳。
"""

import numpy as np
from scipy.optimize import fsolve  # 求根教学：解A_c = A_j得C_i*
from model_core import fvcb_core  # 导入模块1核心

def run_single(params, C_i, I=None, T=25.0):
    """
    单点模拟 (教学：桥接模块1，A_net = fvcb_core(...)).
    返回：dict {A_c, A_j, A_p, A_net}.
    """
    return fvcb_core(params, C_i, I, T)

def scan_variables(params, scan_ranges, parallel=False):
    """
    批量扫描 (教学：numpy.meshgrid网格，e.g., C_i x T = 101x21=2k点<0.1s).
    scan_ranges: dict { 'C_i': (0, 100, 101), 'T': (15, 35, 21), 'I': None }
    返回：dict { 'data': 3D np.array (var1, var2, A_net), 'grid': meshgrids }
    修复：shape = tuple(shape_list) 确保 ndindex 用整数元组。
    """
    vars_list = [k for k in scan_ranges if scan_ranges[k] is not None]
    if len(vars_list) == 0:
        raise ValueError("至少指定一个扫描变量！教学：e.g., {'C_i': (0,100,101)}")

    # 网格生成 (教学：C_i x T → A_net[T,C_i])
    ranges = {k: scan_ranges[k] for k in vars_list}
    grids = np.meshgrid(*[np.linspace(*ranges[k], scan_ranges[k][2]) for k in vars_list], indexing='ij')
    shape_list = [len(np.linspace(*ranges[k], scan_ranges[k][2])) for k in vars_list]
    shape = tuple(shape_list)  # 修复：list → tuple，ndindex 兼容

    if len(vars_list) > 3:
        print("⚠️ 教学提示: 高维>3慢，考虑降维 (e.g., 只C_i x T)。")

    # 批量计算 (向量化，教学：避免for循环慢)
    A_net_grid = np.full(shape + (1,), np.nan)  # 预分配 [T,C_i,1]
    for idx in np.ndindex(shape):  # 修复后：idx 为 tuple (0,0) 等
        inputs = {vars_list[i]: grids[i][idx] for i in range(len(vars_list))}
        C_i_val = inputs.get('C_i', 50.0)  # 默认C_i
        I_val = inputs.get('I', None)
        T_val = inputs.get('T', 25.0)
        result = run_single(params, C_i_val, I_val, T_val)
        A_net_grid[idx] = result['A_net']

    return {
        'data': A_net_grid.squeeze(),  # A_net [T,C_i] or [C_i]
        'grid': dict(zip(vars_list, grids)),
        'shape': shape
    }

def find_transition_points(params, var_range, var_name='C_i', T=25.0, num_points=100):
    """
    计算三阶段交点 (教学：fsolve求根 A_c(C_i*) = A_j(C_i*), C_i*≈25 Pa Rubisco-RuBP).
    var_range: (min, max, num) for C_i/I
    返回：dict { 'ca_j': C_i*, 'j_p': C_i*, 'a_values': at交点 }
    优化：**try-except fsolve** 防无解 (fallback interp 教学备用)；**['A_net'][0].item()** 转标量。
    """
    var_array = np.linspace(*var_range, num_points)
    result = run_single(params, var_array, T=T)  # 批量得A_c/A_j/A_p
    A_c, A_j, A_p = result['A_c'], result['A_j'], result['A_p']

    # 求根函数 (教学：def eq(C_i): return A_c(C_i) - A_j(C_i); fsolve(eq, guess=20))
    def eq_ca_j(C_i_guess):
        def eq(C_i):
            return fvcb_core(params, C_i, T=T)['A_c'] - fvcb_core(params, C_i, T=T)['A_j']
        try:
            sol = fsolve(eq, C_i_guess, xtol=1e-6)[0]  # 收敛阈值教学
            if abs(eq(sol)) > 1e-3:  # 校验残差
                raise ValueError("fsolve 未收敛！")
            if sol < 0:
                print("⚠️ 教学提示: 负交点？模型参数调优 (e.g., 升 Jmax)。")
            return sol
        except:
            print("🔧 教学备用: 用interp估ca_j (fsolve失败时)。")
            return np.interp(0, A_c - A_j, var_array)  # 符号变零点

    def eq_j_p(C_i_guess):
        def eq(C_i):
            return fvcb_core(params, C_i, T=T)['A_j'] - fvcb_core(params, C_i, T=T)['A_p']
        try:
            sol = fsolve(eq, C_i_guess, xtol=1e-6)[0]
            if abs(eq(sol)) > 1e-3:
                raise ValueError("fsolve 未收敛！")
            if sol < 0:
                print("⚠️ 教学提示: 负交点？检查 TPU 参数。")
            return sol
        except:
            print("🔧 教学备用: 用interp估j_p。")
            return np.interp(0, A_j - A_p, var_array)

    # 初始猜想 (教学：低C_i Rubisco主导, 高C_i TPU平台)
    ca_j_point = eq_ca_j(20.0)  # Rubisco-RuBP ~20-30 Pa
    j_p_point = eq_j_p(50.0)   # RuBP-TPU ~40-60 Pa

    # 交点A值 **修复：提取 [0].item() 转标量，防 ndarray 格式崩**
    a_ca_j = fvcb_core(params, ca_j_point, T=T)['A_net'][0].item()
    a_j_p = fvcb_core(params, j_p_point, T=T)['A_net'][0].item()

    print(f"📍 交点计算: Rubisco-RuBP at {ca_j_point:.1f} {var_name}, A={a_ca_j:.1f}")
    print(f"📍 交点计算: RuBP-TPU at {j_p_point:.1f} {var_name}, A={a_j_p:.1f}")

    return {
        'ca_j': ca_j_point,
        'j_p': j_p_point,
        'a_ca_j': a_ca_j,
        'a_j_p': a_j_p
    }

def sensitivity_analysis(params, perturbations, num_runs=100):
    """
    敏感性分析 (教学：蒙特卡洛，±10%扰动V_cmax, 跑100次得A_mean±std at C_i=40 Pa).
    perturbations: dict { 'Vcmax25': 0.1 } (std frac)
    返回：pd.DataFrame (param, mean_A, std_A, delta_%)
    """
    import pandas as pd  # 教学表
    base_result = run_single(params, C_i=40.0)  # 基准A at典型C_i=40 Pa
    base_A = base_result['A_net'][0].item()  # **修复：标量提取**

    results = []
    np.random.seed(42)
    for _ in range(num_runs):
        pert_params = params.copy()
        for param, std_frac in perturbations.items():
            if param in pert_params:
                pert_params[param] *= np.random.normal(1, std_frac)
        pert_A = run_single(pert_params, C_i=40.0)['A_net'][0].item()  # **修复：标量**
        results.append(pert_A)

    mean_A = np.mean(results)
    std_A = np.std(results)
    delta_pct = ((mean_A - base_A) / base_A) * 100

    df = pd.DataFrame({
        'perturbation': list(perturbations.keys()),
        'mean_A': [mean_A],
        'std_A': [std_A],
        'delta_%': [delta_pct]
    })
    print(f"📈 敏感性: 扰动后 A_mean={mean_A:.1f} ±{std_A:.1f} ({delta_pct:+.1f}%)")
    return df

# CLI入口 (教学：python simulation_engine.py --ci_range 0 100 --temp_range 15 35)
if __name__ == "__main__":
    from parameter_manager import PRESETS  # 导入模块2预设
    import argparse
    parser = argparse.ArgumentParser(description="FvCB模拟引擎 - 教学批量运行/交点")
    parser.add_argument('--preset', choices=list(PRESETS.keys()), default='c3_wheat')
    parser.add_argument('--ci_range', nargs=3, type=float, default=[0, 100, 101], help="C_i min max steps")
    parser.add_argument('--temp_range', nargs=3, type=float, default=[15, 35, 5], help="T min max steps")
    args = parser.parse_args()

    params = PRESETS[args.preset]
    print(f"🚀 运行模拟 (预设: {args.preset})")

    # 扫描示例
    scan_ranges = {'C_i': tuple(args.ci_range), 'T': tuple(args.temp_range)}
    sim_data = scan_variables(params, scan_ranges)
    print(f"扫描数据形状: {sim_data['shape']} (e.g., max A_net={np.max(sim_data['data']):.1f})")

    # 交点示例
    transitions = find_transition_points(params, args.ci_range)

    # 敏感性示例
    sens = sensitivity_analysis(params, {'Vcmax25': 0.1}, num_runs=50)
    print(sens)
