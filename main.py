"""
main.py - FvCB 完整系统主入口
整合模块1-6：参数 → 核心 → 模拟 → 可视化 → 不确定性 → 应用预测。
教学导向：一键跑全流程 (e.g., python main.py --mode full --preset c3_wheat)，或选模块。
作者：GPT-5-Thinking (基于Farquhar 1980 FvCB框架，知识至2024.6)
依赖：所有模块 + numpy, scipy, matplotlib, plotly, emcee, pyyaml, pandas
使用：python main.py --help | --mode application --stress drought
"""

import argparse
import sys
from parameter_manager import PRESETS, load_from_yaml, validate_params, add_variation
from model_core import fvcb_core
from simulation_engine import run_single, scan_variables, find_transition_points, sensitivity_analysis
from visualization_engine import plot_a_ci_curve, plot_temperature_response, plot_heatmap, plot_sensitivity
from uncertainty_engine import bootstrap_ci, plot_bootstrap_ci, mcmc_calibration, plot_mcmc_posterior
from application_engine import simulate_daily_gpp, simulate_seasonal_yield, plot_yield_forecast_plotly, ball_berry_gs


def run_core_demo(params, C_i=30.0, T=25.0):
    """**核心演示**：单点 A_net 计算 + 输出 🌅"""
    result = fvcb_core(params, C_i, T=T)
    print(f"🌅 **核心 FvCB**：C_i={C_i} Pa, T={T}°C → A_net={result['A_net'][0]:.2f} μmol m⁻² s⁻¹")
    print(f"   调整: V_cmax={result['params_adjusted']['Vcmax']:.1f}, Γ*={result['params_adjusted']['Gamma_star']:.2f}")
    return result


def run_simulation(params, ci_range=(0, 100, 101), temp_range=(15, 35, 21)):
    """**模拟演示**：扫描 + 交点 + 敏感性 📈"""
    print("🚀 **运行模拟**...")
    scan_ranges = {'C_i': ci_range, 'T': temp_range}
    sim_data = scan_variables(params, scan_ranges)
    print(f"   扫描形状: {sim_data['shape']}, max A_net={np.max(sim_data['data']):.1f}")

    transitions = find_transition_points(params, ci_range)
    print(f"   交点: ca_j={transitions['ca_j']:.1f} Pa, j_p={transitions['j_p']:.1f} Pa")

    sens_df = sensitivity_analysis(params, {'Vcmax25': 0.1}, num_runs=50)
    print(f"   敏感性: {sens_df.to_string(index=False)}")
    return sim_data, transitions, sens_df


def run_visualization(params, sim_data, transitions, sens_df, ci_range=(0, 100, 101), temp_range=(15, 35, 21)):
    """**可视化演示**：S曲线/热图/敏感性图 🎨"""
    print("📊 **生成可视化**...")
    plot_a_ci_curve(params, ci_range)
    plot_temperature_response(params, temp_range)
    plot_heatmap(sim_data)
    plot_sensitivity(sens_df)


def run_uncertainty(params, ci_range=(0, 100, 101), num_samples=1000, method='bootstrap'):
    """**不确定性演示**：Bootstrap/MCMC ❓"""
    print(f"🔬 **运行 {method} 不确定性**...")
    if method == 'bootstrap':
        boot_data = bootstrap_ci(params, ci_range, num_samples=num_samples)
        plot_bootstrap_ci(boot_data)
    elif method == 'mcmc':
        # 伪数据教学
        C_i_data = np.array([20, 40, 60])
        A_meas = np.array([run_single(params, ci)['A_net'][0].item() for ci in C_i_data]) + np.random.normal(0, 2, 3)
        mcmc_res = mcmc_calibration(params, C_i_data=C_i_data, A_meas=A_meas, A_sigma=2.0)
        plot_mcmc_posterior(mcmc_res)
    return boot_data if method == 'bootstrap' else mcmc_res


def run_application(params, stress='none', co2='ambient', days=120):
    """**应用演示**：日/季产量预测 🌾"""
    print(f"🌾 **产量预测** (胁迫: {stress}, CO₂: {co2})...")
    base_result = run_single(params, 30.0)
    LAI = 4.0
    base_gpp_leaf = base_result['A_net'][0].item() * 12 * 3600 / 1e6
    base_gpp_canopy = base_gpp_leaf * LAI

    stress_factor = 0.7 if stress == 'drought' else 1.0
    co2_factor = 1.2 if co2 == 'elevated' else 1.0
    daily_data = simulate_daily_gpp(params, stress_factor=stress_factor, co2_factor=co2_factor)
    yield_data = simulate_seasonal_yield(params, days=days, stress_pattern=stress, co2_scenario=co2,
                                         base_gpp_canopy=base_gpp_canopy)
    plot_yield_forecast_plotly(yield_data, daily_data, LAI=LAI)
    print(f"   季产量: {yield_data['yield']:.1f} t/ha")


def main():
    parser = argparse.ArgumentParser(description="**FvCB 主系统**：一键整合光合模拟/预测/不确定性 🎯")
    parser.add_argument('--mode', choices=['core', 'simulation', 'visualization', 'uncertainty', 'application', 'full'],
                        default='full', help="运行模式 (default: full)")
    parser.add_argument('--preset', choices=list(PRESETS.keys()), default='c3_wheat', help="参数预设")
    parser.add_argument('--file', type=str, help="YAML/JSON 参数文件")
    parser.add_argument('--variation', type=float, default=0.0, help="参数变异 std")
    parser.add_argument('--ci_range', nargs=3, type=float, default=[0, 100, 101])
    parser.add_argument('--temp_range', nargs=3, type=float, default=[15, 35, 21])
    parser.add_argument('--uncertainty_method', choices=['bootstrap', 'mcmc'], default='bootstrap')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--stress', choices=['none', 'drought', 'heat'], default='none')
    parser.add_argument('--co2', choices=['ambient', 'elevated'], default='ambient')
    parser.add_argument('--days', type=int, default=120)
    args = parser.parse_args()

    # **加载参数** 📋
    if args.file:
        if args.file.endswith('.yaml') or args.file.endswith('.yml'):
            params = load_from_yaml(args.file)
        else:
            from parameter_manager import load_from_json
            params = load_from_json(args.file)
    else:
        params = PRESETS[args.preset]
    if args.variation > 0:
        params = add_variation(params, args.variation)
    params = validate_params(params)
    print(f"✅ **参数加载**：{args.preset} (V_cmax25={params['Vcmax25']:.1f})")

    # **分模式运行** (full 顺序全跑)
    if args.mode in ['core', 'full']:
        run_core_demo(params)

    sim_data, transitions, sens_df = None, None, None
    if args.mode in ['simulation', 'full', 'visualization']:
        sim_data, transitions, sens_df = run_simulation(params, args.ci_range, args.temp_range)

    if args.mode in ['visualization', 'full'] and sim_data is not None:
        run_visualization(params, sim_data, transitions, sens_df, args.ci_range, args.temp_range)

    if args.mode in ['uncertainty', 'full']:
        run_uncertainty(params, args.ci_range, args.num_samples, args.uncertainty_method)

    if args.mode in ['application', 'full']:
        run_application(params, args.stress, args.co2, args.days)

    print("🏁 **FvCB 系统完成**！查看生成的 PDF/HTML 图表 📈")


if __name__ == "__main__":
    import numpy as np  # 全局依赖
    main()
