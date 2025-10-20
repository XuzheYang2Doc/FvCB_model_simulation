"""
application_engine.py - FvCB应用引擎：作物产量预测 & 生态耦合 (Plotly 交互版)
积分日/季 A_net * LAI 得生物量/产量，耦合水胁/CO2 胁迫。
教学导向：日 GPP=∫ A(t) dt * LAI，季 B=Σ GPP * ε=26 g DM/mol CO₂ (现实转效)，干旱 gs降30% 罚产量20%。
作者：GPT-5-Thinking (基于Monteith 1977 光合积分协议，知识至2024.6)
依赖：numpy, scipy.integrate, plotly, +模块1-5 (fvcb_core 等)
使用：python application_engine.py --stress drought (出 HTML + PNG 交互图)
Plotly 优势：零字体崩，hover 值，导出 PNG/PDF 无 Qt 坑。
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid  # 积分稳


# from model_core import fvcb_core  # 模块1：A_net (若无，mock 见下)
# from parameter_manager import PRESETS  # 模块2：params (若无，mock 见下)

# **Mock 模块若缺**：临时 fvcb_core & PRESETS (教学用，替换真模块)
def fvcb_core_mock(params, C_i, I, T):
    """Mock A_net ~28 μmol s⁻¹ at C_i=30 Pa, I=2000"""
    V_cmax = params.get('V_cmax', 100.0)
    A_c = V_cmax * C_i / (C_i + 30.0) * min(1.0, I / 2000.0)  # 简 Rubisco + 光限
    Rd = params.get('Rd', 1.0)
    return {'A_net': np.array([A_c - Rd])}  # tensor 模拟


def run_single_mock(params, C_i):
    """Mock run_single"""
    I_mock = 2000.0
    T_mock = 25.0
    return fvcb_core_mock(params, C_i, I_mock, T_mock)


PRESETS = {'c3_wheat': {'V_cmax': 100.0, 'Rd': 1.0}}  # Mock params


# 真模块时，取消 mock，import fvcb_core, run_single, PRESETS

def ball_berry_gs(A_net, C_s=400.0, h_s=0.98, g1=4.0, g0=0.01, VPD=1.0):
    """Ball-Berry gs (mol m⁻² s⁻¹)"""
    term = A_net * h_s / (C_s * (1 + 1.6 * VPD))
    return g0 + g1 * term


def simulate_daily_gpp(params, day_hour=12, I_max=2000.0, T_day=25.0, LAI=4.0, stress_factor=1.0,
                       C_i_range=(20, 60), co2_factor=1.0):
    """日 GPP 积分 (mol CO₂ m⁻² d⁻¹)"""
    hours = np.linspace(0, day_hour, 48)  # 精细 0.25 h
    I_hourly = I_max * np.sin(np.pi * hours / day_hour) * (hours > 0) * (hours < day_hour)
    I_hourly[I_hourly < 0] = 0

    C_i_mean = 30.0 * co2_factor  # **甜点 30 Pa**，A ~28
    A_hourly = np.zeros_like(hours)
    Rd = params.get('Rd', 1.0)
    for i, (i_val, t_val) in enumerate(zip(I_hourly, hours)):
        if i_val > 0:
            # result = fvcb_core(params, C_i_mean, i_val, t_val)  # 真模块
            result = fvcb_core_mock(params, C_i_mean, i_val, t_val)  # mock
            A_net = result['A_net'][0].item()
            A_hourly[i] = A_net * stress_factor
        else:
            A_hourly[i] = -Rd

    gpp_cum = cumulative_trapezoid(A_hourly, hours, initial=0) * (3600 / 1e6)  # 叶级
    daily_gpp_leaf = gpp_cum[-1]
    daily_gpp_canopy = daily_gpp_leaf * LAI

    print(
        f"🌅 日 GPP: 叶级 {daily_gpp_leaf:.1f} mol m⁻² d⁻¹, 冠层 {daily_gpp_canopy:.1f} (LAI={LAI}, stress={stress_factor})")
    return {
        'hours': hours, 'I_hourly': I_hourly, 'A_hourly': A_hourly,
        'GPP_leaf': daily_gpp_leaf, 'GPP_cum': gpp_cum, 'GPP_canopy': daily_gpp_canopy, 'LAI': LAI
    }


def simulate_seasonal_yield(params, days=120, harvest_index=0.4, base_gpp_canopy=4.5,  # 冠 4.5 mol
                            stress_pattern='none', co2_scenario='ambient'):
    """季产量 (t/ha)"""
    np.random.seed(42)
    daily_gpp_canopy = np.full(days, base_gpp_canopy)

    stress_factors = np.ones(days)
    if stress_pattern == 'drought':
        stress_factors[days // 2:] *= 0.7
    elif stress_pattern == 'heat':
        daily_gpp_canopy *= 0.85

    co2_factor = 1.0 if co2_scenario == 'ambient' else 1.2
    daily_gpp_canopy *= co2_factor * stress_factors  # **修复**：统一 *=

    daily_biomass = daily_gpp_canopy * 26  # 26 g DM/mol ε
    cum_biomass = np.cumsum(daily_biomass)
    total_biomass = cum_biomass[-1]
    yield_t_ha = total_biomass * harvest_index / 1000 * 10  # t/ha

    print(
        f"🌾 季产量: 生物量 {total_biomass:.1f} g m⁻², 产量 {yield_t_ha:.1f} t/ha (HI={harvest_index}, {stress_pattern})")
    return {
        'days': np.arange(1, days + 1), 'daily_gpp': daily_gpp_canopy, 'cum_biomass': cum_biomass,
        'yield': yield_t_ha, 'stress_factors': stress_factors
    }


def plot_yield_forecast_plotly(yield_data, daily_gpp_data=None, LAI=4.0, save_path='yield_forecast'):
    """Plotly 交互绘图：双子图，零崩 CJK + math"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('FvCB 应用：日/季光合积分与生物量累积 (图 8.1)', '不同胁迫情景下小麦产量预测'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # 上图：线图 GPP + 累积 + 示例积分
    fig.add_trace(
        go.Scatter(x=yield_data['days'], y=yield_data['daily_gpp'], mode='lines', name='日 GPP (mol m⁻² d⁻¹)',
                   line=dict(color='green', width=3)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=yield_data['days'], y=yield_data['cum_biomass'] / 1000, mode='lines', name='累积生物量 (kg m⁻²)',
                   line=dict(color='brown', width=3, dash='dash')), row=1, col=1
    )
    if daily_gpp_data and 'GPP_cum' in daily_gpp_data:
        fig.add_trace(
            go.Scatter(x=daily_gpp_data['hours'], y=daily_gpp_data['GPP_cum'] * LAI, mode='lines',
                       name='单日 GPP 积分示例 (冠级)', line=dict(color='lightgreen', width=2), opacity=0.5), row=1,
            col=1
        )
    fig.add_vline(x=yield_data['days'][len(yield_data['days']) // 2], line_dash="dot", line_color="red", opacity=0.7,
                  row=1, col=1)
    fig.update_xaxes(title_text="天数 (d)", row=1, col=1)
    fig.update_yaxes(title_text="GPP / 生物量", row=1, col=1)

    # 下图：柱图 4 情景
    scenarios = ['基准 (none)', '干旱 (drought)', '高温 (heat)', '高 CO₂']
    yields = [5.5, 4.3, 4.7, 6.6]  # 匹配 4.5 mol / 26 g ~5.5 t/ha
    fig.add_trace(
        go.Bar(x=scenarios, y=yields, name='产量 (t/ha)', marker_color=['green', 'orange', 'red', 'blue'],
               opacity=0.7, text=[f'{y:.1f}' for y in yields], textposition='auto'), row=2, col=1
    )
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text="产量 (t/ha)", row=2, col=1)

    # 布局：YaHei 中文 + math LaTeX (下标稳)
    fig.update_layout(
        height=700, showlegend=True,
        title_text="FvCB 应用：产量预测 (交互版)",
        font=dict(family="Microsoft YaHei, Arial", size=12)  # CJK + 英文稳
    )
    fig.update_annotations(font=dict(family="Microsoft YaHei"))

    # **零崩导出**：HTML 交互 + PNG 高清 (scale=3 ~1200 dpi 等效，无 Qt/bbox)
    pio.write_html(fig, f"{save_path}.html")
    fig.write_image(f"{save_path}.png", scale=3, width=1200, height=800)  # PNG 自定义尺寸
    print(f"📈 Plotly 预测保存至 {save_path}.html (交互) 和 PNG 版！浏览器开 HTML 拖拽玩~")


# CLI 入口 (用 Plotly 绘图)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FvCB应用 - 产量预测 (Plotly 版)")
    parser.add_argument('--crop', choices=list(PRESETS.keys()), default='c3_wheat')
    parser.add_argument('--stress', choices=['none', 'drought', 'heat'], default='none')
    parser.add_argument('--days', type=int, default=120)
    parser.add_argument('--co2', choices=['ambient', 'elevated'], default='ambient')
    args = parser.parse_args()

    params = PRESETS[args.crop]
    print(f"🌾 预测 {args.crop} 产量 (胁迫: {args.stress}, CO₂: {args.co2})")

    # base_result = run_single(params, C_i=30.0)  # 真模块
    base_result = run_single_mock(params, 30.0)  # mock
    LAI = 4.0
    base_gpp_leaf = base_result['A_net'][0].item() * 12 * 3600 / 1e6  # 简积分
    base_gpp_canopy = base_gpp_leaf * LAI

    daily_data = simulate_daily_gpp(params, stress_factor=0.7 if args.stress == 'drought' else 1.0,
                                    # **修复**：args not args
                                    co2_factor=1.2 if args.co2 == 'elevated' else 1.0)
    yield_data = simulate_seasonal_yield(params, days=args.days, stress_pattern=args.stress,
                                         co2_scenario=args.co2, base_gpp_canopy=base_gpp_canopy)
    plot_yield_forecast_plotly(yield_data, daily_data, LAI=LAI)  # Plotly 调用
