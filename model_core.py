"""
model_core.py - FvCB模型核心引擎
用于模拟单叶净光合速率 A_net，受 C_i、光、温度影响。
教学导向：学生可修改函数学三限速机制（Rubisco/RuBP/TPU）。
作者：GPT-5-Thinking (基于Berry & Farquhar 1980等文献，知识至2024.6)
"""

import numpy as np
# import sympy as sp  # 可选：符号推导教学 (e.g., sp.solve(A_c - A_j, C_i))

# 物理常数 (教学：解释R=8.314 J/mol K, T in K)
R = 8.314  # J/mol K
T0 = 298.0  # 25°C in K

def temperature_response_arrhenius(V_ref, E_a, T):
    """
    Arrhenius温度响应：V(T) = V(25°C) * exp[ E_a (T-25) / (R T 298) ]
    E_a: 活化能 (kJ/mol → J/mol)
    教学：Rubisco E_a≈65 kJ/mol，高T下V_cmax↑但K_c也↑抵消。
    """
    T_k = T + 273.15  # °C → K
    return V_ref * np.exp(E_a * 1000 * (T_k - T0) / (R * T_k * T0))

def temperature_response_q10(V_ref, Q10, T):
    """
    Q10温度响应：V(T) = V(25°C) * Q10^{(T-25)/10}
    教学：光呼吸Γ* Q10=2.3 (升温罚重)，R_d Q10=2.0 (呼吸加速)。
    """
    return V_ref * (Q10 ** ((T - 25) / 10))

def fvcb_core(params, C_i, I=None, T=25.0):
    """
    FvCB核心计算：返回 A_c, A_j, A_p, A_net 数组。
    输入：
        params: dict, 参考参数 { 'Vcmax25': 80.0, 'J25': 160.0, ... } (见默认)
        C_i: np.array or float, 叶间CO2浓度 (Pa)
        I: np.array or float, 光量子通量 (μmol m⁻² s⁻¹), 若None用饱和J
        T: float, 温度 (°C)
    输出：dict { 'A_c': array, 'A_j': array, 'A_p': array, 'A_net': array }
    教学：C_i为数组时，向量化计算S形曲线；I限速RuBP再生。
    """
    # 默认C3植物参数 (25°C, μmol m⁻² s⁻¹ / Pa)
    default_params = {
        'Vcmax25': 80.0,      # Rubisco max carboxylation
        'Ea_Vcmax': 65.0,     # kJ/mol
        'J25': 160.0,         # RuBP regeneration max
        'Ea_J': 37.0,         # kJ/mol (光限低E_a)
        'Tp25': 10.0,         # TPU max Pi release
        'Ea_Tp': 50.0,        # kJ/mol
        'Kc25': 30.0,         # CO2 Michaelis const (Pa)
        'Ea_Kc': 80.0,        # kJ/mol (高T K_c↑抑制)
        'Ko25': 16500.0,      # O2 Michaelis const (Pa)
        'Ea_Ko': 37.0,        # kJ/mol
        'Gamma_star25': 3.74, # Photorespiration compens (Pa)
        'Q10_Gamma': 2.3,     # 升温罚重
        'Rd25': 1.0,          # Dark respiration
        'Q10_Rd': 2.0,        # 呼吸加速
        'n': 0.5,             # TPU光呼吸罚系数 (≈0.5 C3)
        'O2': 21000.0         # Atmospheric O2 (Pa)
    }
    params = {**default_params, **params}  # 合并用户覆盖

    # 温度调整参数 (粗体公式：**V_cmax(T) = V_cmax25 * Arrhenius**)
    Vcmax = temperature_response_arrhenius(params['Vcmax25'], params['Ea_Vcmax'], T)
    Jmax = temperature_response_arrhenius(params['J25'], params['Ea_J'], T)
    Tp = temperature_response_arrhenius(params['Tp25'], params['Ea_Tp'], T)
    Kc = temperature_response_arrhenius(params['Kc25'], params['Ea_Kc'], T)
    Ko = temperature_response_arrhenius(params['Ko25'], params['Ea_Ko'], T)
    Gamma_star = temperature_response_q10(params['Gamma_star25'], params['Q10_Gamma'], T)
    Rd = temperature_response_q10(params['Rd25'], params['Q10_Rd'], T)
    O2 = params['O2']  # O2无T依赖

    # 向量化 C_i (if scalar, np.atleast_1d)
    C_i = np.atleast_1d(C_i)
    if I is not None:
        I = np.atleast_1d(I)
        # J(I) = I * φ * (1 - exp(-k I)) approx, 简化为饱和: J = Jmax * (I / (I + I_sat))
        I_sat = 500.0  # 半饱和光强 (μmol m⁻² s⁻¹)
        J = Jmax * (I / (I + I_sat))  # 光响应 (教学：低I RuBP限早显)
    else:
        J = Jmax * np.ones_like(C_i)  # 饱和光假设

    # **核心方程** (粗体：**A_c = (C_i - Γ*) V_cmax / (C_i + K_c (1 + O2/K_o))**)
    vc_term = (C_i - Gamma_star) / (C_i + Kc * (1 + O2 / Ko))  # 羧化效率
    A_c = vc_term * Vcmax - Rd  # Rubisco限速

    # RuBP限速：**A_j = J (C_i - Γ*) / (4 C_i + 8 Γ*)**
    aj_denom = 4 * C_i + 8 * Gamma_star
    A_j = J * (C_i - Gamma_star) / aj_denom - Rd  # 光限 (J从I得)

    # TPU限速：**A_p = 3 (C_i - Γ*) Tp / (C_i - (1 + 3 n) Γ*)**
    ap_denom = C_i - (1 + 3 * params['n']) * Gamma_star
    A_p = np.where(ap_denom > 0,  # 防负分 (C_i > Γ* (1+3n))
                   3 * (C_i - Gamma_star) * Tp / ap_denom - Rd,
                   -Rd)  # 低C_i纯呼吸

    # 净光合：**A_net = min(A_c, A_j, A_p)**
    A_net = np.minimum.reduce([A_c, A_j, A_p])

    # 符号教学可选：用SymPy验极限 (e.g., lim C_i→∞ A_c = Vcmax - Rd)
    # if import sympy: sp.limit(vc_term * Vcmax - Rd, C_i, sp.oo) == Vcmax - Rd

    return {
        'A_c': A_c,
        'A_j': A_j,
        'A_p': A_p,
        'A_net': A_net,
        'params_adjusted': {  # 输出调整后值教学
            'Vcmax': Vcmax, 'J': J.mean() if I is not None else Jmax,
            'Gamma_star': Gamma_star, 'Rd': Rd
        }
    }

# 示例运行 (教学：测试单点/曲线)
if __name__ == "__main__":
    # 单点：25°C, C_i=30 Pa, 饱和光
    params = {}  # 用默认
    result_single = fvcb_core(params, C_i=30.0, T=25.0)
    print("单点示例 (C_i=30 Pa, 25°C):")
    print(f"A_net = {result_single['A_net'][0]:.2f} μmol m⁻² s⁻¹")
    print(f"调整参数: V_cmax={result_single['params_adjusted']['Vcmax']:.1f}")

    # 曲线：C_i=0-100 Pa
    C_i_range = np.linspace(0, 100, 101)
    result_curve = fvcb_core(params, C_i_range, T=25.0)
    print("\n曲线示例 (max A_net):", np.max(result_curve['A_net']))
    print("补偿点 (A_net=0):", np.interp(0, result_curve['A_net'], C_i_range))
