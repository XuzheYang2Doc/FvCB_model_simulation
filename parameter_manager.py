"""
parameter_manager.py - FvCB参数管理器
加载/验证/变异参数，支持YAML/JSON配置文件。
教学导向：学生配置C3/C4预设，理解参数生理范围（e.g., V_cmax ↑光合强）。
作者：GPT-5-Thinking (基于Sharkey 2016参数综述，知识至2024.6)
依赖：numpy, pyyaml (pip install pyyaml)
"""

import json
import os
import argparse
import numpy as np
import yaml  # YAML易读教学

# 预设模板 (教学：C3 vs C4对比，光呼吸n↓ C4高效)
PRESETS = {
    'c3_wheat': {  # C3小麦25°C默认 (模块1兼容)
        'Vcmax25': 80.0,  # **敏感：最大羧化，草本20-100**
        'Ea_Vcmax': 65.0,  # kJ/mol, 升温激活
        'J25': 160.0,  # **敏感：RuBP再生，光限关键**
        'Ea_J': 37.0,
        'Tp25': 10.0,  # TPU Pi释放
        'Ea_Tp': 50.0,
        'Kc25': 30.0,  # **敏感：CO2亲和 (Pa), 高T↑抑制**
        'Ea_Kc': 80.0,
        'Ko25': 16500.0,  # O2亲和 (Pa)
        'Ea_Ko': 37.0,
        'Gamma_star25': 3.74,  # **敏感：光呼吸补偿 (Pa), Q10重罚**
        'Q10_Gamma': 2.3,
        'Rd25': 1.0,  # 呼吸基线
        'Q10_Rd': 2.0,
        'n': 0.5,  # **敏感：TPU光呼吸罚 (C3高)**
        'O2': 21000.0  # 大气O2固定
    },
    'c4_maize': {  # C4玉米 (教学：n↓0.1, V_cmax高, 无光呼吸罚)
        'Vcmax25': 120.0,
        'Ea_Vcmax': 70.0,
        'J25': 220.0,
        'Ea_J': 40.0,
        'Tp25': 12.0,
        'Ea_Tp': 55.0,
        'Kc25': 25.0,  # 更高效
        'Ea_Kc': 75.0,
        'Ko25': 20000.0,
        'Ea_Ko': 35.0,
        'Gamma_star25': 1.0,  # 低光呼吸
        'Q10_Gamma': 1.8,
        'Rd25': 0.8,
        'Q10_Rd': 1.8,
        'n': 0.1,  # C4低罚
        'O2': 21000.0
    }
}

# 生理范围 (教学：夹紧防无效参数, e.g., V_cmax<10生理荒谬)
VALIDATION_RANGES = {
    'Vcmax25': (10.0, 300.0),  # μmol m⁻² s⁻¹
    'J25': (50.0, 500.0),
    'Tp25': (5.0, 20.0),
    'Kc25': (10.0, 100.0),  # Pa
    'Ko25': (10000.0, 30000.0),
    'Gamma_star25': (1.0, 10.0),
    'Rd25': (0.1, 5.0),
    'n': (0.05, 1.0),  # 罚系数
    'Ea_Vcmax': (30.0, 100.0),  # kJ/mol
    # ... (其他E_a/Q10类似, 教学：高E_a T敏感强)
    'T': (5.0, 50.0)  # 模拟T范围 (°C)
}


def load_from_yaml(file_path):
    """
    从YAML文件加载参数 (教学：学生编辑c3_params.yaml测试敏感性)。
    示例YAML:
    Vcmax25: 100.0  # 学生注释
    n: 0.3          # 低罚模拟
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"参数文件 {file_path} 不存在！教学提示：检查路径或用预设。")
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)
    return validate_params(loaded)


def load_from_json(file_path):
    """类似YAML, JSON备份学生实验。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON文件 {file_path} 不存在！")
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    return validate_params(loaded)


def validate_params(params):
    """
    验证+夹紧参数 (教学：警告异常值, e.g., T=60>50 '热损伤风险！')。
    返回：清洗dict。
    """
    validated = {}
    for key, value in params.items():
        if key in VALIDATION_RANGES:
            min_val, max_val = VALIDATION_RANGES[key]
            if isinstance(value, (int, float)):
                if value < min_val:
                    print(f"⚠️ 警告: {key}={value} < {min_val}, 夹紧生理下限 (教学: 低V_cmax模拟弱光叶)。")
                    value = min_val
                elif value > max_val:
                    print(f"⚠️ 警告: {key}={value} > {max_val}, 夹紧上限 (教学: 高T模拟热应激)。")
                    value = max_val
            validated[key] = value
        else:
            validated[key] = value  # 未验证键通过 (e.g., 用户自定义)

    # 特殊：T夹紧 (若提供)
    if 'T' in validated and (validated['T'] < 5 or validated['T'] > 50):
        print("🔥 教学提示: T超出5-50°C, 模拟热/冷损伤！")

    print(f"✅ 参数验证通过: {len(validated)}键 (e.g., V_cmax25={validated.get('Vcmax25', 'N/A'):.1f})")
    return validated


def add_variation(params, variation_std=0.1):
    """
    添加随机变异 (教学：±10%模拟种间/叶异质, np.random.normal)。
    输入：dict → 变异dict (种子固定教学复现)。
    """
    np.random.seed(42)  # 固定种子复现
    varied = params.copy()
    for key in ['Vcmax25', 'J25', 'Tp25', 'Kc25']:  # **敏感参数**变异
        if key in varied:
            varied[key] *= np.random.normal(1, variation_std)  # 乘性变异
            print(f"🎲 变异: {key} {params[key]:.1f} → {varied[key]:.1f} (std={variation_std})")
    return varied


def save_to_json(params, file_path):
    """导出JSON (教学：学生保存'我的参数实验.json')。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
    print(f"💾 参数保存至 {file_path}")


# CLI入口 (教学：python parameter_manager.py --preset c3_wheat --variation 0.1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FvCB参数管理器 - 教学参数加载/变异")
    parser.add_argument('--preset', choices=list(PRESETS.keys()), default='c3_wheat',
                        help="预设: c3_wheat 或 c4_maize")
    parser.add_argument('--file', type=str, help="YAML/JSON文件路径")
    parser.add_argument('--variation', type=float, default=0.0, help="随机变异std (0=无)")
    parser.add_argument('--output', type=str, help="导出JSON路径")
    args = parser.parse_args()

    # 加载
    if args.file:
        if args.file.endswith('.yaml') or args.file.endswith('.yml'):
            params = load_from_yaml(args.file)
        else:
            params = load_from_json(args.file)
    else:
        params = PRESETS[args.preset]
        print(f"📋 加载预设 '{args.preset}': V_cmax25={params['Vcmax25']:.1f}")

    # 变异+验证
    if args.variation > 0:
        params = add_variation(params, args.variation)
    params = validate_params(params)

    # 输出/保存
    print("\n最终参数dict (模块1兼容):")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")

    if args.output:
        save_to_json(params, args.output)
