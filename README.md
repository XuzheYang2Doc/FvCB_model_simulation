# FvCB_model_simulation
# FvCB 光合模型：教育模拟软件

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FvCB 光合模型**（**Farquhar-von Caemmerer-Berry** 模型，Farquhar et al., 1980）是植物生理学中模拟叶片净光合速率（**A_net**，单位：μmol m⁻² s⁻¹）的经典框架。本软件作为其教育性实现，提供了一个模块化的计算平台，用于定量分析光合限速机制、环境响应函数以及生态应用。该模型的核心在于三限速过程的数学表述：
\[
A_\text{net} = \min(A_c, A_j, A_p) - R_d
\]
其中，**A_c** 为 Rubisco 羧化限速（低 **C_i** 区主导），**A_j** 为 RuBP 再生限速（光量子通量 **I** 调控），**A_p** 为 TPU 磷酸盐释放限速（高 **C_i** 平台），**R_d** 为暗呼吸速率。温度依赖通过 **Arrhenius 函数**（激活能 **E_a** ≈ 65 kJ mol⁻¹）和 **Q10 光呼吸补偿**（**Γ*** Q10 = 2.3）纳入，参数基于生理文献（Sharkey et al., 2016）。软件支持参数变异分析（蒙特卡洛方法，std = 0.1）和不确定性量化（Bootstrap 置信区间与 MCMC 后验采样），便于教学中探讨参数敏感性和实验误差传播（e.g., 测量噪声 σ = 2 μmol m⁻² s⁻¹ 下 CI 带宽 ≈ 4 μmol）。

**设计导向**：针对本科/研究生植物生理学课程，强调可解释性——实时参数变异、生理验证（e.g., **V_cmax25** 夹紧至 [10, 300] μmol m⁻² s⁻¹）和输出如 S 形 A-C_i 曲线（过渡点 ca_j ≈ 25 Pa）。知识截止 2024 年 6 月，参数源自 Sharkey et al. (2016) 综述。**深入思考**：通过 GUI 交互，用户可直观探索如高温下 **Γ*** 指数上升主导 **A_net** 下降（≈15% at 35°C）或高 CO₂ 下曲线偏移（ca_j ↓ ≈10 Pa）的生理机制，推动从描述性学习向定量模拟转型。🌱

## 特性
- **核心 FvCB 引擎**：向量化计算 **A_net = min(A_c, A_j, A_p) - R_d**。
- **参数管理**：C3/C4 预设、YAML/JSON 加载、蒙特卡洛变异（std = 0.1）。
- **模拟工具**：网格扫描（NumPy meshgrid）、过渡点求解（SciPy fsolve）、敏感性分析（Pandas DataFrame 输出 Δ%）。
- **可视化**：Matplotlib 绘图（A-C_i 曲线、T 响应、T-C_i 热图、敏感性条形图；400 dpi PDF）。
- **不确定性分析**：Bootstrap 95% CI（分位数法）和 MCMC 后验采样（emcee，Rhat < 1.1 收敛）。
- **生态耦合**：日 GPP 积分（cumulative_trapezoid）、季节产量（Monteith 1977，ε = 26 g DM mol⁻¹ CO₂）、Ball-Berry 气孔导度（gs）与 VPD/胁迫耦合。
- **接口**：CLI（argparse）脚本化；Tkinter GUI 交互探索（滑块、嵌入预览）。
- **输出**：控制台日志、PDF（静态）、HTML/PNG（交互产量预测）。

## 安装
1. **克隆仓库**：
git clone https://github.com/yourusername/fvcb-model.git
cd fvcb-model

markdown
复制

2. **设置虚拟环境**（推荐）：
python -m venv fvcb_env

Windows: fvcb_env\Scripts\activate
macOS/Linux: source fvcb_env/bin/activate
markdown
复制

3. **安装依赖**：
pip install numpy scipy matplotlib plotly pyyaml pandas emcee

markdown
复制
- **Tkinter** 内置（Windows 若缺：安装 ActiveTcl；macOS：`brew install python-tk`）。

4. **验证安装**：
python -c "from model_core import fvcb_core; print('✅ 安装成功！')"

shell
复制

## 快速开始
### CLI 模式（main.py）
运行核心模拟：
python main.py --mode core --preset c3_wheat

markdown
复制
**输出**：`🌅 A_net = 28.12 μmol m⁻² s⁻¹ (C_i=30 Pa, T=25°C)`。

全流程：
python main.py --mode full --variation 0.05

markdown
复制
**输出**：日志序列 + 文件（e.g., `a_ci_curve.pdf`、`yield_forecast.html`）。

### GUI 模式（ui.py）
启动交互界面：
python ui.py

markdown
复制
- 选择预设（e.g., C4 玉米），调整滑块（e.g., T 范围 15–35°C）。
- 点击按钮（e.g., “全流程”）获取实时预览和输出。🖥️

## 使用指南
### CLI 示例
- **模拟自定义范围**：
python main.py --mode simulation --ci_range 0 100 101 --temp_range 15 35 21

markdown
复制
**输出**：过渡点（ca_j ≈ 25 Pa）和敏感性（**V_cmax** 扰动 ΔA_net ≈ 4%）。

- **不确定性量化**：
python main.py --mode uncertainty --uncertainty_method mcmc --num_samples 2000

markdown
复制
**输出**：后验 **V_cmax** ≈ 78 ± 3 μmol m⁻² s⁻¹；`mcmc_posterior.pdf`。

- **产量预测**：
python main.py --mode application --stress drought --co2 elevated --days 150

markdown
复制
**输出**：产量 ≈ 5.8 t ha⁻¹；交互 HTML 图。

**完整选项**：`python main.py --help`。

### GUI 工作流
1. **配置**：下拉预设；滑块变异/**C_i**/T 范围；胁迫/CO₂/天数选项。
2. **运行**：按钮触发模块（e.g., “可视化”生成 PDF + 更新嵌入 **A_net** 预览）。
3. **探索**：日志窗诊断；状态栏进度（e.g., “max A_net = 35”）。
4. **输出**：PDF 当前目录；产量图浏览器打开。

**自定义参数**：创建 `my_params.yaml`：
Vcmax25: 90.0 # 升高模拟光适应叶
n: 0.3 # 减罚模拟杂交品系

markdown
复制
CLI 用 `--file my_params.yaml`；GUI 文件浏览器加载。

### 模块概述
| 模块                  | 目的                                      | 关键函数/输出                  |
|-----------------------|-------------------------------------------|--------------------------------|
| **model_core.py**    | FvCB 核心（A_net 计算）                   | `fvcb_core(params, C_i, I, T)` → dict (A_c, A_j, A_p, A_net) |
| **parameter_manager.py** | 参数：预设、加载/验证/变异              | `PRESETS`、`validate_params()` → 夹紧 dict |
| **simulation_engine.py** | 批量运行、过渡点、敏感性                | `scan_variables()`、`find_transition_points()` → 网格、ca_j/j_p |
| **visualization_engine.py** | 绘图（曲线、热图、条形）               | `plot_a_ci_curve()` → PDF + 标注 |
| **uncertainty_engine.py** | Bootstrap/MCMC 不确定性                 | `bootstrap_ci()` → CI 数组；`mcmc_calibration()` → 后验 |
| **application_engine.py** | GPP/产量模拟（日/季节）                 | `simulate_seasonal_yield()` → t ha⁻¹；Plotly 预测 |
| **main.py**          | CLI 整合器                                | `main()` → 工作流协调          |
| **ui.py**            | Tkinter GUI                               | `FvCBGUI()` → 交互窗口         |

独立运行模块：e.g., `python simulation_engine.py --ci_range 0 100 101`。

## 教学资源与示例
- **实验 1：限速机制**：GUI 调 **C_i** 范围，绘 S 曲线 → 识别 Rubisco 区（ca_j < 30 Pa）。讨论低 CO₂ 适应。📊
- **实验 2：环境响应**：CLI `--temp_range 15 45 11` → 热图 PDF。分析 **Γ*** 罚（**A_net** ↓15% at 35°C）。
- **实验 3：实地数据不确定性**：MCMC 伪测量（σ=2 μmol）→ 比较 CI 宽度跨 **C_i**。
- **实验 4：气候情景**：应用模式 drought + elevated CO₂ → 产量 t ha⁻¹；量化 gs 反馈（Ball-Berry）。

**报告模板**：参数 → 输出（e.g., **A_net** 曲线）→ 解释（e.g., ∂A/∂T < 0 at 高 T）。

## 贡献
欢迎贡献！Fork、创建分支（`git checkout -b feature/xyz`）、提交变更并开 PR。重点：
- 新预设（e.g., C4 亚型）。
- 扩展（e.g., SymPy 符号极限）。
- 修复（e.g., fsolve 收敛）。
