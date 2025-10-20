"""
ui.py - FvCB 光合模型 GUI 界面 (Tkinter + Matplotlib嵌入)
教学导向：拖拽参数 → 点击运行 → 实时图表/日志，探索三限速机制。
作者：GPT-5-Thinking (基于Tkinter教育协议，知识至2024.6)
依赖：tkinter (内置), matplotlib (嵌入), + main.py 全模块 + webbrowser (打开HTML)
使用：python ui.py → 选模式 → Run！ (生成PDF/HTML)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser  # 打开Plotly HTML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from parameter_manager import PRESETS, load_from_yaml, load_from_json, validate_params, add_variation
from model_core import fvcb_core
from simulation_engine import run_single, scan_variables, find_transition_points, sensitivity_analysis
from visualization_engine import plot_a_ci_curve  # 示例嵌入S曲线
from uncertainty_engine import bootstrap_ci, plot_bootstrap_ci
from application_engine import simulate_daily_gpp, simulate_seasonal_yield, plot_yield_forecast_plotly
import sys
import os


class FvCBGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("**FvCB 光合模拟器** 🌱 - 交互教学工具")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # **变量绑定**：实时更新
        self.preset_var = tk.StringVar(value="c3_wheat")
        self.file_path = tk.StringVar()
        self.variation_var = tk.DoubleVar(value=0.0)
        self.ci_min, self.ci_max, self.ci_steps = tk.DoubleVar(value=0), tk.DoubleVar(value=100), tk.IntVar(value=101)
        self.t_min, self.t_max, self.t_steps = tk.DoubleVar(value=15), tk.DoubleVar(value=35), tk.IntVar(value=21)
        self.stress_var = tk.StringVar(value="none")
        self.co2_var = tk.StringVar(value="ambient")
        self.days_var = tk.IntVar(value=120)
        self.unc_method_var = tk.StringVar(value="bootstrap")
        self.num_samples_var = tk.IntVar(value=1000)

        self.params = PRESETS["c3_wheat"]  # 默认
        self.fig, self.ax = plt.subplots(figsize=(5, 4))  # 嵌入画布预览
        self.canvas = None

        self.setup_ui()
        self.update_params()  # 初始加载

    def setup_ui(self):
        # **菜单栏**：文件/帮助
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="用户指南", command=lambda: messagebox.showinfo("指南", "详见控制台或README！😊"))
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

        # **主框架**：左侧参数 (Frame1) | 中间按钮 (Frame2) | 右侧输出 (Frame3)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：参数面板 📋
        left_frame = ttk.LabelFrame(main_frame, text="**参数配置**", padding=10)
        left_frame.grid(row=0, column=0, sticky="nswe", padx=(0, 5))

        ttk.Label(left_frame, text="预设:").grid(row=0, column=0, sticky="w")
        preset_combo = ttk.Combobox(left_frame, textvariable=self.preset_var, values=list(PRESETS.keys()), state="readonly")
        preset_combo.grid(row=0, column=1, sticky="ew")
        preset_combo.bind("<<ComboboxSelected>>", self.update_params)

        ttk.Label(left_frame, text="参数文件:").grid(row=1, column=0, sticky="w")
        file_entry = ttk.Entry(left_frame, textvariable=self.file_path, width=20)
        file_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(left_frame, text="浏览", command=self.browse_file).grid(row=1, column=2)

        ttk.Label(left_frame, text="变异 std:").grid(row=2, column=0, sticky="w")
        variation_scale = ttk.Scale(left_frame, from_=0.0, to=0.2, variable=self.variation_var, orient=tk.HORIZONTAL)
        variation_scale.grid(row=2, column=1, sticky="ew")
        ttk.Label(left_frame, textvariable=self.variation_var).grid(row=2, column=2)

        # 扫描范围
        ttk.Label(left_frame, text="C_i 范围:").grid(row=3, column=0, sticky="w")
        ttk.Scale(left_frame, from_=0, to=100, variable=self.ci_min, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky="ew")
        ttk.Entry(left_frame, textvariable=self.ci_max, width=5).grid(row=3, column=2)

        ttk.Label(left_frame, text="T 范围:").grid(row=4, column=0, sticky="w")
        ttk.Scale(left_frame, from_=10, to=40, variable=self.t_min, orient=tk.HORIZONTAL).grid(row=4, column=1, sticky="ew")
        ttk.Entry(left_frame, textvariable=self.t_max, width=5).grid(row=4, column=2)

        # 应用选项
        ttk.Label(left_frame, text="胁迫:").grid(row=5, column=0, sticky="w")
        stress_combo = ttk.Combobox(left_frame, textvariable=self.stress_var, values=["none", "drought", "heat"], state="readonly")
        stress_combo.grid(row=5, column=1, sticky="ew")

        ttk.Label(left_frame, text="CO₂:").grid(row=6, column=0, sticky="w")
        co2_combo = ttk.Combobox(left_frame, textvariable=self.co2_var, values=["ambient", "elevated"], state="readonly")
        co2_combo.grid(row=6, column=1, sticky="ew")

        ttk.Label(left_frame, text="天数:").grid(row=7, column=0, sticky="w")
        ttk.Entry(left_frame, textvariable=self.days_var, width=5).grid(row=7, column=1)

        # 不确定性
        ttk.Label(left_frame, text="不确定方法:").grid(row=8, column=0, sticky="w")
        unc_combo = ttk.Combobox(left_frame, textvariable=self.unc_method_var, values=["bootstrap", "mcmc"], state="readonly")
        unc_combo.grid(row=8, column=1, sticky="ew")

        ttk.Label(left_frame, text="样本数:").grid(row=9, column=0, sticky="w")
        ttk.Entry(left_frame, textvariable=self.num_samples_var, width=5).grid(row=9, column=1)

        left_frame.grid_columnconfigure(1, weight=1)

        # 中间：按钮面板 🎮
        mid_frame = ttk.Frame(main_frame)
        mid_frame.grid(row=0, column=1, sticky="ns", padx=5)

        buttons = [
            ("**核心演示** 🌅", self.run_core, "lightgreen"),
            ("**模拟** 📈", self.run_simulation, "lightblue"),
            ("**可视化** 🎨", self.run_visualization, "yellow"),
            ("**不确定性** ❓", self.run_uncertainty, "orange"),
            ("**应用预测** 🌾", self.run_application, "lightcoral"),
            ("**全流程** 🚀", self.run_full, "purple")
        ]
        for i, (text, cmd, color) in enumerate(buttons):
            btn = tk.Button(mid_frame, text=text, command=cmd, bg=color, fg="white", font=("Arial", 10, "bold"), height=2, width=12)
            btn.grid(row=i, column=0, pady=5, sticky="ew")

        # 右侧：输出面板 📊
        right_frame = ttk.LabelFrame(main_frame, text="**日志 & 预览**", padding=10)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self.log_text = tk.Text(right_frame, height=15, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 嵌入Matplotlib画布（预览S曲线）
        self.canvas_frame = tk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.update_preview()  # 初始预览

        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪！😊")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_var.set("欢迎！选参数 → 点击运行。")

    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("YAML/JSON", "*.yaml *.yml *.json")])
        if file:
            self.file_path.set(file)
            self.update_params()

    def update_params(self, event=None):
        if self.file_path.get():
            try:
                if self.file_path.get().endswith(('.yaml', '.yml')):
                    self.params = load_from_yaml(self.file_path.get())
                else:
                    self.params = load_from_json(self.file_path.get())
            except Exception as e:
                messagebox.showerror("错误", f"文件加载失败：{e}")
        else:
            self.params = PRESETS[self.preset_var.get()]
        if self.variation_var.get() > 0:
            self.params = add_variation(self.params, self.variation_var.get())
        self.params = validate_params(self.params)
        self.log_text.insert(tk.END, f"✅ 参数更新：V_cmax={self.params.get('Vcmax25', 'N/A'):.1f}\n")
        self.log_text.see(tk.END)
        self.update_preview()  # 刷新预览

    def update_preview(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.ax.clear()
        C_i = np.linspace(0, 100, 101)
        result = fvcb_core(self.params, C_i)
        self.ax.plot(C_i, result['A_net'], 'b-', lw=2, label='A_net 预览')
        self.ax.set_xlabel('C_i (Pa)')
        self.ax.set_ylabel('A_net')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log_status(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.status_var.set(msg)
        self.root.update()

    def run_core(self):
        try:
            self.log_status("🌅 运行核心...")
            result = fvcb_core(self.params, 30.0, T=25.0)
            self.log_status(f"A_net={result['A_net'][0]:.2f} μmol m⁻² s⁻¹")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_simulation(self):
        try:
            self.log_status("📈 运行模拟...")
            ci_range = (self.ci_min.get(), self.ci_max.get(), self.ci_steps.get())
            t_range = (self.t_min.get(), self.t_max.get(), self.t_steps.get())
            sim_data = scan_variables(self.params, {'C_i': ci_range, 'T': t_range})
            transitions = find_transition_points(self.params, ci_range)
            sens_df = sensitivity_analysis(self.params, {'Vcmax25': 0.1}, num_runs=50)
            self.log_status(f"max A_net={np.max(sim_data['data']):.1f}, ca_j={transitions['ca_j']:.1f}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_visualization(self):
        try:
            self.log_status("🎨 生成可视化...")
            ci_range = (self.ci_min.get(), self.ci_max.get(), self.ci_steps.get())
            t_range = (self.t_min.get(), self.t_max.get(), self.t_steps.get())
            plot_a_ci_curve(self.params, ci_range)  # 生成PDF
            self.log_status("PDF已保存！（查看a_ci_curve.pdf）")
            self.update_preview()  # 刷新嵌入
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_uncertainty(self):
        try:
            self.log_status(f"❓ 运行{self.unc_method_var.get()}...")
            ci_range = (self.ci_min.get(), self.ci_max.get(), self.ci_steps.get())
            if self.unc_method_var.get() == 'bootstrap':
                boot_data = bootstrap_ci(self.params, ci_range, num_samples=self.num_samples_var.get())
                plot_bootstrap_ci(boot_data)
            else:  # mcmc
                C_i_data = np.array([20, 40, 60])
                A_meas = np.array([run_single(self.params, ci)['A_net'][0].item() for ci in C_i_data]) + np.random.normal(0, 2, 3)
                mcmc_res = mcmc_calibration(self.params, C_i_data=C_i_data, A_meas=A_meas, A_sigma=2.0)
                plot_mcmc_posterior(mcmc_res)
            self.log_status("不确定性图已保存！")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_application(self):
        try:
            self.log_status("🌾 运行应用预测...")
            base_result = run_single(self.params, 30.0)
            LAI = 4.0
            base_gpp_leaf = base_result['A_net'][0].item() * 12 * 3600 / 1e6
            base_gpp_canopy = base_gpp_leaf * LAI
            stress_factor = 0.7 if self.stress_var.get() == 'drought' else 1.0
            co2_factor = 1.2 if self.co2_var.get() == 'elevated' else 1.0
            daily_data = simulate_daily_gpp(self.params, stress_factor=stress_factor, co2_factor=co2_factor)
            yield_data = simulate_seasonal_yield(self.params, days=self.days_var.get(), stress_pattern=self.stress_var.get(),
                                                 co2_scenario=self.co2_var.get(), base_gpp_canopy=base_gpp_canopy)
            plot_yield_forecast_plotly(yield_data, daily_data, LAI=LAI)
            webbrowser.open(f"file://{os.path.abspath('yield_forecast.html')}")  # 浏览器打开交互图
            self.log_status(f"产量={yield_data['yield']:.1f} t/ha，HTML已打开！")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_full(self):
        self.run_core()
        self.run_simulation()
        self.run_visualization()
        self.run_uncertainty()
        self.run_application()
        self.log_status("🏁 全流程完成！检查输出文件📈")


if __name__ == "__main__":
    root = tk.Tk()
    app = FvCBGUI(root)
    root.mainloop()
