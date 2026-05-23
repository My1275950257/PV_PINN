# PV_PINN

基于 PINN 物理参数反演与随机森林分类的光伏阵列故障诊断项目。

## 运行顺序

请先进入项目目录：

```powershell
cd C:\Users\MY\Desktop\workspace\PV_PINN
```

1. 生成仿真数据集

```powershell
python 01_generate_data.py
```

默认生成论文难度数据集：每类 2000 条，共 12000 条，包含辐照度/温度扰动、轻微故障、边界样本和采集噪声。快速调试可运行：

```powershell
python 01_generate_data.py --difficulty easy --samples-per-class 200
```

2. 训练 PINN 参数反演模型

```powershell
python 02_train_pinn.py
```

3. 评估 Rs/Rsh 参数反演效果

```powershell
python 03_evaluate_params.py
```

4. 单独绘制故障诊断混淆矩阵

```powershell
python 04_plot_confusion_matrix.py
```

5. 运行完整对比实验并生成论文图

```powershell
python 05_run_comparison_experiment.py
```

6. 多噪声水平鲁棒性实验

```powershell
python 06_noise_robustness.py
```

7. 模块消融实验

```powershell
python 07_ablation_study.py
```

8. 多随机种子交叉验证

```powershell
python 08_cross_validation.py
```

9. 融合特征重要性分析

```powershell
python 09_feature_importance.py
```

10. 真实运行点覆盖性分析

```powershell
python 10_real_data_coverage.py
```

## 文件说明

- `01_generate_data.py`：生成正常、老化、PID、二极管短路、局部阴影、热斑 6 类仿真 I-V 曲线数据。
- `02_train_pinn.py`：训练 PINN 物理参数反演模型，输出 `pinn_model_v3.pth`。
- `03_evaluate_params.py`：绘制 `Rs`、`Rsh` 真实值与预测值对比图。
- `04_plot_confusion_matrix.py`：使用 `PINN参数 + 随机森林` 绘制中文混淆矩阵。
- `05_run_comparison_experiment.py`：对比 KNN、SVM、纯RF、MLP、PINN-RF，并生成完整实验图表。
- `06_noise_robustness.py`：测试 0%-10% 多级测量噪声下各方法准确率变化。
- `07_ablation_study.py`：比较原始曲线、I-V特征、PINN参数、融合特征的贡献。
- `08_cross_validation.py`：进行多随机种子 5 折交叉验证，输出 `mean ± std`。
- `09_feature_importance.py`：输出 PINN-RF 物理参数的重要性排序。
- `10_real_data_coverage.py`：对比仿真样本空间与真实运行点覆盖范围。
- `99_manual_comparison_plot.py`：手动写死指标的论文展示图脚本，不属于主运行流程。
- `pinn_model.py`：PINN 网络结构和物理残差损失。
- `iv_features.py`：I-V 曲线电气特征提取。
- `experiment_utils.py`：实验公共函数，包括数据加载、PINN特征提取、分类器构建和评价。
- `paths.py`：统一输出路径管理。

## 输出位置

- 数据集：`data/train_dataset.npz`
- 模型：`pinn_model_v3.pth`
- 生成图片：`outputs/figures/`
- 结果表格：`outputs/tables/`
