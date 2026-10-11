# Task 1
## 项目说明

- 任务需要的图表、实验表格及思考内容的回答在 [note](report/note.md) 中给出。
- 任务要求实现的功能在 [notebook](MLP.ipynb) 中分步给出。MLP 搭建、训练、加载等具体功能按单元格实现。
- 由 [notebook](MLP.ipynb) 整合得到的完整代码位于 [MLP_code](MLP.py)。
- 对照实验表格位于 [MLP_对照实验.xlsx](report/MLP_对照实验.xlsx)。

## 目录结构

```text
.
├── README.md                     # 项目说明
├── MLP.ipynb                     # 分步骤实现及运行过程
├── MLP.py                        # 由 Notebook 整理得到的完整代码
├── mlp_moons.pth                 # 训练得到的模型文件
├── figures/                      # MLP.py 生成的图表
├── experiments/                  # 对照实验的原始记录
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   └── 7/
└── report/
    ├── note.md                   # 图表、实验表格说明及思考题回答
    ├── MLP_对照实验.xlsx          # 对照实验汇总表
    └── images/                   # note.md 引用的图片
```

其中：

- `experiments/` 保存实验原始数据，每个编号目录包含对应实验的配置与结果截图。
- `figures/` 保存 [MLP_code](MLP.py) 运行后生成的图表。
- `report/images/` 保存 [note](report/note.md) 中引用的图片。
- 脚本、Notebook、模型文件和 `figures/` 保持在同一层级，以保留代码中现有的相对路径。

## AI 使用说明

- MLP 模型搭建、训练和加载等核心代码由人工手写实现。
- 图表绘制、训练速度监控和内存使用监控相关代码由 AI 辅助实现。
- 完成所有要求的功能后，使用 AI 将 Notebook 整理为 `.py` 文件。
- 实验过程由人工完成，AI 辅助进行数据整理和表格绘制。
