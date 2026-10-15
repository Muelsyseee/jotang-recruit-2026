# 猫狗分类

## 阅读说明

- 实践1和实践2的代码、相关文件及运行结果分别位于同名文件夹 `实践1/` 和 `实践2/` 中，Notebook 中保留了对应的运行结果。
- 拓展代码位于 [实践2/CNN/extra_mission.ipynb](./实践2/CNN/extra_mission.ipynb)。
- 包括“趁热打铁”在内，所有回答均整理在 [note.md](./note.md) 中。可跟随 `note.md` 阅读，并参照对应的 Notebook 审查代码。
- Notebook 单元格按照题目顺序排列，方便逐题对照。

## 文件夹结构

```text
Jotang-ml-task3/
├── README.md
├── note.md                       # 各部分回答与阅读索引
├── images/                       # 笔记插图与测试图片
├── 实践1/                        # 实践1代码、相关文件及运行结果
│   ├── image_test.ipynb
│   ├── Conv2d_test.ipynb
│   └── outer1.png
└── 实践2/                        # 实践2代码、相关文件及运行结果
    └── CNN/
        ├── CNN.ipynb             # 实践2主体代码与运行结果
        ├── extra_mission.ipynb   # 拓展代码与运行结果
        ├── predict.py           # 单张图片推理程序
        ├── 炼丹记录.xlsx         # 实验记录
        ├── data/
        │   ├── raw/             # 原始数据，按 Cat、Dog 分类
        │   └── split/           # 划分后的数据，各子集按 Cat、Dog 分类
        │       ├── train/
        │       ├── val/
        │       └── test/
        ├── figures/             # 训练准确率与损失曲线图
        └── models/              # 训练结果（模型权重）
            ├── model.pth
            ├── model_enhance.pth
            ├── model_new.pth
            └── model_best.pth   # 挑选出的最佳模型
```

上图省略了工具配置和 Notebook 检查点等隐藏目录。

`CNN/figures/` 保存训练图像，`CNN/models/` 保存训练结果。其中 `model_best.pth` 是挑选出的最佳模型，供模型测试、错误样本分析及 `predict.py` 推理使用。

## 依赖包

使用 Python 3，代码需要以下第三方包：

| 包名 | 用途 |
| --- | --- |
| `numpy` | 图像数组处理与数值计算 |
| `Pillow` | 图片读取、格式转换与预处理，代码中以 `PIL` 导入 |
| `matplotlib` | 图像展示、训练曲线及特征图可视化 |
| `torch` | 张量计算、模型搭建、训练与推理 |
| `torchvision` | 数据集加载、图像变换与预训练模型 |
| `scikit-learn` | 数据集划分与混淆矩阵，代码中以 `sklearn` 导入 |
| `notebook` | 使用 Jupyter Notebook 打开和运行 `.ipynb` 文件 |

安装命令：

```bash
python3 -m pip install numpy Pillow matplotlib torch torchvision scikit-learn notebook
```

`pathlib`、`random`、`shutil`、`time` 为 Python 标准库，无需额外安装。
