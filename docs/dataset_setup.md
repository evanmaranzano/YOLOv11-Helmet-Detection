# 数据集准备指南

为了训练安全帽检测模型，我们需要准备标注好的数据集。常用的数据集是 "Safety Helmet Wearing Dataset" (SHWD)。

## 1. 数据集下载
推荐来源：
- **Kaggle**: (搜索 "Safety Helmet Detection")
- **GitHub**: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (需自行转换为 YOLO 格式)
- **Roboflow**: https://universe.roboflow.com/ (搜索 "safety helmet" 并下载为 YOLOv8/v11 格式) 推荐！

## 2. 目录结构配置
请将下载的数据集解压到 `datasets/` 目录，并确保结构如下：

```
datasets/
├── data.yaml  (已创建，请根据实际情况修改路径)
├── images/
│   ├── train/
│   │   ├── 00001.jpg
│   │   └── ...
│   └── val/
│       ├── 00002.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 00001.txt
    │   └── ...
    └── val/
        ├── 00002.txt
        └── ...
```

## 3. 标签说明
`data.yaml` 中默认配置了两个类别：
- 0: helmet (佩戴安全帽)
- 1: head (未佩戴安全帽)

如果你的数据集此处定义不同，请修改 `datasets/data.yaml`。

## 4. 快速测试 (使用生成的假数据)
如果你暂时没有数据，可以运行 `src/create_dummy_data.py` 生成少量测试数据，以验证环境和代码是否跑通。
