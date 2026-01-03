# 数据集准备指南 (Dataset Setup Guide)

本项目使用的数据集源自 Roboflow Universe，并在其基础上进行了整理。

**数据集来源 (Source):**
*   **平台**: Roboflow Universe
*   **发布者**: Zayed Uddin Chowdhury
*   **链接**: [https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset/dataset/3](https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset/dataset/3)
*   **格式**: YOLOv8 / YOLOv11

---

## 2. 目录结构配置 (Directory Structure)

为确保 `data.yaml` 能正确索引，请将下载的数据集解压到 `datasets/` 目录：

```text
datasets/
└── Safety-Helmet-Wearing-Dataset.v3-base-dataset.yolov11/
    ├── README.dataset.txt
    ├── README.roboflow.txt
    ├── data.yaml  <-- 核心配置文件
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 3. 标签说明 (Classes)
本数据集包含两个类别：
*   **hat** (佩戴安全帽)
*   **person** (未佩戴安全帽 / 头部)

*(注：在 `data.yaml` 中，`hat` 对应 index 0，`person` 对应 index 1)*

## 4. 引用 (Citation)
如果您在学术研究中使用了该数据集，建议引用：
```bibtex
@misc{ safety-helmet-wearing-dataset_dataset,
    title = { Safety Helmet Wearing Dataset Dataset },
    type = { Open Source Dataset },
    author = { Zayed Uddin Chowdhury },
    howpublished = { \url{ https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset } },
    url = { https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { may },
    note = { visited on 2024-01-04 },
}
```
