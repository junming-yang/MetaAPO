# [ICLR 2026] MetaAPO

[English](README.md) | [中文](README_zh.md)

<div align="center">

**Alignment through Meta-Weighted Online Sampling: Bridging the Gap Between Data Generation and Preference Optimization**  
Junming Yang, Ning Xu, Biao Liu, Shiqi Qiao, Xin Geng

[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-%23c83232)](https://arxiv.org/abs/2509.23371)
[![Models](https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=)](https://huggingface.co/collections/jmyang/meta-apo)

</div>

## Overview

![MetaAPO overview](docs/metaapo.png)

偏好优化对于将大语言模型（LLMs）对齐到人类价值与意图至关重要，但常见挑战是：预先收集的离线偏好数据与不断演化的策略之间存在分布不匹配。**Meta-Weighted Adaptive Preference Optimization（MetaAPO）** 引入了一个轻量级的 meta-learner 作为“对齐差距估计器（alignment gap estimator）”，用于评估在线采样相对于 offline 数据的潜在收益。该机制可指导更有针对性的在线生成，并在优化目标中为样本分配逐样本的 meta-weights，从而动态平衡 online/offline 数据的质量与分布。实验在 AlpacaEval 2、Arena-Hard 和 MT-Bench 上均获得稳定提升，并将在线标注成本降低约 42%。

## Quick Start

### Requirements

- Python 3.10+（建议）
- 建议使用带 CUDA 的 GPU 环境
- 依赖：`requirements.txt`

### 安装

```bash
conda create -n metaapo python=3.10 -y
conda activate metaapo
pip install -r requirements.txt
pip install -e .
```

### 运行

请在仓库根目录执行：

```bash
bash scripts/run_metaapo.sh
```

> [!TIP]
> `run_metaapo.sh` 是默认/主要训练入口。
> 你也可以运行：
> - `bash scripts/run_metaapo_simpo.sh`：SIMPO 变体。
> - `bash scripts/run_metaapo_qwen.sh`：Qwen 2.5 7B 变体。
> - `bash scripts/run_dpo.sh`：DPO 基线。

> [!NOTE]
> `run_metaapo.sh` 默认使用 HuggingFace 的模型和数据集（见脚本中的 `*_MODEL_PATH` / `DATASET_PATH`）。请替换为你可访问的资源，并在需要时完成 HuggingFace 鉴权登录。

## 致谢与引用

本项目基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 框架开发。

如果本工作对你有帮助，欢迎引用：

```bibtex
@article{yang2025alignment,
  title={Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization},
  author={Yang, Junming and Xu, Ning and Liu, Biao and Qiao, Shiqi and Geng, Xin},
  journal={arXiv preprint arXiv:2509.23371},
  year={2025}
}
```
