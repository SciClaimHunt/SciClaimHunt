# 🔬 SciClaimHunt: Scientific Claim Verification Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue)](https://huggingface.co/datasets/your-username/SciClaimHunt)
[![Paper](https://img.shields.io/badge/arXiv-2502.10003-b31b1b.svg)](https://arxiv.org/abs/2502.10003)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

Official codebase for the [**SciClaimHunt**](https://huggingface.co/datasets/your-username/SciClaimHunt) dataset — a large-scale resource (~110K samples) for **scientific claim verification** with evidence and full paper context.

---

## 📦 Dataset

📁 **Hosted on Hugging Face**:  
👉 [SciClaimHunt Dataset]([https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt))

🧾 Format: CSV with three columns — `evidence`, `claim`, `full_paper`

📄 [Related Paper (arXiv:2502.10003)](https://arxiv.org/abs/2502.10003)

---

## ⚙️ Features

- Baseline models: `BERT`, `RoBERTa`, `LLaMA`, `GAT`, `MHA`
- Preprocessing, training, and evaluation scripts
- Use with Hugging Face `datasets` and `transformers`

---

## 🚀 Quick Start

```bash
