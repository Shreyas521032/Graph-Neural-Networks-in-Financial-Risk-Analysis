
# 📊 Graph Neural Networks in Financial Risk Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Graph%20Neural%20Nets-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🧠 Project Overview

This project explores how **Graph Neural Networks (GNNs)** can be applied in the field of **Financial Risk Analysis** to extract meaningful insights from financial news and market sentiment. It simulates news data, processes it into graph-structured input, and uses GNNs to predict associated risk levels (Low, Medium, High).

> 🔍 Focus: Text classification & risk prediction using GCN layers from `torch_geometric`.

---

## 🗂️ Features

- ✅ Text preprocessing and vectorization
- ✅ Risk classification based on sentiment and impact
- ✅ Graph construction for GNN input
- ✅ GCN-based model using `PyTorch Geometric`
- ✅ Evaluation metrics: Accuracy, Precision, Recall, F1-score
- ✅ Visualizations and interpretability

---

## 📁 Dataset

The dataset contains **500 synthetic news headlines** from categories like:

- 📈 Market
- 🏦 Banking
- 🌍 Geopolitical
- 🪙 Crypto
- 🧾 Economy
- 🏢 Corporate

Each record includes:
- Headline
- Category
- News Source
- Published Date
- Sentiment Score
- Impact Score
- Risk Level (Target)

You can find the dataset here:  
`financial_risk_dataset_enhanced.csv`

---

## 🧪 Model Pipeline

```
📰 Raw Text → 🧹 Preprocessing → 📊 TF-IDF / Vectorization → 
🔗 Graph Construction → 🧠 GNN Model (GCN) → 🎯 Risk Prediction
```

- **Library Used**: `torch_geometric`
- **Model**: `GCNConv` layers with ReLU activation
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam

---

## 📊 Evaluation Metrics

- 🎯 Accuracy
- 📏 Precision
- 🧮 Recall
- 📐 F1-score

These help assess how well the model predicts each risk level.

---

## 🛠️ Installation

Make sure to install dependencies before running the notebook:

```bash
pip install torch torchvision
pip install torch-geometric
pip install pandas numpy nltk scikit-learn matplotlib
```

---

## 🚀 Usage

Clone this repo:

```bash
git clone https://github.com/Shreyas521032/Graph-Neural-Networks-in-Financial-Risk-Analysis.git
cd Graph-Neural-Networks-in-Financial-Risk-Analysis
```

Then run the notebook:

```bash
jupyter notebook GNN_Financial_Risk_Analysis.ipynb
```

---

## 📈 Visualizations

- 🧠 t-SNE visualizations of headline embeddings
- 📉 Training/Validation loss graphs
- 🌐 Risk prediction confusion matrix

---

## 📌 Future Work

- 🔍 Integrate real-time financial news API
- ⚙️ Try different GNN architectures (e.g., GAT, GraphSAGE)
- 📡 Add explainability using SHAP or GNNExplainer
- 📦 Deploy model as API or Streamlit app

---

## 🙌 Acknowledgements

Built with ❤️ using:
- [`PyTorch Geometric`](https://pytorch-geometric.readthedocs.io)
- [`NLTK`](https://www.nltk.org/)
- [`scikit-learn`](https://scikit-learn.org/)
- Community-driven motivation for research in AI & Finance

---
