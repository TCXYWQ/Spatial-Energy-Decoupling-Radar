# Spatial-Energy Decoupling Mechanism for Continuous FMCW Radar Vital Sign Measurement Under Extreme Topologies

This repository provides the official PyTorch implementation of the paper: **"Spatial-Energy Decoupling Mechanism for Continuous FMCW Radar Vital Sign Measurement Under Extreme Topologies"**, submitted to *IEEE Transactions on Instrumentation and Measurement*.

---

## 🌟 Overview

Contactless multi-target vital sign monitoring using FMCW radar faces two fundamental bottlenecks: **severe spatial-energy crosstalk** between close targets and **respiratory harmonic masking**. 

Our proposed system introduces a physics-informed architecture to overcome these challenges:
- **ICEF (Ising-Coupled Energy Filter):** Utilizes an adaptive annealing mechanism to suppress intra-target respiratory harmonics and random thermal noise.
- **DSEG (Dynamic Spatial-Energy Gating):** Strips the physical phase leakage between proximal and distal targets via a dynamic cross-attention mechanism.
- **Mamba Dynamics Tracking:** Integrates a state-space model with linear-time complexity, O(N), to maintain measurement stability under extreme occlusions.

## 📁 Repository Structure

~~~text
Spatial-Energy-Decoupling-Radar/
│
├── README.md                   # Project documentation
├── requirements.txt            # Python environment dependencies
├── Data pre-processing.py      # Pre-processes raw .bin data into CWT features and labels
├── loto378.py                  # Core script for 3x3 Grid Leave-One-Topology-Out (LOTO) validation
└── net25_1_ising_optim.py      # Full training script with Ising optimization and DSEG
~~~
*(Note: The pre-processing script will automatically generate a `Mamba_Dataset_Complex` directory to store the extracted features locally.)*

## 🚀 Getting Started

### 1. Environment Setup
The code is tested on Python 3.10.12. We highly recommend using a virtual environment (e.g., Conda). Install the required dependencies via:
~~~bash
pip install -r requirements.txt
~~~
*(Note: The `requirements.txt` includes PyTorch 2.5.1. Please ensure your CUDA version matches, or install the appropriate PyTorch version for your specific GPU from the official website.)*

### 2. Data Preparation
Our methodology is validated on the following open-source dataset:
- **Lei et al. (Mendeley Data):** [60 GHz multi-person vital sign data](https://data.mendeley.com/datasets/684v4r8wfr/1) - Used for 3x3 grid physical boundary evaluation.

Download the raw data and run the pre-processing script to generate the training features:
~~~bash
python "Data pre-processing.py"
~~~

### 3. Training & Evaluation
To perform the **Leave-One-Topology-Out (LOTO)** cross-validation on extreme topologies (e.g., Position 8: mid-field complete overlap):
~~~bash
python loto378.py
~~~
To run the full model training with optimized Ising-Coupled filters and hard example mining:
~~~bash
python net25_1_ising_optim.py
~~~

## 📊 Performance Highlights

- **Accuracy:** Achieves a global Mean Absolute Error (MAE) of **3.88 BPM** across the stringent 3x3 grid topologies.
- **Zero-Shot Robustness:** Effectively constrains measurement errors on completely unseen extreme spatial layouts (e.g., pure energy shadowing and near-field saturation) via LOTO cross-validation.
- **Efficiency:** Ultra-low single-thread inference latency of **13.91 ms** with a highly compact parameter count of **0.719 M**.

## ✍️ Authors

- **Chenxing Tan** - *First Author* - Harbin Institute of Technology (HIT)
- **Yuguan Hou** - *Corresponding Author* - HIT
- **Kuang Zhang** - HIT
- **Zhonghao Yuan** - HIT

## 📜 Citation
If you find this work or the code useful for your research, please cite our paper:
*(Citation details will be updated upon publication)*

---
*For any questions regarding the code, please contact: 2022112532@stu.hit.edu.cn*