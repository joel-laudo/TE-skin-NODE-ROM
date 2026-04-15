# TE-skin-NODE-ROM
Code for the paper:

**"Stable Long-Horizon Predictions of Neural ODE Reduced-Order Models of Tissue Expansion via Learned Feature Feedback"**

## 🧠 Overview
This repository contains the implementation of Neural Ordinary Differential Equation (NODE) reduced-order models (ROMs) for tissue expansion. The model learns low-dimensional latent dynamics of the coupled deformation and growth response and uses learned feature feedback to improve long-horizon stability and accuracy. The framework is designed to provide a fast surrogate for high-fidelity finite element simulations of tissue expansion.

## 📓 Training Notebooks

The repository includes four model variants, each implemented and trained in a separate Jupyter notebook. Each notebook contains the model architecture, rollout implementation (coupled deformation and growth), and training procedure.

* **Model A:** `Train_NODE_Model_A_vanilla.ipynb`
* **Model B:** `Train_NODE_Model_B_scalar_Ag.ipynb`
* **Model C:** `Train_NODE_Model_C_PCA.ipynb`
* **Model D:** `Train_NODE_Model_D_CNN.ipynb`

These correspond to the four approaches evaluated in the paper.

