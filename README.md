# Genetic Algorithm for Cubic Polynomial Fitting
基于纯 Python 从零实现的遗传算法：三次多项式拟合

## 📖 Introduction
This repository demonstrates a from-scratch implementation of a Genetic Algorithm (GA) to solve a continuous optimization problem. The goal is to find the optimal coefficients `[a, b, c, d]` for a cubic polynomial (`y = ax^3 + bx^2 + cx + d`) that best fits the noisy **UCI Wine Quality dataset** (Alcohol vs. Quality).

## 📂 Repository Structure
* `/src`: Contains the core pure Python implementation of the Genetic Algorithm.
* `/data`: The extracted `wine_alcohol_quality.csv` dataset.
* `/results`: Generated visualizations including training curves and residual plots.

## ⚙️ Algorithm Configurations
The GA is built without relying on optimization libraries, implementing the following operators:
* **Population Size**: 80
* **Max Generations**: 300
* **Selection**: Tournament Selection (k=3)
* **Crossover**: Blend Crossover (alpha=0.5, rate=0.8)
* **Mutation**: Gaussian Mutation (scale=0.5, rate=0.2)
* **Fitness Function**: Mean Squared Error (MSE) computed via Horner's method for numerical stability.


## 🚀 Performance
* **GA Best MSE**: ~0.52
* **Baseline (`numpy.polyfit`) MSE**: 0.4991
The GA successfully converges to a near-optimal solution comparable to the deterministic least-squares method, demonstrating robust global search capabilities in a gradient-free environment.

## 📄 Full Report
For mathematical formulations and detailed analysis, see `GA_Polynomial_Fitting_Report.pdf`.
