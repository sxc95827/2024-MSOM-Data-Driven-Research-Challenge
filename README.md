# Pharmaceutical Manufacturing Process Optimization using LSTM for the 2024 MSOM Data-Driven Research Challenge

## Overview

This project is a demo used for the 2024 MSOM Data-Driven Research Challenge. The challenge provides a pharmaceutical manufacturing dataset that includes continuous manufacturing data for tablet production, with approximately 300 million data points collected over 120 hours. The goal is to optimize processes like blending, compression, and coating operations in real-time using advanced machine learning techniques.

## Problem Statement

The dataset presents a series of challenges related to maintaining a "state of control" over pharmaceutical production, particularly during continuous manufacturing. Variability in input materials and process parameters can significantly impact product quality. The primary challenge is to predict and control these parameters using time-series data from the manufacturing process to minimize production failures and maintain high product quality.

## Approach

I have developed a deep learning solution using Long Short-Term Memory (LSTM) networks to handle the time-series data from the manufacturing process. The LSTM model is designed to:

1. Predict the critical process parameters (CPPs) over time.
2. Anticipate deviations from the optimal "golden profile" of CPPs and make real-time corrective recommendations.
3. Optimize the blending, compression, and coating stages based on real-time sensor data.

### Key Features of the Solution:
- **LSTM Networks:** These networks are well-suited for time-series forecasting and help predict future values of process parameters based on historical data.
- **Real-time Monitoring:** By using sensor data collected from various stages of the production process (e.g., Loss-In-Weight feeders, blenders, and tablet press), the model ensures continuous monitoring.
- **Preventive Actions:** The model helps identify process disturbances early and takes preventive actions to maintain product quality.

## Dataset

The dataset used in this project is provided as part of the [2024 MSOM Data Challenge](https://doi.org/10.1287/msom.2024.0860) and includes the following data files:
- **Machine Data:** Sensor data collected every second, covering process parameters like mass flow rate, blend potency, and compression force.
- **Contextual Quality Data:** Measurements of tablet weight, thickness, and hardness, as well as the concentration of active ingredients.
  
Refer to the dataset documentation for a more detailed description of the data structure and variables.

## Code

In the provided Jupyter notebook `all_train.ipynb`, I walk through the process of loading the data, preprocessing it for the LSTM model, and training the model to forecast future process values. Key steps include:
- Data normalization and scaling
- Defining the LSTM architecture
- Training the model and evaluating its performance

## Results

The trained LSTM model shows promising results in forecasting key process parameters, helping optimize the production line by reducing variability and ensuring the process remains within control limits.


## Next Steps

Future work involves fine-tuning the LSTM model and exploring reinforcement learning methods for real-time control and optimization of the entire production process.

---

Feel free to adjust this description to better suit your project and add any additional details you'd like to highlight.
