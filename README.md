# 🚀 NASA Turbofan Engine RUL Prediction
This project implements a Deep Learning pipeline to predict the **Remaining Useful Life (RUL)** of aircraft engines using the NASA CMAPSS dataset.

## 🧠 How the Algorithm "Thinks" & Works

The system processes high-dimensional sensor data through a structured pipeline:

1. **Data Acquisition (Fetching):** The algorithm loads raw sensor readings (21 different sensors) from the CMAPSS dataset.
2. **Preprocessing (Normalization):** Since sensors have different scales (e.g., Temperature vs. Pressure), the "brain" applies **Min-Max Scaling**. This ensures no single sensor dominates the learning process due to its numerical magnitude.
3. **Sliding Window (Memory):** Instead of looking at a single point in time, the algorithm uses a "window" of the last `N` cycles. This mimics human memory, allowing the model to see **trends** rather than just snapshots.
4. **Feature Selection:** Sensors with constant values (noise) are dropped. The algorithm focuses only on signals that show "degradation patterns."
5. **Prediction (RUL Mapping):** The final layer calculates a regression value, answering: *"How many cycles are left until the probability of failure reaches 100%?"*

6. ## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas (Data manipulation), NumPy (Matrix operations), Scikit-Learn (Scaling), TensorFlow/Keras (Neural Network Architecture), Matplotlib (Visualization).

* ## 📊 Visualization & Results
The model's performance is evaluated using **RMSE (Root Mean Square Error)** and **MAE (Mean Absolute Error)** to measure the deviation between predicted and actual RUL.
