# 🔋 Physics‑Informed Neural Networks (PINNs) for EV Motorcycle Energy Consumption & Range Prediction

> **Deep Learning Meets Physics: Hybrid Models for Accurate EV Range Prediction**

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)
![Status](https://img.shields.io/badge/status-Active%20Development-yellow.svg)

---

## 📋 Overview

This project develops **Physics-Informed Neural Networks (PINNs)** to predict energy consumption and range for electric motorcycles under varying riding conditions. By embedding governing physics equations (energy balance, aerodynamic drag, rolling resistance) directly into the neural network loss function, we create hybrid models that:

- ✅ Require **minimal training data**
- ✅ Generalize **beyond training conditions**
- ✅ Maintain **physical consistency**
- ✅ Deliver **interpretable results**

### 🎯 Problem Statement

Traditional range prediction methods suffer from:
- ❌ **Data Scarcity**: Limited real-world EV motorcycle datasets
- ❌ **Non-Generalization**: Models fail on unseen driving cycles
- ❌ **Black Box**: Neural networks lack physical interpretability
- ❌ **Thermal Effects**: Difficulty capturing temperature-dependent battery dynamics

**Solution**: Embed physics directly into the neural network to improve accuracy and reduce data requirements.

---

## 🛠️ Technologies & Stack

| Component | Technology |
|-----------|-----------|
| **Core Language** | Python 3.8+ |
| **Deep Learning** | PyTorch |
| **Physics Framework** | DeepXDE or custom PyTorch |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Plotly |
| **Optimization** | Adam, L-BFGS |

---

## 📅 Weekend-by-Weekend Plan

### **Week 1: Physics Formulation & Data Collection**
- [ ] Formulate governing ODE/PDE for vehicle dynamics:
  - **Energy Balance**: dE/dt = Power_input - Power_dissipation
  - **Aerodynamic Drag**: P_drag = 0.5 × ρ × C_d × A × v³
  - **Rolling Resistance**: P_roll = C_rr × m × g × v
  - **Motor Efficiency**: η(P) = 0.85 - 0.15 × (P/P_max)²
- [ ] Collect/simulate baseline data:
  - UDDS cycle (Urban Dynamometer Driving Schedule)
  - WLTC cycle (Worldwide Harmonized Light-Duty Test Cycle)
  - Real-world GPS/CAN bus data (if available)
- **Deliverable**: Physics equations document + dataset

### **Week 2: PINN Architecture & Training**
- [ ] Design PINN architecture:
  - Input: [speed, throttle, grade, temp, battery_SOC]
  - Output: [energy_consumption, power, efficiency]
  - Embed physics loss: L_total = L_data + λ × L_physics
- [ ] Train on limited data (100-500 samples)
- [ ] Validate on withheld test set
- [ ] Compare vs. baseline models (ODE solver, RF, traditional NN)
- **Deliverable**: Trained PINN model + validation plots

### **Week 3: Integration & Demo Preparation**
- [ ] Build range prediction pipeline:
  - Input: Driving cycle profile
  - Output: Expected range + confidence bounds
- [ ] Create interactive demo/visualization
- [ ] Prepare 5-minute presentation for Demo Day
- **Deliverable**: Trained model + demo notebook + presentation

---

## 📦 Project Structure

```
pinn-ev-motorcycle-energy-prediction/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── physics/                           # Physics formulations
│   ├── __init__.py
│   ├── dynamics.py                    # Vehicle dynamics equations
│   ├── energy_balance.py              # Energy conservation equations
│   └── coefficients.py                # Drag, rolling resistance params
├── data/                              # Datasets
│   ├── raw/
│   │   ├── udds_cycle.csv             # Urban driving cycle
│   │   ├── wltc_cycle.csv             # Worldwide harmonized cycle
│   │   └── realworld_data.csv         # Optional real GPS/CAN data
│   ├── processed/
│   │   ├── train_set.npz
│   │   ├── val_set.npz
│   │   └── test_set.npz
│   └── synthetic/                     # Simulated scenarios
│       └── monte_carlo_scenarios.csv
├── models/                            # PINN implementations
│   ├── __init__.py
│   ├── pinn_base.py                   # Base PINN class
│   ├── physics_loss.py                # Custom physics loss functions
│   └── training.py                    # Training loop & optimization
├── baselines/                         # Baseline models
│   ├── ode_solver.py                  # Traditional ODE numerical solver
│   ├── random_forest.py               # ML baseline
│   └── standard_nn.py                 # Standard neural network
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_pinn_training.ipynb
│   ├── 03_validation_comparison.ipynb
│   └── 04_demo.ipynb
├── results/                           # Outputs
│   ├── models/
│   │   ├── pinn_trained.pth           # Trained PINN weights
│   │   └── baseline_models/
│   ├── plots/
│   │   ├── energy_consumption.png
│   │   ├── range_prediction.png
│   │   └── comparison_vs_baselines.png
│   └── reports/
│       ├── technical_report.pdf
│       └── presentation.pdf
├── requirements.txt                   # Dependencies
└── config.yaml                        # Hyperparameters & settings
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/herambh03/pinn-ev-motorcycle-energy-prediction.git
cd pinn-ev-motorcycle-energy-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
import torch
from models.pinn_base import PINN
from physics.energy_balance import EnergyBalance
from data.loader import load_training_data

# Load training data
X_train, y_train = load_training_data('data/processed/train_set.npz')

# Initialize physics model
physics = EnergyBalance(
    vehicle_mass=150,      # kg
    drag_coefficient=0.8,
    frontal_area=0.35
)

# Create PINN
pinn = PINN(
    input_size=5,          # [v, throttle, grade, temp, SOC]
    output_size=2,         # [energy, power]
    hidden_layers=[64, 64],
    physics_model=physics,
    lambda_physics=0.1
)

# Train
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
pinn.train_model(X_train, y_train, optimizer, epochs=1000)

# Predict range
speed_profile = torch.tensor([...])  # Driving cycle
predicted_energy = pinn(speed_profile)
range_km = (battery_capacity_kwh / predicted_energy.mean()) * 100
```

---

## 📊 Physics Equations Embedded in PINN

### **Energy Balance (Primary ODE)**

```
dE/dt = P_motor × η_motor - P_drag - P_rolling - P_climbing
```

Where:
- **P_motor**: Motor input power (from throttle command)
- **η_motor(P)**: Motor efficiency curve (typically 85-92%)
- **P_drag**: Aerodynamic drag = 0.5 × ρ × C_d × A × v³
- **P_rolling**: Rolling resistance = C_rr × m × g × v
- **P_climbing**: Grade resistance = m × g × sin(θ) × v

### **Battery Model**

```
V_battery = V_ocv - I × r_internal - η_thermal
SOC(t) = SOC(0) - ∫ I(t)/C dt
```

### **Motor Efficiency Map**

```
η(P) = η_max × exp(-β × (P/P_rated)²)
```

---

## 📈 Training Strategy

### **Hybrid Loss Function**

```
L_total = L_MSE + λ_physics × L_physics + λ_reg × L_regularization

L_MSE = ||y_pred - y_true||²

L_physics = ||dE/dt - f_physics||²  (enforces energy balance)

L_regularization = ||θ||²  (prevent overfitting)
```

### **Key Hyperparameters**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Network Architecture** | 5-64-64-2 | Input: 5D, Output: 2D |
| **λ_physics** | 0.1-0.5 | Physics penalty weight |
| **Batch Size** | 32 | Small datasets benefit from mini-batches |
| **Learning Rate** | 1e-3 → 1e-4 | Decay schedule |
| **Optimizer** | Adam → L-BFGS | Two-stage training |
| **Max Epochs** | 5000 | Early stopping if validation plateaus |

---

## 🎯 Validation Approach

### **1. Data-Driven Validation**
- Hold-out test set (20% of data)
- Cross-validation (K-fold, K=5)
- Metrics: RMSE, MAE, R²

### **2. Physics-Driven Validation**
- Energy conservation check: Σ(Power_in) = Σ(Power_out) ± ε
- Efficiency curve monotonicity
- Aero drag follows v³ scaling

### **3. Comparative Benchmarking**
- PINN vs. ODE solver (numerical baseline)
- PINN vs. Random Forest (ML baseline)
- PINN vs. Standard Neural Network (no physics)

---

## 📊 Expected Outputs

✅ **Trained PINN Model** (.pth weights file)  
✅ **Range Prediction API** (Python/REST interface)  
✅ **Validation Report** (Accuracy vs. baselines)  
✅ **Interactive Demo Notebook** (Streamlit or Jupyter)  
✅ **Technical Paper** (Methods + results)  

---

## 📚 References & Resources

- [Physics-Informed Neural Networks (Raissi et al., 2019)](https://doi.org/10.1016/j.jcp.2018.10.045)
- [DeepXDE Documentation](https://deepxde.readthedocs.io/)
- [EV Energy Models Review](https://doi.org/10.1016/j.rser.2021.111023)
- [UDDS Driving Cycle](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 🤝 Contributing

Contributions, issues, and feature requests welcome!

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For research inquiries or collaborations, please open an issue or contact the project owner.

---

**Last Updated**: 2026-05-03 | **Status**: 🟡 In Active Development | **Next Phase**: Physics Formulation & Data Collection
