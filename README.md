# Mass-Spring-Damper System with Physics Informed Neural Networks

![Mass Spring Damper Animation](3c8332237d0380.gif)

## Project Overview

This project implements a Physics Informed Neural Network (PINN) to model and simulate a nonlinear Multiple-Input-Multiple-Output (MIMO) mass-spring-damper system. The implementation is being developed in MATLAB and is currently in early development stages.

## System Description

The system consists of three identical point masses ($m$ = 0.5 kg each) connected by nonlinear springs with three dampers (damping constant $d$ = 0.25 N·s/m each). The system has the following characteristics:

- Nonlinear spring stiffness: $F(x) = kΔx + kₚΔx^3$ where $k$ = 217 N/m and $k_p$ = 63.5 N/m³

- Control inputs: $u_1$ and $u_3$ applied to the first and third masses

- External disturbances: Represented as $dist$ in the system

- Output states: $x_1$ and $x_3$ positions (denoted as $y_{ctr}$) which need to be controlled

The system dynamics are governed by the following coupled nonlinear differential equations:

```math
\begin{alignat}{1}
m\ddot{x}_1 &= k(-2x_1 + x_2) + k_p[-(x_1)^3 + (x_2 - x_1)^3] + d(\dot{x}_2 - 2\dot{x}_1) + u_1 \\
m\ddot{x}_2 &= k(x_1 - 2x_2 + x_3) + k_p[(x_3 - x_2)^3 - (x_2 - x_1)^3] + d(\dot{x}_1 - 2\dot{x}_2) + \text{dist} \\
m\ddot{x}_3 &= k(x_2 - x_3) + k_p(x_2 - x_3)^3 + d(\dot{x}_2 - \dot{x}_3) + u_3
\end{alignat}
```

## Physics Informed Neural Networks Approach

This project uses Physics Informed Neural Networks (PINNs) to solve and predict the behavior of this nonlinear system. The PINN incorporates the physical laws directly into the neural network training process by:

- Encoding the nonlinear differential equations as part of the loss function

- Enforcing the physical constraints of the system

## Current Status

This project is currently in early development. Additional features and documentation will be added as the project progresses.

![Low Frequency](4d7f93b62c0e18.jpg)

![High Frequency](5a7db1492e8c06.jpg)

### Neural Network Training Results

This repository contains training results for our physics-informed neural network model.

### Training Summary

- **Total Training Time**: 04:21:49
- **GPU Memory Used**: 1.34 GB (NVIDIA GeForce RTX 3070)
- **Best Validation Loss**: 154.1845 (at iteration 1250)
- **Early Stopping**: Applied with patience of 20

#### Training Loss Progression

The graph below shows the total loss and its components (prediction loss and physics loss) throughout training:

![Training Loss](2b8d159f37c604.jpg)

The total loss combines prediction loss (weight 2.0) and physics loss (weight increasing from 0.1 to 3.0). Red dashed lines indicate learning rate changes.

#### Validation Performance

Validation loss was measured at regular intervals. The best model was saved at iteration 1250:

![Validation Loss](7a3e0d5f91c826.jpg)

#### Hyperparameter Progression

The physics loss weight (λ) was gradually increased during training, while learning rate was decreased twice:

![Hyperparameters](9f4e71c82d6a05.jpg)

#### Key Observations

- Best validation performance was achieved early in training, suggesting possible overfitting with continued training
- The physics loss weight (λ) increased from 0.24 to 1.55 throughout training
- Learning rate decreased from 0.001 to 0.00049 over the course of training
- Training balanced prediction accuracy with physics-based constraints

## Contact

For questions or feedback, please open an issue on GitHub or contact alvin.sutandar@gmail.com
