[![Demo](https://img.youtube.com/vi/mCCggAHRHGE/0.jpg)](https://youtu.be/mCCggAHRHGE)

Nonlinear MIMO Mass-Spring-Damper System with Physics Informed Neural Networks

Click the image above to watch the demo video

Project Overview

This project implements a Physics Informed Neural Network (PINN) to model and simulate a nonlinear Multiple-Input-Multiple-Output (MIMO) mass-spring-damper system. The implementation is being developed in MATLAB and is currently in early development stages.

System Description

The system consists of three identical point masses (m = 0.5 kg each) connected by nonlinear springs with three dampers (damping constant d = 0.25 N·s/m each). The system has the following characteristics:

Nonlinear spring stiffness: F(x) = kΔx + kₚΔx³ where k = 217 N/m and kₚ = 63.5 N/m³

Control inputs: u₁ and u₃ applied to the first and third masses

External disturbances: Represented as "dist" in the system

Output states: x₁ and x₃ positions (denoted as yₖᵣ) which need to be controlled

The system dynamics are governed by the following coupled nonlinear differential equations:

mẍ₁ = k(-2x₁ + x₂) + kₚ[-(x₁)³ + (x₂ - x₁)³] + d(ẋ₂ - 2ẋ₁) + u₁
mẍ₂ = k(x₁ - 2x₂ + x₃) + kₚ[(x₃ - x₂)³ - (x₂ - x₁)³] + d(ẋ₁ - 2ẋ₂) + dist
mẍ₃ = k(x₂ - x₃) + kₚ(x₂ - x₃)³ + d(ẋ₂ - ẋ₃) + u₃


Physics Informed Neural Networks Approach

This project uses Physics Informed Neural Networks (PINNs) to solve and predict the behavior of this nonlinear system. The PINN incorporates the physical laws directly into the neural network training process by:

- Encoding the nonlinear differential equations as part of the loss function

- Enforcing the physical constraints of the system

Current Status

This project is currently in early development. Additional features and documentation will be added as the project progresses.

Contact

For questions or feedback, please open an issue on GitHub or contact alvin.sutandar@gmail.com
