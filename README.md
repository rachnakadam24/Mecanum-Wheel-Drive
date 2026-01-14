# Mecanum-Wheel-Drive

This repository contains a comprehensive simulation suite for a Mecanum-wheeled robot, organized into four standalone Python modules. The project systematically progresses from fundamental mathematical modeling (Kinematics & Dynamics) to advanced control strategies (Robust Sliding Mode Control) and fully autonomous navigation (A* Path Planning with Dynamic Obstacle Avoidance).

---

##  Repository Contents

### 1. Kinematics Simulation (`kinematics_sim.py`)
**Description:**
Validates the fundamental geometry and mathematical modeling of the 4-wheel Mecanum drive. It demonstrates the translation between **Global Map Coordinates** (what the planner sees) and **Individual Wheel Speeds** (what the motors need).

* **Inverse Kinematics (Global $\to$ Wheels):** Validates "Field Oriented Control" (Headless Mode).
* **Forward Kinematics (Wheels $\to$ Global):** Validates Odometry.
* **Visuals:** Generates an animated GIF showing velocity vectors on each wheel.
* **Usage:**
    * Uncomment `mode = "global_to_wheels"` to test Controller Logic.
    * Uncomment `mode = "wheels_to_global"` to test Odometry.

### 2. Dynamics Simulation (`dynamics_sim.py`)
**Description:**
Simulates the robot's physical behavior, acting as the "Plant" or "Real World" physics engine. Unlike the kinematics script, this accounts for **Mass, Inertia, Drag, and Friction**.

* **Newton-Euler Formulation:** Implements $\mathbf{M}\ddot{q} + \mathbf{C}\dot{q} = \mathbf{B}\tau$.
* **Reflected Inertia:** Accounts for the "virtual mass" added by spinning heavy wheels.
* **Dual Modes:**
    * **Force Mode:** Input = Body Force (N). Demonstrates inertia resisting acceleration.
    * **Torque Mode:** Input = Motor Torque (Nm). Demonstrates low-level motor actuation.
* **Usage:** Set `mode` to `"force"` or `"torque"` in the main block to visualize the response.

### 3. SMC Controller & Stability Analysis (`smc_controller.py`)
**Description:**
Implements a **Robust Sliding Mode Controller (SMC)** designed to force the robot to follow a complex reference trajectory (the letter 'R'). It includes mathematical proofs of stability.

* **Trajectory Tracking:** Follows a complex path (Line $\to$ Curve $\to$ Diagonal).
* **Robust Control:** Uses a `tanh(s)` boundary layer to eliminate chattering while fighting disturbances.
* **Stability Proof:** Real-time plotting of the **Lyapunov Function** $V(s)$ to proving convergence.
* **Usage:** Run the script to see two windows: one for tracking performance and one for stability analysis.

### 4. Final Integrated Simulation (`final_simulation.py`)
**Description:**
The final integration of the project, combining Kinematics, Dynamics, and the SMC Controller with high-level **Motion Planning**.

* **A* (A-Star) Planner:** Calculates optimal paths around static walls.
* **Dynamic Obstacle Avoidance:** Reactively re-plans paths if moving obstacles breach a 1.5m safety radius.
* **Variable Terrain:** Simulates "Ice" (Low Friction) and "Mud" (High Friction) zones to stress-test the controller.
* **Interactive GUI:**
    1.  **Click** on the map to set Goal points.
    2.  Click **"PLAN PATH"**.
    3.  Click **"START"**.

---

##  Installation & Setup

1.  **Clone the repository** (or download the files).
2.  **Install Dependencies:**
    This project requires Python 3.x and the following scientific libraries:
    ```bash
    pip install numpy matplotlib
    ```

##  How to Run

Each file is standalone. You can run them individually from your terminal:

```bash
# Run Kinematics Validation
python kinematics_sim.py

# Run Physics Engine
python dynamics_sim.py

# Run Controller Test
python smc_controller.py

# Run Full Autonomous Demo
python final_simulation.py
