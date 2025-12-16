import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from dataclasses import dataclass

# ==========================================
# 1. System Parameters
# ==========================================
@dataclass
class MecanumParams:
    # Physics
    m: float = 20.0       # Mass [kg]
    Iz: float = 1.8       # Rotational Inertia [kg*m^2]
    L: float = 0.25       # Half-length [m]
    W: float = 0.20       # Half-width [m]
    r: float = 0.0762     # Wheel radius [m]
    Iw: float = 0.015     # Wheel inertia [kg*m^2]
    
    # Friction / Damping
    bw: float = 0.05      # Wheel bearing friction
    Dvx: float = 2.0      # Linear Drag X
    Dvy: float = 2.0      # Linear Drag Y
    Dowl: float = 0.3     # Rotational Drag

    # Controller Gains (Sliding Mode)
    lambda_xy: float = 4.0   # Sliding surface slope (Position convergence)
    lambda_th: float = 4.0   # Sliding surface slope (Heading convergence)
    k_xy: float = 10.0       # Robust gain (Position)
    k_th: float = 5.0        # Robust gain (Heading)
    boundary: float = 0.1    # Smoothing boundary layer (tanh width)

# ==========================================
# 2. Math Helpers
# ==========================================
def rotation2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def J_mecanum(L, W):
    """Jacobian: Body Velocity -> Wheel Velocity"""
    sum_dim = L + W
    return np.array([
        [1, -1, -sum_dim],
        [1,  1,  sum_dim],
        [1,  1, -sum_dim],
        [1, -1,  sum_dim]
    ])

# ==========================================
# 3. Trajectory Generator: The Letter 'R'
# ==========================================
def generate_R_trajectory(t):
    """
    Returns desired [x, y, theta], [vx, vy, omega], [ax, ay, alpha]
    for a path tracing the letter 'R'.
    """
    # Timing for segments
    t_spine = 4.0
    t_loop  = 4.0
    t_leg   = 3.0
    t_wait  = 1.0
    
    # State initialization
    qd  = np.zeros(3) # pos
    dqd = np.zeros(3) # vel
    ddqd = np.zeros(3)# acc

    # Segment 1: Vertical Spine (0,0) -> (0, 2)
    if t < t_spine:
        # Simple linear interpolation
        ratio = t / t_spine
        qd = np.array([0.0, 2.0 * ratio, 0.0])
        dqd = np.array([0.0, 2.0 / t_spine, 0.0])
        
    # Segment 2: The Loop (Semi-circle from (0,2) to (0,1))
    elif t < (t_spine + t_loop):
        # Local time in this segment
        tau = t - t_spine
        # Angle goes from 90 deg (pi/2) to -90 deg (-pi/2)
        # Center of circle is at (0, 1.5), Radius is 0.5? 
        # Actually let's make a nicer loop: Center (0, 1.5), R=0.8, sweeping -pi/2 to 3pi/2?
        # Let's do simple parametric arc:
        # Start (0,2), Out to (0.8, 1.5), Back to (0,1)
        
        # Angle phi goes from pi/2 to -pi/2
        phi = (np.pi/2) - (np.pi * (tau / t_loop))
        dphi = -np.pi / t_loop
        
        R_arc = 0.6
        cx, cy = 0.0, 1.4
        
        # But we need it to start at (0,2). 
        # At phi=pi/2: x=0, y=1.4+0.6=2. Correct.
        # At phi=-pi/2: x=0, y=1.4-0.6=0.8. Close enough to middle.
        
        qd[0] = cx + R_arc * np.cos(phi)
        qd[1] = cy + R_arc * np.sin(phi)
        qd[2] = 0.0 # Maintain heading 0
        
        dqd[0] = -R_arc * np.sin(phi) * dphi
        dqd[1] =  R_arc * np.cos(phi) * dphi

    # Segment 3: The Diagonal Leg (0, 0.8) -> (1, 0)
    elif t < (t_spine + t_loop + t_leg):
        tau = t - (t_spine + t_loop)
        ratio = tau / t_leg
        
        start_pt = np.array([0.0, 0.8])
        end_pt   = np.array([1.2, 0.0])
        
        vec = end_pt - start_pt
        
        qd[0] = start_pt[0] + vec[0] * ratio
        qd[1] = start_pt[1] + vec[1] * ratio
        dqd[0] = vec[0] / t_leg
        dqd[1] = vec[1] / t_leg
        
    # Stop
    else:
        qd = np.array([1.2, 0.0, 0.0])
        dqd = np.zeros(3)

    return qd, dqd, ddqd

# ==========================================
# 4. Controller & Simulation
# ==========================================
def sliding_mode_control(q, dq, qd, dqd, ddqd, params, M, C):
    """
    Computes body forces F using SMC.
    Control Law: u = M(ddqd + lambda*de + K*tanh(s)) + C*dq
    """
    # 1. Error
    e  = qd - q
    de = dqd - dq
    
    # 2. Sliding Surface (s = de + lambda*e)
    # We define separate lambdas for linear (xy) and angular (th)
    Lam = np.diag([params.lambda_xy, params.lambda_xy, params.lambda_th])
    s = de + Lam @ e
    
    # 3. Robust Switching Term (tanh instead of sign to prevent chattering)
    # We use a boundary layer 'phi' to smooth the transition
    phi = params.boundary
    K_gain = np.diag([params.k_xy, params.k_xy, params.k_th])
    
    v_switch = np.tanh(s / phi)
    
    # 4. Compute Required Acceleration (v)
    # This is the term (ddqd + lambda*de + K*switch)
    accel_term = ddqd + Lam @ de + K_gain @ v_switch
    
    # 5. Compute Force (Inverse Dynamics)
    # F = M * accel + C * dq
    F_body = M @ accel_term + C @ dq
    
    return F_body, s

def simulate_R_path(params, T=12.0, dt=0.01):
    # --- Matrix Setup ---
    J = J_mecanum(params.L, params.W)
    r = params.r
    
    # 1. Mass Matrix (M_total = Body + Reflected Wheel Inertia)
    JTJ = J.T @ J
    M_body = np.diag([params.m, params.m, params.Iz])
    M_refl = (params.Iw / (r**2)) * JTJ
    M_total = M_body + M_refl
    
    # 2. Damping Matrix (C_total = Body Drag + Wheel Friction)
    C_body = np.diag([params.Dvx, params.Dvy, params.Dowl])
    C_refl = (params.bw / (r**2)) * JTJ
    C_total = C_body + C_refl
    
    # 3. Input Matrix (B maps Wheel Torque -> Body Force)
    B_mat = (1.0/r) * J.T
    B_pinv = np.linalg.pinv(B_mat)

    # --- Loop Setup ---
    N = int(T / dt)
    t_hist = np.linspace(0, T, N)
    
    # State: [x, y, th, vx, vy, om]
    x_state = np.zeros(6) 
    
    # Data logging
    X_log = np.zeros((N, 6))
    Xd_log = np.zeros((N, 3)) # Desired pos
    phi_dot_log = np.zeros((N, 4))
    
    for k, t in enumerate(t_hist):
        # Extract State
        q = x_state[:3]
        dq = x_state[3:]
        
        # 1. Get Trajectory Command
        qd, dqd, ddqd = generate_R_trajectory(t)
        
        # 2. Sliding Mode Controller
        # Calculate necessary Body Force to track trajectory
        F_cmd, s_val = sliding_mode_control(q, dq, qd, dqd, ddqd, params, M_total, C_total)
        
        # 3. Force Allocation (Torques)
        tau = B_pinv @ F_cmd
        
        # 4. Plant Dynamics (Forward Simulation)
        # Apply Torque -> Get Acceleration
        F_propulsion = B_mat @ tau
        forcing = F_propulsion - (C_total @ dq)
        ddq = np.linalg.solve(M_total, forcing)
        
        # 5. Integration
        dq_next = dq + ddq * dt
        
        # Kinematics (World Frame)
        Rb = rotation2d(q[2])
        v_world = Rb @ dq_next[:2]
        q_next = q.copy()
        q_next[:2] += v_world * dt
        q_next[2]  += dq_next[2] * dt
        
        # Update
        x_state = np.concatenate([q_next, dq_next])
        
        # Log
        X_log[k, :] = x_state
        Xd_log[k, :] = qd
        phi_dot_log[k, :] = (J @ dq_next) / r

    return t_hist, X_log, Xd_log, phi_dot_log

# ==========================================
# 5. Animation (Robust Version)
# ==========================================
def animate_simulation(t, X, Xd, phi_dot, params, skip=3):
    idx = np.arange(0, len(t), skip)
    t_a, X_a, Xd_a, phi_a = t[idx], X[idx, :], Xd[idx, :], phi_dot[idx, :]
    
    # Fix Plot Limits
    all_x = np.concatenate([X[:,0], Xd[:,0]])
    all_y = np.concatenate([X[:,1], Xd[:,1]])
    pad = 0.5
    xmin, xmax = np.min(all_x)-pad, np.max(all_x)+pad
    ymin, ymax = np.min(all_y)-pad, np.max(all_y)+pad

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("SMC Dynamics Tracking: Letter 'R'")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3)

    # Shapes
    body_pts = np.array([
        [ params.L,  params.W], [ params.L, -params.W],
        [-params.L, -params.W], [-params.L,  params.W]
    ])
    body_poly = Polygon(body_pts, closed=True, fill=False, lw=2, ec='black')
    ax.add_patch(body_poly)
    
    # Desired Path (Dashed Line)
    ax.plot(Xd[:,0], Xd[:,1], 'r--', lw=1, alpha=0.5, label='Reference (R)')
    
    # Actual Path (Solid Line)
    trace, = ax.plot([], [], 'b-', lw=1.5, label='Actual SMC')
    ax.legend(loc='upper right')

    # Wheels
    wheel_offsets = np.array([
        [ params.L, -params.W], [ params.L,  params.W], 
        [-params.L,  params.W], [-params.L, -params.W]  
    ])
    wheels = [Circle((0,0), 0.05, color='k', fill=True) for _ in range(4)]
    for w in wheels: ax.add_patch(w)

    # Velocity Arrows
    arrows = [FancyArrowPatch((0,0),(0,0), arrowstyle='->', mutation_scale=15) for _ in range(4)]
    for arr in arrows: ax.add_patch(arr)
    
    roller_dirs = np.array([[1, -1], [1, 1], [1, 1], [1, -1]])
    roller_dirs = (roller_dirs.T / np.linalg.norm(roller_dirs, axis=1)).T

    def update(frame):
        x, y, th = X_a[frame, 0], X_a[frame, 1], X_a[frame, 2]
        
        # Robot Body
        R = rotation2d(th)
        body_world = (R @ body_pts.T).T + np.array([x, y])
        body_poly.set_xy(body_world)
        
        # Trace
        trace.set_data(X_a[:frame, 0], X_a[:frame, 1])
        
        # Wheels & Arrows
        for i in range(4):
            wc_local = wheel_offsets[i]
            wc_world = (R @ wc_local) + np.array([x, y])
            wheels[i].center = wc_world
            
            w_speed = phi_a[frame, i]
            
            # Show arrow if moving
            if abs(w_speed) > 0.1:
                arrows[i].set_visible(True)
                direction = roller_dirs[i] * np.sign(w_speed)
                vec = (R @ direction) * 0.02 * abs(w_speed) # Scale length by speed
                arrows[i].set_positions(wc_world, wc_world + vec)
                arrows[i].set_color('red' if w_speed > 0 else 'green')
            else:
                arrows[i].set_visible(False)
                
        return [body_poly, trace, *wheels, *arrows]

    ani = animation.FuncAnimation(fig, update, frames=len(t_a), interval=20, blit=False)
    plt.show()

def plot_stability_analysis(t, X, params):
    """
    Re-calculates the sliding surface 's' and Lyapunov function 'V' 
    from the simulation history to verify stability.
    """
    # 1. Re-initialize containers
    N = len(t)
    s_hist = np.zeros((N, 3))
    V_hist = np.zeros(N)
    
    # Diagonal Gains Matrix for Lambda
    Lam = np.diag([params.lambda_xy, params.lambda_xy, params.lambda_th])

    # 2. Re-compute errors for every time step
    for k in range(N):
        # Current State
        q = X[k, :3]
        dq = X[k, 3:]
        
        # Desired State (Must call generator again to get velocity dqd)
        qd, dqd, _ = generate_R_trajectory(t[k])
        
        # Calculate Errors
        e = qd - q
        de = dqd - dq
        
        # Calculate Sliding Surface: s = de + Lambda * e
        s = de + Lam @ e
        
        # Calculate Lyapunov Function: V = 0.5 * s^T * s
        V = 0.5 * np.dot(s.T, s)
        
        s_hist[k, :] = s
        V_hist[k] = V

    # 3. Plot Results
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot A: Sliding Surface (s) convergence
    axes[0].plot(t, s_hist[:, 0], label='$s_x$')
    axes[0].plot(t, s_hist[:, 1], label='$s_y$')
    axes[0].plot(t, s_hist[:, 2], label='$s_{\\theta}$')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_title("Sliding Surface Convergence ($s \\to 0$)")
    axes[0].set_ylabel("Surface Value $s$")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot B: Lyapunov Function (V) decay
    axes[1].plot(t, V_hist, color='purple', lw=2, label='$V = 0.5 s^T s$')
    axes[1].set_title("Lyapunov Function Decay ($\dot{V} < 0$)")
    axes[1].set_ylabel("Energy $V$")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    params = MecanumParams()
    print("Simulating SMC Tracking of Letter 'R'...")
    
    # 1. Run Simulation
    t, X, Xd, phi = simulate_R_path(params, T=12.0)
    
    # 2. Plot Tracking Performance
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, X[:,0], label='x')
    axes[0].plot(t, X[:,1], label='y')
    axes[0].plot(t, Xd[:,0], 'k--', label='Ref x')
    axes[0].plot(t, Xd[:,1], 'k:', label='Ref y')
    axes[0].set_title("Position Tracking")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(t, Xd[:,0] - X[:,0], label='Error X')
    axes[1].plot(t, Xd[:,1] - X[:,1], label='Error Y')
    axes[1].set_title("Tracking Error (SMC)")
    axes[1].set_ylabel("Error [m]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Animate
    animate_simulation(t, X, Xd, phi, params, skip=4)

    params = MecanumParams()
    
    # 1. Run Simulation
    t, X, Xd, phi = simulate_R_path(params, T=12.0)
    
    # 2. Run Stability Analysis
    print("Generating Stability Analysis Plots...")
    plot_stability_analysis(t, X, params)