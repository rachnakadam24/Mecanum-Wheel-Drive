import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from dataclasses import dataclass

# =============================
# Robot parameters & Helpers
# =============================
@dataclass
class MecanumParams:
    m: float = 20.0       # Mass [kg]
    Iz: float = 1.8       # Inertia [kg*m^2]
    L: float = 0.25       # Half-length [m]
    W: float = 0.20       # Half-width [m]
    r: float = 0.0762     # Wheel radius [m]
    Iw: float = 0.015     # Wheel inertia [kg*m^2]
    bw: float = 0.05      # Wheel friction [N*m*s/rad]
    Dvx: float = 2.0      # Body Drag X [N*s/m]
    Dvy: float = 2.0      # Body Drag Y [N*s/m]
    Dowl: float = 0.3     # Body Drag Yaw [N*m*s/rad]

def J_mecanum(L, W):
    a = (L + W)
    return np.array([
        [ 1.0, -1.0, -a], # FR
        [ 1.0,  1.0,  a], # FL
        [ 1.0,  1.0, -a], # RL
        [ 1.0, -1.0,  a], # RR
    ], float)

def rotation2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

# =============================
# INPUT: BODY FORCES (The "Driver")
# =============================
def force_schedule(t):
    """
    Defines the FORCE applied to the robot body over time.
    Format: [Fx (N), Fy (N), Torque (Nm)]
    """
    # 1. Push Forward (0-3s)
    if t < 3.0:
        return np.array([15.0, 0.0, 0.0]) 
    
    # 2. Coast (3-5s) - Watch drag slow it down
    elif t < 5.0:
        return np.array([0.0, 0.0, 0.0])
        
    # 3. Push Sideways / Strafe (5-8s)
    elif t < 8.0:
        return np.array([0.0, -15.0, 0.0]) 
        
    # 4. Spin (8-11s)
    elif t < 11.0:
        return np.array([0.0, 0.0, 6.0])   
        
    # 5. Stop pushing
    else:
        return np.array([0.0, 0.0, 0.0])

def torque_schedule(t):
    """
    Input: Time
    Output: [Tau_FL, Tau_FR, Tau_RL, Tau_RR] in Nm
    """
    # 1. Drive Forward (All wheels positive)
    if t < 3.0:
        return np.array([2.0, 2.0, 2.0, 2.0]) 
        
    # 2. Spin in Place (Left side neg, Right side pos)
    elif t < 6.0:
        return np.array([-2.0, 2.0, -2.0, 2.0])
        
    # 3. Stop
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])
# =============================
# Simulation (Open Loop / Inverse Dynamics)
# =============================
def simulate_open_loop(params, mode="force", T=12.0, dt=0.01):
    # --- Matrix Setup (Same as before) ---
    J = J_mecanum(params.L, params.W)
    r = params.r
    B_mat = (1.0 / r) * J.T              # Maps Torque -> Force
    B_pinv = np.linalg.pinv(B_mat)       # Maps Force -> Torque

    # Plant Physics Matrices
    JTJ = J.T @ J
    M_total = np.diag([params.m, params.m, params.Iz]) + (params.Iw / (r**2)) * JTJ
    C_total = np.diag([params.Dvx, params.Dvy, params.Dowl]) + (params.bw / (r**2)) * JTJ

    # Time setup
    N = int(T / dt)
    t_hist = np.linspace(0, T, N)
    
    # Storage
    X_hist = np.zeros((N, 6))
    phi_dot_hist = np.zeros((N, 4))
    # We store 4 values for command now to handle torque inputs
    cmd_hist = np.zeros((N, 4)) 
    
    # Initial State
    x_state = np.zeros(6) 

    for k, t in enumerate(t_hist):
        q = x_state[0:3]
        qdot = x_state[3:6]
        
        # ==========================================
        #       THE SWITCHING LOGIC
        # ==========================================
        if mode == "force":
            # INVERSE DYNAMICS:
            # 1. Input = Desired Body Force
            F_cmd = force_schedule(t) 
            # 2. Convert to Torque
            tau_motor = B_pinv @ F_cmd
            
            # Store for plotting (pad F_cmd to 4 values)
            cmd_hist[k, :3] = F_cmd

        elif mode == "torque":
            # FORWARD DYNAMICS:
            # 1. Input = Direct Motor Torque
            tau_motor = torque_schedule(t)
            
            # Store for plotting
            cmd_hist[k, :] = tau_motor

        # ==========================================
        #       PHYSICS (Always Forward)
        # ==========================================
        
        # Calculate Net Force on Body from these Torques
        F_propulsion = B_mat @ tau_motor
        
        # Solve Motion: M * a = F_prop - Friction
        forcing = F_propulsion - (C_total @ qdot)
        qddot = np.linalg.solve(M_total, forcing)
        
        # Integration
        qdot_next = qdot + qddot * dt
        Rb = rotation2d(q[2])
        v_world = Rb @ qdot_next[0:2]
        q_next = q.copy()
        q_next[0] += v_world[0] * dt
        q_next[1] += v_world[1] * dt
        q_next[2] += qdot_next[2] * dt
        
        # Store Data
        x_state = np.concatenate([q_next, qdot_next])
        X_hist[k, :] = x_state
        phi_dot_hist[k, :] = (J @ qdot_next) / r
        
    return t_hist, X_hist, phi_dot_hist, cmd_hist
# =============================
# Animation 
# =============================
def animate_simulation(t, X, phi_dot, params, skip=3):
    
    # Downsample for smoother playback speed
    idx = np.arange(0, len(t), skip)
    t_a, X_a, phi_a = t[idx], X[idx, :], phi_dot[idx, :]
    
    # FIXED: Use np.max on the array, not python max()
    pad = 1.0
    ax_lim = np.max(np.abs(X[:, 0:2])) + pad 

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Open Loop Dynamics (Input = Force)")
    ax.grid(True, alpha=0.3)

    # Robot Body (Rectangle)
    body_pts = np.array([
        [ params.L,  params.W], [ params.L, -params.W],
        [-params.L, -params.W], [-params.L,  params.W]
    ])
    body_poly = Polygon(body_pts, closed=True, fill=False, lw=2)
    ax.add_patch(body_poly)
    
    # Wheels (Circles)
    wheel_offsets = np.array([
        [ params.L, -params.W], [ params.L,  params.W], # FR, FL
        [-params.L,  params.W], [-params.L, -params.W]  # RL, RR
    ])
    wheels = [Circle((0,0), 0.05, color='k', fill=False) for _ in range(4)]
    for w in wheels: ax.add_patch(w)

    # Arrows (Wheel Velocities)
    arrows = [FancyArrowPatch((0,0),(0,0), arrowstyle='->', mutation_scale=15, color='r') for _ in range(4)]
    for arr in arrows: ax.add_patch(arr)
    
    # Path Trace
    trace, = ax.plot([], [], 'b-', lw=1, alpha=0.5)
    
    # Roller Direction Vectors (for arrows)
    roller_dirs = np.array([[1, -1], [1, 1], [1, 1], [1, -1]])
    roller_dirs = (roller_dirs.T / np.linalg.norm(roller_dirs, axis=1)).T

    def update(frame):
        x, y, th = X_a[frame, 0], X_a[frame, 1], X_a[frame, 2]
        
        # Rotate Body Points
        R = rotation2d(th)
        body_world = (R @ body_pts.T).T + np.array([x, y])
        body_poly.set_xy(body_world)
        
        # Update Trace
        trace.set_data(X_a[:frame, 0], X_a[:frame, 1])
        
        # Update Wheels & Arrows
        for i in range(4):
            # Wheel Center
            wc_local = wheel_offsets[i]
            wc_world = (R @ wc_local) + np.array([x, y])
            wheels[i].center = wc_world
            
            # Arrow (Direction & Magnitude)
            w_speed = phi_a[frame, i]
            scale = 0.05 * w_speed # Scaling factor for visualization
            
            if abs(w_speed) > 0.1:
                arrows[i].set_visible(True)
                # Arrow points along roller axis
                direction = roller_dirs[i] * (1 if w_speed >= 0 else -1)
                
                # Rotate arrow direction to world frame
                arrow_vec = (R @ direction) * abs(scale)
                arrows[i].set_positions(wc_world, wc_world + arrow_vec)
                arrows[i].set_color('red' if w_speed >= 0 else 'green')
            else:
                arrows[i].set_visible(False)
                
        return [body_poly, trace, *wheels, *arrows]

    ani = animation.FuncAnimation(fig, update, frames=len(t_a), interval=20, blit=False)
    
    # Display the animation
    plt.show()

# =============================
# Main
# =============================
if __name__ == "__main__":
    params = MecanumParams()
    
    # TOGGLE MODE HERE
    current_mode = "torque"  # Options: "force" or "torque"
    # current_mode = "force"
    
    print(f"Simulating in {current_mode} mode...")
    t, X, phi_dot, cmds = simulate_open_loop(params, mode=current_mode)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot 1: Inputs
    ax = axes[0]
    if current_mode == "force":
        ax.plot(t, cmds[:,0], label='Fx [N]')
        ax.plot(t, cmds[:,1], label='Fy [N]')
        ax.plot(t, cmds[:,2], label='Torque [Nm]')
        ax.set_title("Input: Body Forces")
    else:
        ax.plot(t, cmds[:,0], label='Tau FL')
        ax.plot(t, cmds[:,1], label='Tau FR')
        ax.plot(t, cmds[:,2], label='Tau RL')
        ax.plot(t, cmds[:,3], label='Tau RR')
        ax.set_title("Input: Wheel Torques [Nm]")
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Velocity
    ax = axes[1]
    ax.plot(t, X[:, 3], label='Vx')
    ax.plot(t, X[:, 4], label='Vy')
    ax.plot(t, X[:, 5], label='Omega')
    ax.set_title("Resulting Robot Velocity")
    ax.legend()
    ax.grid(True)
    
    
    print("Animating...")
    animate_simulation(t, X, phi_dot, params, skip=4)