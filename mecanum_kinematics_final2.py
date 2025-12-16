"""
Mecanum drive kinematic simulator + animation (single axes, no explicit colors).

Adds per-wheel velocity arrows:
- Red  : omega_i >= 0  (forward along roller axis)
- Green: omega_i <  0  (backward along roller axis)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, FancyArrowPatch

# --------- Kinematics: transforms & matrices ---------
def T_G_to_R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def T_R_to_G(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def M_inverse_X(Lx, Ly):
    """Inverse kinematics matrix for 'X' pattern at 45° rollers (wheel rates from [vxR,vyR,omega])."""
    L = Lx + Ly
    return np.array([[ 1, -1, -L],
                     [ 1,  1,  L],
                     [ 1,  1, -L],
                     [ 1, -1,  L]], dtype=float)

def B_forward_X(Lx, Ly, r):
    """Forward kinematics matrix for 'X' pattern (body vel from wheel rates)."""
    L = Lx + Ly
    return np.array([[  r/4,     r/4,     r/4,     r/4   ],
                     [ -r/4,     r/4,     r/4,    -r/4   ],
                     [ -r/(4*L), r/(4*L), -r/(4*L), r/(4*L) ]], dtype=float)

# --------- Global <-> Wheel Kinematics ---------

def global_to_wheels(v_global, theta, Lx, Ly, r):
    """
    Calculates wheel velocities directly from global frame velocity.
    
    Args:
        v_global: np.array([v_Gx, v_Gy, omega_global])
        theta: Current robot heading (radians)
        Lx, Ly, r: Robot geometry parameters
    Returns:
        np.array([w1, w2, w3, w4])
    """
    # 1. Rotate Global Velocity to Robot Frame
    # v_Robot = R_global_to_robot * v_Global
    v_robot = T_G_to_R(theta) @ v_global
    
    # 2. Robot Frame Velocity to Wheel Velocities (Inverse Kinematics)
    # omega = M * v_Robot
    M = M_inverse_X(Lx, Ly)
    wheels = (1.0 / r) * (M @ v_robot)
    
    return wheels

def wheels_to_global(wheels, theta, Lx, Ly, r):
    """
    Calculates global frame velocity directly from wheel velocities.
    
    Args:
        wheels: np.array([w1, w2, w3, w4])
        theta: Current robot heading (radians)
        Lx, Ly, r: Robot geometry parameters
    Returns:
        np.array([v_Gx, v_Gy, omega_global])
    """
    # 1. Wheel Velocities to Robot Frame Velocity (Forward Kinematics)
    # v_Robot = B * omega
    B = B_forward_X(Lx, Ly, r)
    v_robot = B @ wheels
    
    # 2. Rotate Robot Frame Velocity to Global Frame
    # v_Global = R_robot_to_global * v_Robot
    v_global = T_R_to_G(theta) @ v_robot
    
    return v_global

# --------- Piecewise command schedules ---------
def schedule_global(dt):
    """
    Commands in GLOBAL frame: (duration, v_Gx, v_Gy, omega_Global)
    This demonstrates 'Field Oriented Control' where inputs are relative to the map, 
    not the robot's nose.
    """
    return [
        # Duration, vGx, vGy, omegaG
        (3.0,  0.5,  0.0,  0.0),  # Move East (Global X)
        (3.0,  0.0,  0.5,  0.0),  # Move North (Global Y) - Robot slides sideways if facing East
        (3.0,  0.0,  0.0,  0.5),  # Spin in place (Global Heading)
        (3.0,  0.4,  0.4,  0.0),  # Move North-East Diagonally
        (4.0,  0.0,  0.0,  0.0),  # Stop
    ]

def schedule_wheels_input(dt):
    """
    Direct Wheel Commands: (duration, w1, w2, w3, w4)
    Useful for testing Forward Kinematics (Odometry).
    """
    return [
        # Duration, FL, FR, RL, RR
        (3.0,   10.0,  10.0,  10.0,  10.0), # All wheels forward (Move Robot X)
        (3.0,  -10.0,  10.0,  10.0, -10.0), # Strafe (Move Robot Y)
        (3.0,  -10.0,  10.0, -10.0,  10.0), # Rotate (Robot Omega)
        (7.0,    0.0,   0.0,   0.0,   0.0), # Stop
    ]

def expand_schedule(seg_list, cols, N, dt):
    out = np.zeros((N, cols))
    idx = 0
    for seg in seg_list:
        dur = seg[0]
        steps = int(np.round(dur / dt))
        block = np.array(seg[1:], dtype=float).reshape(1, -1)
        end = min(N, idx + steps)
        out[idx:end, :] = block
        idx = end
        if idx >= N:
            break
    return out

# --------- Simulation core ---------
# Modified simulation loop demonstrating the usage
def simulate_new_modes(mode="global_to_wheels", T_total=16.0, dt=0.02, r=0.05, Lx=0.20, Ly=0.15):
    N = int(np.round(T_total / dt))
    
    # 1. Expand the schedule based on the selected mode
    if mode == "global_to_wheels":
        # Global Velocity Input (3 columns: vGx, vGy, omega)
        cmds = expand_schedule(schedule_global(dt), 3, N, dt)
    elif mode == "wheels_to_global":
        # Wheel Velocity Input (4 columns: w1, w2, w3, w4)
        cmds = expand_schedule(schedule_wheels_input(dt), 4, N, dt)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Initialize storage
    xs, ys, thetas = np.zeros(N), np.zeros(N), np.zeros(N)
    w_hist = np.zeros((N, 4))
    vG_hist = np.zeros((N, 3))
    
    # Initial State
    x, y, theta = 0.0, 0.0, 0.0

    for k in range(N):
        # Get the command for this timestep
        curr_cmd = cmds[k]

        if mode == "global_to_wheels":
            # Input: Global Vel -> Output: Wheel Vels
            v_global_cmd = curr_cmd # [vxG, vyG, omega]
            
            # Inverse Kinematics (Global -> Robot -> Wheels)
            w_cmd = global_to_wheels(v_global_cmd, theta, Lx, Ly, r)
            
            # Forward Kinematics (for verification/plotting)
            v_act = wheels_to_global(w_cmd, theta, Lx, Ly, r)

        elif mode == "wheels_to_global":
            # Input: Wheel Vels -> Output: Global Vel
            w_cmd = curr_cmd # [w1, w2, w3, w4]
            
            # Forward Kinematics (Wheels -> Robot -> Global)
            v_act = wheels_to_global(w_cmd, theta, Lx, Ly, r)

        # Integration (Dead Reckoning)
        x += v_act[0] * dt
        y += v_act[1] * dt
        theta += v_act[2] * dt
        
        # Store history
        xs[k], ys[k], thetas[k] = x, y, theta
        w_hist[k] = w_cmd
        vG_hist[k] = v_act

    return xs, ys, thetas, w_hist, vG_hist, dt

# --------- Animation ---------
def animate_path(xs, ys, thetas, w_hist, Lx, Ly, r, dt,
                 save_path_gif="mecanum_sim.gif", save_path_png="mecanum_final.png"):
    N = len(xs)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect('equal', adjustable='box')

    pad = 0.5 + max(Lx, Ly)
    xmin, xmax = np.min(xs)-pad, np.max(xs)+pad
    ymin, ymax = np.min(ys)-pad, np.max(ys)+pad
    if xmin == xmax: xmin, xmax = -1, 1
    if ymin == ymax: ymin, ymax = -1, 1

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Mecanum Drive Kinematic Simulation (global frame)")

    path_line, = ax.plot([], [], lw=1.5)

    body_len = 2*Lx*0.9
    body_wid = 2*Ly*0.9
    corners_init = np.array([[-body_len/2, -body_wid/2],
                             [ body_len/2, -body_wid/2],
                             [ body_len/2,  body_wid/2],
                             [-body_len/2,  body_wid/2]], dtype=float)
    body = Polygon(corners_init, closed=True, fill=False)
    ax.add_patch(body)

    wheel_offsets = np.array([[ Lx,  Ly],
                              [ Lx, -Ly],
                              [-Lx,  Ly],
                              [-Lx, -Ly]], dtype=float)

    wheel_patches = []
    for _ in range(4):
        c = Circle((0,0), 0.03, fill=False)
        ax.add_patch(c)
        wheel_patches.append(c)

    t_R = np.array([[ 1, -1],   # FL
                    [ 1,  1],   # FR
                    [ 1,  1],   # RL
                    [ 1, -1]],  # RR
                   dtype=float)
    t_R = (t_R.T / np.linalg.norm(t_R, axis=1)).T

    head_len = max(Lx, Ly) * 0.9
    heading_line, = ax.plot([], [], lw=1.2)

    arrow_patches = []
    for _ in range(4):
        arr = FancyArrowPatch(
            (0, 0), (0, 0),
            arrowstyle='->',
            mutation_scale=24,   # same head size
            lw=2.4,              # same shaft thickness
            shrinkA=0, shrinkB=0,
            connectionstyle='arc3,rad=0.0',
            zorder=5
        )

        arr.set_visible(False)
        ax.add_patch(arr)
        arrow_patches.append(arr)

    def R2G(theta, pts_R, center):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return (R @ pts_R.T).T + center

    def Rvec2G(theta, vec_R):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return R @ vec_R

    # bigger arrows
    arrow_scale = 0.80
    max_arrow   = 1.20 * max(Lx, Ly)

    def init():
        path_line.set_data([], [])
        heading_line.set_data([], [])
        body.set_xy(corners_init)
        for c in wheel_patches:
            c.center = (0,0)
        for arr in arrow_patches:
            arr.set_visible(False)
            arr.set_positions((0,0), (0,0))
        return [path_line, body, heading_line, *wheel_patches, *arrow_patches]

    def update(k):
        x, y, th = xs[k], ys[k], thetas[k]
        w = w_hist[k]

        path_line.set_data(xs[:k+1], ys[:k+1])

        corners_G = R2G(th, corners_init, np.array([x, y]))
        body.set_xy(corners_G)

        for i, c in enumerate(wheel_patches):
            p_G = R2G(th, wheel_offsets[i].reshape(1, 2), np.array([x, y]))[0]
            c.center = (p_G[0], p_G[1])

            v_lin = r * float(abs(w[i]))
            if v_lin < 1e-5:
                arrow_patches[i].set_visible(False)
                arrow_patches[i].set_positions((p_G[0], p_G[1]), (p_G[0], p_G[1]))
                continue

            sign = 1.0 if w[i] >= 0.0 else -1.0
            vec_R = sign * t_R[i] * min(arrow_scale * v_lin, max_arrow)
            vec_G = Rvec2G(th, vec_R)

            start = (p_G[0], p_G[1])
            end   = (p_G[0] + vec_G[0], p_G[1] + vec_G[1])

            arr = arrow_patches[i]
            arr.set_visible(True)
            arr.set_positions(start, end)
            arr.set_color('red' if w[i] >= 0.0 else 'green')

        head_R = np.array([[0, 0], [head_len, 0]])
        head_G = R2G(th, head_R, np.array([x, y]))
        heading_line.set_data(head_G[:, 0], head_G[:, 1])

        return [path_line, body, heading_line, *wheel_patches, *arrow_patches]

    ani = animation.FuncAnimation(
        fig, update, frames=N, init_func=init, blit=False, interval=1000*dt
    )

    try:
        from matplotlib.animation import PillowWriter
        ani.save(save_path_gif, writer=PillowWriter(fps=max(1, int(1/dt))))
        plt.close(fig)
        return True, save_path_gif
    except Exception:
        fig.savefig(save_path_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return False, save_path_png
# ---------Plots-------
def plot_results(dt, w_hist, vG_hist, Lx, Ly, r):
    """
    Plots Global Velocities, Robot (Body) Velocities, and Wheel Velocities vs Time.
    """
    N = w_hist.shape[0]
    time_axis = np.arange(N) * dt

    # 1. Recompute Body Frame Velocities (v_Rx, v_Ry, omega)
    # We use the Forward Kinematics matrix B: v_R = B @ w
    B = B_forward_X(Lx, Ly, r) 
    # w_hist is (N,4), we need (4,N) for matrix multiplication -> result (3,N) -> transpose to (N,3)
    vR_hist = (B @ w_hist.T).T

    # 2. Setup Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # --- Plot A: Global Velocities ---
    ax = axes[0]
    ax.plot(time_axis, vG_hist[:, 0], label=r'$v_{Gx}$ (Global X)', lw=2)
    ax.plot(time_axis, vG_hist[:, 1], label=r'$v_{Gy}$ (Global Y)', lw=2)
    ax.plot(time_axis, vG_hist[:, 2], label=r'$\omega$ (Heading Rate)', linestyle='--', color='orange')
    ax.set_ylabel("Global Vel [m/s, rad/s]")
    ax.set_title("Global Velocities vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # --- Plot B: Robot (Body) Velocities ---
    ax = axes[1]
    ax.plot(time_axis, vR_hist[:, 0], label=r'$v_{Rx}$ (Forward)', color='tab:blue', lw=2)
    ax.plot(time_axis, vR_hist[:, 1], label=r'$v_{Ry}$ (Strafe)', color='tab:green', lw=2)
    ax.plot(time_axis, vR_hist[:, 2], label=r'$\omega$ (Turn)', color='tab:orange', linestyle='--')
    ax.set_ylabel("Body Vel [m/s, rad/s]")
    ax.set_title("Robot (Body) Velocities vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # --- Plot C: Wheel Velocities ---
    ax = axes[2]
    labels = ['FL', 'FR', 'RL', 'RR'] 
    colors = ['red', 'green', 'blue', 'purple']
    
    for i in range(4):
        ax.plot(time_axis, w_hist[:, i], label=f'$\omega_{i+1}$ ({labels[i]})', color=colors[i], alpha=0.8, lw=1.5)
        
    ax.set_ylabel("Wheel Speed [rad/s]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Wheel Velocities vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=4)

    plt.tight_layout()
    plt.show()

# --------- Run a demo ---------
if __name__ == "__main__":
    r  = 0.05
    Lx = 0.20
    Ly = 0.15
    dt = 0.02
    T_total = 16.0

    # Choose your mode here:
    # mode = "global_to_wheels"   
    mode = "wheels_to_global" 

    print(f"Simulating Mode: {mode}")
    
    xs, ys, thetas, w_hist, vG_hist, dt = simulate_new_modes(
        mode=mode, T_total=T_total, dt=dt, r=r, Lx=Lx, Ly=Ly
    )
    
    print("Plotting velocities...")
    plot_results(dt, w_hist, vG_hist, Lx, Ly, r)
    
    print("Generating animation...")
    ok, path = animate_path(xs, ys, thetas, w_hist, Lx=Lx, Ly=Ly, r=r, dt=dt)
    print(f"Saved {'animation' if ok else 'final frame'}: {path}")

