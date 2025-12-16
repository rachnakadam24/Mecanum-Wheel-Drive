import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, Rectangle
from dataclasses import dataclass
import heapq
import random

# ==========================================
# 1. System Parameters
# ==========================================
@dataclass
class MecanumParams:
    m: float = 20.0       
    Iz: float = 1.8       
    L: float = 0.25       
    W: float = 0.20       
    r: float = 0.0762     
    Iw: float = 0.015     
    
    # Nominal Friction
    bw: float = 0.05      
    Dvx: float = 2.0      
    Dvy: float = 2.0      
    Dowl: float = 0.3     

    # Robust Controller Gains
    lambda_xy: float = 4.0   
    lambda_th: float = 4.0   
    k_xy: float = 20.0        
    k_th: float = 10.0        
    boundary: float = 0.1    

def rotation2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def J_mecanum(L, W):
    sum_dim = L + W
    return np.array([[1, -1, -sum_dim], [1, 1, sum_dim], [1, 1, -sum_dim], [1, -1, sum_dim]])

def get_matrices(params, friction_mult=1.0):
    J = J_mecanum(params.L, params.W)
    JTJ = J.T @ J
    r = params.r
    
    M_total = np.diag([params.m, params.m, params.Iz]) + (params.Iw / (r**2)) * JTJ
    
    C_body = np.diag([params.Dvx, params.Dvy, params.Dowl]) * friction_mult
    C_refl = (params.bw / (r**2)) * JTJ * friction_mult
    C_total = C_body + C_refl
    
    B_mat = (1.0/r) * J.T
    B_pinv = np.linalg.pinv(B_mat)
    
    return M_total, C_total, B_mat, B_pinv

# ==========================================
# 2. Environment Objects
# ==========================================
class RoamingObstacle:
    def __init__(self, start_pos, radius, bounds, speed=0.02):
        self.pos = np.array(start_pos, dtype=float)
        self.radius = radius
        self.speed = speed
        self.bounds = bounds # (min_x, max_x, min_y, max_y)
        self.target = self.pick_new_target()
        
    def pick_new_target(self):
        # Pick a random point within bounds (with buffer)
        pad = 1.0
        tx = random.uniform(self.bounds[0]+pad, self.bounds[1]-pad)
        ty = random.uniform(self.bounds[2]+pad, self.bounds[3]-pad)
        return np.array([tx, ty])

    def update(self):
        # Vector to target
        vec = self.target - self.pos
        dist = np.linalg.norm(vec)
        
        if dist < self.speed:
            # Reached target, pick new one
            self.target = self.pick_new_target()
        else:
            # Move towards target
            direction = vec / dist
            self.pos += direction * self.speed

@dataclass
class FrictionZone:
    x: float; y: float; w: float; h: float
    mult: float; label: str; color: str
    def contains(self, px, py): return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

# ==========================================
# 3. Path Planning & Control
# ==========================================
class AStarPlanner:
    def __init__(self, x_lim, y_lim, resolution=0.25, robot_radius=0.4):
        self.res = resolution; self.rr = robot_radius
        self.min_x, self.max_x = x_lim; self.min_y, self.max_y = y_lim
        self.x_w = int(round((self.max_x - self.min_x)/self.res))
        self.y_w = int(round((self.max_y - self.min_y)/self.res))
        self.static_obs = []

    def set_static_obstacles(self, obs_list): self.static_obs = obs_list
    def calc_idx(self, p, min_p): return int(round((p - min_p)/self.res))
    def calc_pos(self, i, min_p): return i * self.res + min_p

    def verify(self, node, dyn_obs):
        px = self.calc_pos(node.x, self.min_x); py = self.calc_pos(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y: return False
        for (ox, oy, r) in self.static_obs:
            if np.hypot(px-ox, py-oy) <= (r + self.rr): return False
        for o in dyn_obs:
            if np.hypot(px-o.pos[0], py-o.pos[1]) <= (o.radius + self.rr + 0.3): return False
        return True

    class Node:
        def __init__(self, x, y, c, p): self.x=x; self.y=y; self.c=c; self.p=p
        def __lt__(self, o): return self.c < o.c

    def plan(self, sx, sy, gx, gy, dyn_obs=[]):
        sn = self.Node(self.calc_idx(sx, self.min_x), self.calc_idx(sy, self.min_y), 0, -1)
        gn = self.Node(self.calc_idx(gx, self.min_x), self.calc_idx(gy, self.min_y), 0, -1)
        
        opens = []; heapq.heappush(opens, (0, sn))
        visited = dict(); visited[(sn.x, sn.y)] = sn
        
        moves = [(1,0,1), (0,1,1), (-1,0,1), (0,-1,1), (1,1,1.4), (1,-1,1.4), (-1,1,1.4), (-1,-1,1.4)]

        while opens:
            cost, curr = heapq.heappop(opens)
            if curr.x == gn.x and curr.y == gn.y: gn.p = curr.p; break

            for dx, dy, dc in moves:
                nx, ny = curr.x + dx, curr.y + dy
                node = self.Node(nx, ny, curr.c + dc, curr)
                if not self.verify(node, dyn_obs): continue
                if (nx, ny) not in visited or visited[(nx, ny)].c > node.c:
                    visited[(nx, ny)] = node
                    heapq.heappush(opens, (node.c + np.hypot(nx-gn.x, ny-gn.y), node))
                    
        if gn.p == -1: return None
        rx, ry = [], []
        curr = visited.get((gn.x, gn.y), None)
        while curr:
            rx.append(self.calc_pos(curr.x, self.min_x))
            ry.append(self.calc_pos(curr.y, self.min_y))
            curr = curr.p if isinstance(curr.p, self.Node) else None
        return np.array(list(zip(rx[::-1], ry[::-1])))

def gen_traj(path, avg_vel=1.5):
    if path is None or len(path) < 2: return None, 0
    dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
    times = dists / avg_vel
    t_tot = np.sum(times)
    t_acc = np.insert(np.cumsum(times), 0, 0.0)
    
    def get_st(t):
        if t >= t_tot: return np.array([path[-1,0], path[-1,1], 0]), np.zeros(3), np.zeros(3)
        i = max(0, min(np.searchsorted(t_acc, t)-1, len(times)-1))
        dt = t_acc[i+1] - t_acc[i]
        frac = (t - t_acc[i]) / dt
        p = path[i] + (path[i+1]-path[i])*frac
        v = (path[i+1]-path[i])/dt
        return np.array([p[0], p[1], 0]), np.array([v[0], v[1], 0]), np.zeros(3)
    return get_st, t_tot

def smc_control(q, dq, qd, dqd, ddqd, params, M_nom, C_nom):
    e, de = qd - q, dqd - dq
    Lam = np.diag([params.lambda_xy, params.lambda_xy, params.lambda_th])
    K = np.diag([params.k_xy, params.k_xy, params.k_th]) 
    s = de + Lam @ e
    return M_nom @ (ddqd + Lam @ de + K @ np.tanh(s/params.boundary)) + C_nom @ dq

# ==========================================
# 4. Modern GUI App
# ==========================================
class ModernApp:
    def __init__(self):
        self.params = MecanumParams()
        self.M_nom, self.C_nom, self.B, self.B_pinv = get_matrices(self.params, friction_mult=1.0)
        
        self.planner = AStarPlanner((-2, 12), (-2, 12), resolution=0.25)
        self.planner.set_static_obstacles([(3, 3, 1.2), (7, 7, 1.5), (3, 8, 1.0), (8, 2, 1.0)])
        
        # Roaming Obstacles (Random movement within bounds)
        self.dyn_obs = [
            RoamingObstacle((1, 5), 0.6, bounds=(-1, 11, -1, 11), speed=0.03), 
            RoamingObstacle((6, 1), 0.6, bounds=(-1, 11, -1, 11), speed=0.03)
        ]
        
        self.zones = [
            FrictionZone(1, 8, 3, 3, 0.1, "Ice", "#a8e6cf"),      
            FrictionZone(8, 4, 3, 3, 3.0, "Sand", "#fdcb6e"),    
            FrictionZone(4, 0, 3, 2, 5.0, "Mud", "#634228")      
        ]

        self.c = {
            'bg': '#2d3436', 'map': '#353b48', 'grid': '#636e72',
            'obs': '#d63031', 'dyn': '#0984e3', 'path': '#00b894', 
            'trace': '#74b9ff', 'robot': '#a29bfe', 'goal': '#fab1a0',
            'text': '#dfe6e9', 'wheel': '#2d3436'
        }
        
        self.fig = plt.figure(figsize=(10, 7), facecolor=self.c['bg'])
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
        
        self.ax = self.fig.add_subplot(gs[0])
        self.ax.set_facecolor(self.c['map'])
        self.ax.set_xlim(-2, 12); self.ax.set_ylim(-2, 12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color=self.c['grid'], linestyle='--', alpha=0.5)
        self.ax.set_title("Robust Mecanum Controller vs Variable Friction", color=self.c['text'])
        
        for z in self.zones:
            rect = Rectangle((z.x, z.y), z.w, z.h, color=z.color, alpha=0.5)
            self.ax.add_patch(rect)
            self.ax.text(z.x + 0.2, z.y + z.h - 0.5, z.label, color='white', fontsize=9, fontweight='bold')

        for (ox, oy, r) in self.planner.static_obs:
            self.ax.add_patch(Circle((ox, oy), r, color=self.c['obs'], alpha=0.8, ec='black'))
            
        self.dyn_patches = [Circle(o.pos, o.radius, color=self.c['dyn'], alpha=0.9, ec='white') for o in self.dyn_obs]
        for p in self.dyn_patches: self.ax.add_patch(p)

        self.path_line, = self.ax.plot([], [], '--', color=self.c['path'], lw=3, label='Planned')
        self.trace_line, = self.ax.plot([], [], '-', color=self.c['trace'], lw=2, label='Actual')
        
        # --- Robot Visuals (Mecanum Style) ---
        # Main Body
        self.robot_body = Polygon(self.get_body_corners(0,0,0), closed=True, fc=self.c['robot'], ec='white')
        self.ax.add_patch(self.robot_body)
        # 4 Wheels
        self.wheels = []
        for _ in range(4):
            w = Polygon(np.zeros((4,2)), closed=True, fc=self.c['wheel'], ec='white', lw=1)
            self.ax.add_patch(w); self.wheels.append(w)
        
        self.goals = []; self.goal_artists = []
        self.path = None; self.traj_func = None; self.sim_running = False
        self.q = np.zeros(3); self.dq = np.zeros(3); self.curr_goal = 0; self.t_curr = 0.0

        # Controls
        ax_ctrl = self.fig.add_subplot(gs[1]); ax_ctrl.axis('off')
        ax_ctrl.text(0.1, 0.9, "INSTRUCTIONS:\n1. Click Map for Goals\n2. Plan Path\n3. Start Simulation\n\nObstacles roam freely!", 
                     color=self.c['text'], fontsize=10, va='top')
        
        btn_bg = '#b2bec3'; hover = '#dfe6e9'
        self.b_plan = Button(plt.axes([0.75, 0.5, 0.15, 0.08]), 'PLAN PATH', color=btn_bg, hovercolor=hover)
        self.b_start = Button(plt.axes([0.75, 0.38, 0.15, 0.08]), 'START', color=btn_bg, hovercolor=hover)
        self.b_reset = Button(plt.axes([0.75, 0.26, 0.15, 0.08]), 'RESET', color=btn_bg, hovercolor=hover)
        
        self.b_plan.on_clicked(self.calc_path)
        self.b_start.on_clicked(self.start)
        self.b_reset.on_clicked(self.reset)
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=30, blit=False)
        
    def get_body_corners(self, x, y, th):
        L, W = self.params.L, self.params.W
        c = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])
        return (rotation2d(th) @ c.T).T + np.array([x, y])

    def get_wheel_corners(self, x, y, th, idx):
        # idx: 0=FR, 1=FL, 2=RL, 3=RR
        L, W = self.params.L, self.params.W
        centers = [[L, -W], [L, W], [-L, W], [-L, -W]] # Body frame centers
        
        # Wheel dims
        wl, ww = 0.1, 0.04
        wc = np.array([[wl/2, ww/2], [wl/2, -ww/2], [-wl/2, -ww/2], [-wl/2, ww/2]])
        
        # Rotate wheel rect
        wc_rot = (rotation2d(th) @ wc.T).T
        
        # Translate to wheel center in world
        center_world = (rotation2d(th) @ np.array(centers[idx])) + np.array([x, y])
        
        return wc_rot + center_world

    def on_click(self, event):
        if event.inaxes != self.ax or self.sim_running: return
        self.goals.append((event.xdata, event.ydata))
        c = Circle((event.xdata, event.ydata), 0.3, color=self.c['goal'], zorder=10)
        t = self.ax.text(event.xdata, event.ydata, str(len(self.goals)), color='black', ha='center', va='center', weight='bold')
        self.ax.add_patch(c); self.goal_artists.extend([c, t])
        self.fig.canvas.draw()

    def calc_path(self, event):
        if not self.goals: return
        print("Planning...")
        full_path = []
        curr = self.q[:2]
        for g in self.goals:
            seg = self.planner.plan(curr[0], curr[1], g[0], g[1], self.dyn_obs)
            if seg is None: print("Blocked!"); return
            full_path.append(seg if not full_path else seg[1:])
            curr = g
        self.path = np.vstack(full_path)
        self.path_line.set_data(self.path[:,0], self.path[:,1])
        self.traj_func, _ = gen_traj(self.path, avg_vel=1.5)
        self.fig.canvas.draw()

    def start(self, event):
        if self.traj_func: self.sim_running = True; self.curr_goal = 0; self.t_curr = 0.0

    def get_current_friction(self, x, y):
        for z in self.zones:
            if z.contains(x, y): return z.mult
        return 1.0

    def update(self, frame):
        # 1. Update Roaming Obstacles
        for i, o in enumerate(self.dyn_obs):
            o.update(); self.dyn_patches[i].center = o.pos

        # 2. Sim Update
        if self.sim_running and self.traj_func:
            # Check for replan
            risk = False
            for o in self.dyn_obs:
                if np.linalg.norm(o.pos - self.q[:2]) < 1.5: risk = True
            
            if frame % 15 == 0 and risk:
                new_p = self.planner.plan(self.q[0], self.q[1], self.goals[self.curr_goal][0], self.goals[self.curr_goal][1], self.dyn_obs)
                if new_p is not None:
                    self.path = new_p; self.path_line.set_data(self.path[:,0], self.path[:,1])
                    self.traj_func, _ = gen_traj(self.path, avg_vel=1.5); self.t_curr = 0.0

            qd, dqd, ddqd = self.traj_func(self.t_curr)
            
            if np.linalg.norm(self.q[:2] - self.goals[self.curr_goal]) < 0.2:
                self.curr_goal += 1
                if self.curr_goal >= len(self.goals): self.sim_running = False; return [self.robot_body]
                self.t_curr = 0.0
                new_p = self.planner.plan(self.q[0], self.q[1], self.goals[self.curr_goal][0], self.goals[self.curr_goal][1], self.dyn_obs)
                if new_p is not None:
                    self.path = new_p; self.path_line.set_data(self.path[:,0], self.path[:,1])
                    self.traj_func, _ = gen_traj(self.path, avg_vel=1.5)

            F = smc_control(self.q, self.dq, qd, dqd, ddqd, self.params, self.M_nom, self.C_nom)
            f_mult = self.get_current_friction(self.q[0], self.q[1])
            _, C_real, _, _ = get_matrices(self.params, friction_mult=f_mult)
            
            dt = 0.03
            tau = self.B_pinv @ F
            F_prop = self.B @ tau
            ddq = np.linalg.solve(self.M_nom, F_prop - C_real @ self.dq)
            
            self.dq += ddq * dt
            self.q[:2] += (rotation2d(self.q[2]) @ self.dq[:2]) * dt
            self.q[2] += self.dq[2] * dt
            self.t_curr += dt
            
            self.robot_body.set_xy(self.get_body_corners(*self.q))
            for i in range(4): self.wheels[i].set_xy(self.get_wheel_corners(self.q[0], self.q[1], self.q[2], i))
            
            xd, yd = self.trace_line.get_data()
            self.trace_line.set_data(np.append(xd, self.q[0]), np.append(yd, self.q[1]))

        return [self.robot_body, self.path_line, self.trace_line] + self.dyn_patches + self.wheels

    def reset(self, event):
        if hasattr(self, 'ani') and self.ani: self.ani.event_source.stop()
        self.sim_running = False
        self.goals = []; [a.remove() for a in self.goal_artists]; self.goal_artists = []
        self.path_line.set_data([], []); self.trace_line.set_data([], [])
        self.q = np.zeros(3); self.dq = np.zeros(3)
        self.robot_body.set_xy(self.get_body_corners(0,0,0))
        for i in range(4): self.wheels[i].set_xy(self.get_wheel_corners(0,0,0,i))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=30, blit=False)
        self.fig.canvas.draw()

if __name__ == "__main__":
    app = ModernApp()
    plt.show()