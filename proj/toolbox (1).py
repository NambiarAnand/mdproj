import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

initial_params = {
    'm': 10.0, 'L2': 2.6, 'L3': 15.2, 'L4': 11.2, 'L5': 17.5,
    'L6': 9, 'L7': 9.2, 'L8': 10.2, 'L9': 8.8, 'L10': 9.0,
    'L11': 12.5, 'L12': 16, 'alpha': 0.0, 'OF': 7.8
}


def solve_pos(L_a, L_b, d, p1, p2, flip=False):
    if d > (L_a + L_b) or d < abs(L_a - L_b) or d == 0:
        raise ValueError("Singularity")
    a = (L_a ** 2 - L_b ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(0, L_a ** 2 - a ** 2))
    p3_base = p1 + a * (p2 - p1) / d
    offset = h * np.array([-(p2[1] - p1[1]), (p2[0] - p1[0])]) / d
    return p3_base - offset if flip else p3_base + offset


def unit_vec(ri, rj):
    diff = rj - ri
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros(2)
    return diff / norm


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def compute_forces(joints, torque, r_crank):
    G1 = joints['G1']
    P2 = joints['P2']
    P4 = joints['P4']
    P5 = joints['P5']
    P6 = joints['P6']
    F  = joints['F']

    F12 = torque / r_crank if abs(r_crank) > 1e-12 else 0.0

    u24 = unit_vec(P2, P4)
    u2F = unit_vec(P2, F)
    u46 = unit_vec(P4, P6)
    u4G = unit_vec(P4, G1)
    u15 = unit_vec(G1, P5)

    denom_F3 = cross2d(u2F, u24)
    F3 = -F12 * cross2d(u24, u46) / denom_F3 if abs(denom_F3) > 1e-12 else 0.0

    denom_F6 = cross2d(u46, u4G)
    F6 = -F12 * cross2d(u24, u4G) / denom_F6 if abs(denom_F6) > 1e-12 else 0.0

    R_top = -(F12 * u24 + F6 * u46)
    F9 = -np.dot(R_top, u15)

    R_ground = -(F6 * u46 + F3 * u2F)

    return F12, F3, F6, F9, np.linalg.norm(R_top), np.linalg.norm(R_ground)


class JansenSimulation:
    def __init__(self, params):
        self.params = params
        self.is_paused = False
        self.crank_angle_input_fn = lambda base_angle_deg, step_idx, total_steps: base_angle_deg
        self.torque = 1.0

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(left=0.3, bottom=0.15)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-30, 15)
        self.ax.grid(True, linestyle=':')

        self.line, = self.ax.plot([], [], 'ro-', lw=2, ms=4, label='Linkage')
        self.trace_line, = self.ax.plot([], [], 'k-', lw=1, alpha=0.4, label='Toe Path')
        self.toe_x, self.toe_y = [], []
        self.joint_order = ['G1', 'G2', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'F']
        self.joint_texts = {
            name: self.ax.text(0, 0, name, fontsize=8, color='tab:blue', visible=False)
            for name in self.joint_order
        }

        self.text_boxes = {}
        for i, (key, val) in enumerate(self.params.items()):
            ax_box = plt.axes([0.1, 0.85 - (i * 0.05), 0.1, 0.03])
            box = TextBox(ax_box, f'{key}: ', initial=str(val))
            box.on_submit(lambda text, k=key: self.update_param(k, text))
            self.text_boxes[key] = box

        ax_pause = plt.axes([0.45, 0.02, 0.1, 0.05])
        self.btn_pause = Button(ax_pause, 'Play/Pause')
        self.btn_pause.on_clicked(self.toggle_pause)

    def update_param(self, key, text):
        try:
            self.params[key] = float(text)
            self.toe_x, self.toe_y = [], []
        except ValueError:
            print(f"Invalid input for {key}")

    def toggle_pause(self, event):
        self.is_paused = not self.is_paused

    def set_crank_angle_input(self, crank_angle_input_fn):
        if not callable(crank_angle_input_fn):
            raise ValueError("crank_angle_input_fn must be callable")
        self.crank_angle_input_fn = crank_angle_input_fn

    def compute_joint_positions(self, crank_angle_deg):
        theta2 = np.radians(crank_angle_deg)
        p = self.params
        G1 = np.array([0.0, 0.0])
        G2 = np.array([
            p['m'] * np.cos(np.radians(p['alpha'])),
            p['m'] * np.sin(np.radians(p['alpha']))
        ])
        P2 = G2 + np.array([p['L2'] * np.cos(theta2), p['L2'] * np.sin(theta2)])
        P3 = solve_pos(p['L12'], p['L11'], np.linalg.norm(G1 - P2), P2, G1, flip=True)
        P4 = solve_pos(p['L10'], p['L8'], np.linalg.norm(G1 - P3), P3, G1, flip=True)
        F  = solve_pos(p['L3'], p['OF'], np.linalg.norm(G1 - P2), P2, G1, flip=False)
        P5 = G1 + (F - G1) * (p['L9'] / p['OF'])
        P6 = solve_pos(p['L6'], p['L7'], np.linalg.norm(P4 - P5), P4, P5, flip=True)
        P7 = solve_pos(p['L4'], p['L5'], np.linalg.norm(P5 - P6), P5, P6, flip=False)
        return {'G1': G1, 'G2': G2, 'P2': P2, 'P3': P3, 'P4': P4,
                'P5': P5, 'P6': P6, 'P7': P7, 'F': F}

    def sample_joint_motion(self, start_deg=0, end_deg=720, step_deg=2, angles=None):
        if angles is None:
            angles = self.generate_crank_angles(start_deg=start_deg, end_deg=end_deg, step_deg=step_deg)
        else:
            angles = np.asarray(angles, dtype=float)
        joint_names = ['G1', 'G2', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'F']
        joint_motion = {name: np.full((len(angles), 2), np.nan) for name in joint_names}
        for i, angle in enumerate(angles):
            try:
                joints = self.compute_joint_positions(angle)
                for name in joint_names:
                    joint_motion[name][i] = joints[name]
            except ValueError:
                continue
        return angles, joint_motion

    def plot_joint_motion(self, start_deg=0, end_deg=720, step_deg=2, angles=None):
        angles, joint_motion = self.sample_joint_motion(start_deg, end_deg, step_deg, angles=angles)
        joint_names = list(joint_motion.keys())
        rows = int(np.ceil(len(joint_names) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows), sharex=True)
        axes = np.array(axes).reshape(-1)
        for idx, name in enumerate(joint_names):
            ax = axes[idx]
            coords = joint_motion[name]
            ax.plot(angles, coords[:, 0], label='x', lw=1.8)
            ax.plot(angles, coords[:, 1], label='y', lw=1.8)
            ax.axvline(360, color='tab:red', linestyle='--', lw=1.5, label='2nd cycle start')
            ax.set_title(f'{name} vs crank angle')
            ax.set_ylabel('Position')
            ax.grid(True, linestyle=':', alpha=0.7)
        for idx in range(len(joint_names), len(axes)):
            axes[idx].axis('off')
        for ax in axes:
            if ax.has_data():
                ax.set_xlim(np.nanmin(angles), np.nanmax(angles))
                ax.set_xlabel('Crank angle (deg)')
        fig.suptitle('Joint Motion Over Two Crank Cycles (0 to 720 deg)', fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        return fig, axes

    def plot_position_vs_time(self, joint_name='P7', angular_speed_deg_per_sec=360.0,
                              start_deg=0, end_deg=720, step_deg=2, angles=None):
        if joint_name not in ['G1', 'G2', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'F']:
            raise ValueError('Unknown joint_name')
        if angles is None:
            angles = self.generate_crank_angles(start_deg=start_deg, end_deg=end_deg, step_deg=step_deg)
        else:
            angles = np.asarray(angles, dtype=float)
        time = angles / float(angular_speed_deg_per_sec)
        _, joint_motion = self.sample_joint_motion(angles=angles)
        coords = joint_motion[joint_name]
        fig, (axx, axy) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axx.plot(time, coords[:, 0], '-b', label=f'{joint_name}.x')
        axy.plot(time, coords[:, 1], '-g', label=f'{joint_name}.y')
        idx_div = int(np.nanargmin(np.abs(angles - 360))) if np.any(~np.isnan(angles)) else 0
        t_div = time[idx_div]
        axx.axvline(t_div, color='r', linestyle='--', lw=1.5)
        axy.axvline(t_div, color='r', linestyle='--', lw=1.5, label='2nd cycle start')
        axx.set_ylabel('X position')
        axy.set_ylabel('Y position')
        axy.set_xlabel('Time (s)')
        axx.grid(True, linestyle=':')
        axy.grid(True, linestyle=':')
        axx.legend(loc='best', fontsize=8)
        axy.legend(loc='best', fontsize=8)
        fig.suptitle(f'{joint_name} position vs time')
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        return fig, (axx, axy)

    def plot_toe_trajectory(self, start_deg=0, end_deg=720, step_deg=2, angles=None):
        if angles is None:
            angles = self.generate_crank_angles(start_deg=start_deg, end_deg=end_deg, step_deg=step_deg)
        else:
            angles = np.asarray(angles, dtype=float)

        _, joint_motion = self.sample_joint_motion(angles=angles)
        toe_coords = joint_motion['P7']

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(toe_coords[:, 0], toe_coords[:, 1], 'k-', lw=1.8, label='Toe trajectory')
        ax.scatter(toe_coords[0, 0], toe_coords[0, 1], color='tab:green', s=45, label='Start')
        ax.scatter(toe_coords[-1, 0], toe_coords[-1, 1], color='tab:red', s=45, label='End')
        ax.set_aspect('equal')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Final End Effector / Toe Trajectory')
        ax.grid(True, linestyle=':')
        ax.legend(loc='best')
        fig.tight_layout()
        return fig, ax

    def update(self, frame):
        if self.is_paused:
            return (self.line, self.trace_line, *self.joint_texts.values())

        try:
            theta = self.crank_angle_input_fn(frame, 0, 1)
            self.current_theta = theta
            joints = self.compute_joint_positions(self.current_theta)
            G1 = joints['G1']
            P2 = joints['P2']
            P3 = joints['P3']
            P4 = joints['P4']
            P5 = joints['P5']
            P6 = joints['P6']
            P7 = joints['P7']
            F  = joints['F']

            for name, coord in joints.items():
                self.joint_texts[name].set_position((coord[0] + 0.35, coord[1] + 0.35))
                self.joint_texts[name].set_visible(True)

            pts = [joints['G2'], P2, P3, G1, P4, P3, P2, F, G1, F, P5, P6, P4, P6, P7, P5]
            self.line.set_data([pt[0] for pt in pts], [pt[1] for pt in pts])

            self.toe_x.append(P7[0])
            self.toe_y.append(P7[1])
            if len(self.toe_x) > 120:
                self.toe_x.pop(0)
                self.toe_y.pop(0)
            self.trace_line.set_data(self.toe_x, self.toe_y)

        except ValueError:
            self.line.set_data([], [])
            for txt in self.joint_texts.values():
                txt.set_visible(False)

        return (self.line, self.trace_line, *self.joint_texts.values())

    def plot_forces_cycle(self, angles):
        F12_list, F3_list, F6_list, F9_list = [], [], [], []
        Rtop_list, Rg_list = [], []

        for theta in angles:
            try:
                theta_mod = self.crank_angle_input_fn(theta, 0, 1)
                joints = self.compute_joint_positions(theta_mod)

                theta_mod = theta % 360

                if 100 <= theta_mod <= 160:
                    torque = 0.5
                else:
                    torque = 2.0

                F12, F3, F6, F9, Rtop, Rg= compute_forces(joints, torque, self.params['L2'])

                F12_list.append(abs(F12))
                F3_list.append(abs(F3))
                F6_list.append(abs(F6))
                F9_list.append(abs(F9))
                Rtop_list.append(abs(Rtop))
                Rg_list.append(abs(Rg))
            except:
                continue

        angles = angles[:len(F12_list)]

        force_series = [
            (F12_list, 'F12'),
            (F3_list, 'F3'),
            (F6_list, 'F6'),
            (F9_list, 'F9'),
            (Rtop_list, 'Top Reaction'),
            (Rg_list, 'Ground Reaction'),
        ]

        for values, label in force_series:
            plt.figure(figsize=(10, 6))
            plt.plot(angles, values, label=label)
            plt.xlabel("Crank Angle")
            plt.ylabel("Force")
            plt.title(label)
            plt.legend()
            plt.grid(True)

        plt.show()


def crank_angle_input(base_angle_deg, step_idx, total_steps):
    if not hasattr(crank_angle_input, "angle"):
        crank_angle_input.angle = 0.0

    step = 2.0

    theta_mod = crank_angle_input.angle % 360

    center = 130.0
    width = 25.0

    T_low = 0.5
    T_high = 2.0

    s = 0.5 * (1 + np.tanh((theta_mod - center) / width))
    torque = T_high * (1 - s) + T_low * s

    omega_scale = 1.0 / torque

    crank_angle_input.angle += step * omega_scale

    return crank_angle_input.angle

sim = JansenSimulation(initial_params)
sim.set_crank_angle_input(crank_angle_input)
animation_angles = np.arange(0, 720, 2)
ani = FuncAnimation(sim.fig, sim.update, frames=animation_angles, interval=20, blit=False)
plt.show()
sim.plot_forces_cycle(animation_angles)
motion_angles = np.arange(0, 720, 2)
sim.plot_joint_motion(angles=motion_angles)
plt.show()
sim.plot_position_vs_time(joint_name='P7', angular_speed_deg_per_sec=360.0, angles=motion_angles)
plt.show()
sim.plot_toe_trajectory(angles=motion_angles)
plt.show()