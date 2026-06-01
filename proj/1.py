import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Link lengths (from your image)
L = {
    'a': 38.0, 'b': 41.5, 'c': 39.3, 'd': 40.1,
    'e': 55.8, 'f': 39.4, 'g': 36.7, 'h': 65.7,
    'i': 49.0, 'j': 50.0, 'k': 61.9, 'l': 7.8, 'm': 15.0
}

# Fixed points
O = np.array([0.0, 0.0])  # crank center
P = np.array([L['a'], 0.0])  # offset joint

def circle_intersection(c1, r1, c2, r2):
    d = np.linalg.norm(c2 - c1)
    if d > r1 + r2:
        return None
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = np.sqrt(max(r1**2 - a**2, 0))
    mid = c1 + a * (c2 - c1) / d
    offset = h * np.array([-(c2[1]-c1[1]), c2[0]-c1[0]]) / d
    return mid + offset  # pick one branch

def compute(theta):
    # crank point
    A = O + L['m'] * np.array([np.cos(theta), np.sin(theta)])

    # solve linkage step by step
    B = circle_intersection(P, L['j'], A, L['l'])
    if B is None: return None

    C = circle_intersection(A, L['b'], B, L['c'])
    if C is None: return None

    D = circle_intersection(C, L['d'], B, L['e'])
    if D is None: return None

    E = circle_intersection(C, L['g'], P, L['i'])
    if E is None: return None

    F = circle_intersection(E, L['h'], C, L['f'])
    if F is None: return None

    return {
        'O': O, 'P': P, 'A': A, 'B': B,
        'C': C, 'D': D, 'E': E, 'F': F
    }

def draw(ax, pts):
    ax.clear()
    
    if pts is None:
        return

    # Draw links
    def link(p, q):
        ax.plot([p[0], q[0]], [p[1], q[1]], 'k-')

    link(pts['O'], pts['A'])
    link(pts['A'], pts['B'])
    link(pts['B'], pts['P'])
    link(pts['A'], pts['C'])
    link(pts['C'], pts['B'])
    link(pts['C'], pts['D'])
    link(pts['D'], pts['B'])
    link(pts['C'], pts['E'])
    link(pts['E'], pts['P'])
    link(pts['E'], pts['F'])
    link(pts['F'], pts['C'])

    # joints
    for p in pts.values():
        ax.plot(p[0], p[1], 'ro')

    ax.set_aspect('equal')
    ax.set_xlim(-100, 150)
    ax.set_ylim(-150, 100)
    ax.grid()

# Setup plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

theta0 = 0.0
pts = compute(theta0)
draw(ax, pts)

# Slider
ax_theta = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_theta, 'Theta', 0, 2*np.pi, valinit=theta0)

def update(val):
    theta = slider.val
    pts = compute(theta)
    draw(ax, pts)
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()