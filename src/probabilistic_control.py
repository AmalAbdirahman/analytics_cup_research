import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os
import sys

# Ensure we can find data_loader.py
try:
    from data_loader import load_match_data
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_match_data

# --- CONFIG ---
TARGET_FRAME = 5059


# --- 1. PHYSICS ENGINE ---
class StochasticGhost:
    """
    Simplified Particle Filter for the Control Model.
    """

    def __init__(self, x, y, vx, vy):
        self.n = 1000
        self.particles = np.zeros((self.n, 4))
        self.particles[:, 0], self.particles[:, 1] = x, y
        self.particles[:, 2], self.particles[:, 3] = vx, vy

    def evolve(self, iterations=25, dt=0.1):
        for _ in range(iterations):
            self.particles[:, 0] += self.particles[:, 2] * dt
            self.particles[:, 1] += self.particles[:, 3] * dt
            self.particles[:, 2] *= 0.95
            self.particles[:, 3] *= 0.95

            # Anisotropic Drift
            speed = np.linalg.norm(self.particles[:, 2:], axis=1) + 0.01
            ux = self.particles[:, 2] / speed
            uy = self.particles[:, 3] / speed
            px, py = -uy, ux

            long_noise = np.random.normal(0, 0.7, self.n)
            lat_noise = np.random.normal(0, 0.25, self.n)

            self.particles[:, 0] += (long_noise * ux) + (lat_noise * px)
            self.particles[:, 1] += (long_noise * uy) + (lat_noise * py)

            # Clamp to SkillCorner Pitch Limits
            self.particles[:, 0] = np.clip(self.particles[:, 0], -52.5, 52.5)
            self.particles[:, 1] = np.clip(self.particles[:, 1], -34, 34)


def get_velocity(df, pid, idx):
    try:
        curr = df.iloc[idx]
        past = df.iloc[max(0, idx - 5)]
        p1 = next((p for p in curr['player_data'] if p['player_id'] == pid), None)
        p2 = next((p for p in past['player_data'] if p['player_id'] == pid), None)
        if p1 and p2: return (p1['x'] - p2['x']) / 0.5, (p1['y'] - p2['y']) / 0.5
    except:
        pass
    return 0, 0


# --- 2. PITCH CONTROL LOGIC ---
def gaussian_influence(px, py, grid_x, grid_y, sigma=3.0):
    return np.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2) / (2 * sigma ** 2))


def run_control_metric():
    """
    Main function to generate the 3-panel Control Surface comparison.
    """
    print(f"🔬 Computing Probabilistic Pitch Control for Frame {TARGET_FRAME}...")
    df = load_match_data()

    # Locate Frame
    try:
        row_idx = df.index[df['frame'] == TARGET_FRAME][0]
    except IndexError:
        print(f"❌ Frame {TARGET_FRAME} not found.")
        return

    row = df.iloc[row_idx]

    # --- COORDINATE FIX: USE -52.5 TO 52.5 (Standard SkillCorner) ---
    grid_x, grid_y = np.meshgrid(np.linspace(-52.5, 52.5, 50), np.linspace(-34, 34, 30))

    deterministic_surface = np.zeros_like(grid_x)
    probabilistic_surface = np.zeros_like(grid_x)

    # Store coordinates for markers
    detected_x, detected_y = [], []
    ghost_x, ghost_y = [], []

    for p in row['player_data']:
        pid = p['player_id']
        x, y = p['x'], p['y']
        vx, vy = get_velocity(df, pid, row_idx)

        # Baseline Influence
        inf = gaussian_influence(x, y, grid_x, grid_y)
        deterministic_surface = np.maximum(deterministic_surface, inf)

        # Ghost Logic
        if not p.get('is_detected', True):
            ghost_x.append(x)
            ghost_y.append(y)

            ghost = StochasticGhost(x, y, vx, vy)
            ghost.evolve()

            # Subsample particles for speed (use 20 particles)
            particle_inf = np.zeros_like(grid_x)
            for i in range(0, 1000, 50):
                px, py = ghost.particles[i, 0], ghost.particles[i, 1]
                particle_inf += gaussian_influence(px, py, grid_x, grid_y)

            particle_inf /= (1000 / 50)
            probabilistic_surface = np.maximum(probabilistic_surface, particle_inf)
        else:
            detected_x.append(x)
            detected_y.append(y)
            probabilistic_surface = np.maximum(probabilistic_surface, inf)

    delta_surface = probabilistic_surface - deterministic_surface

    # --- 3. METRIC QUANTIFICATION ---
    ghost_gain = np.sum(delta_surface[delta_surface > 0])

    # --- 4. VISUALIZATION ---
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), facecolor='#0e1117')

    # Use 'skillcorner' pitch type to align everything automatically
    pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68,
                  pitch_color='#0e1117', line_color='#444d56', linewidth=1)

    # Helper to draw markers
    def draw_markers(ax):
        pitch.scatter(detected_x, detected_y, ax=ax, c='#00f2ff', s=50, ec='none', alpha=0.8)
        pitch.scatter(ghost_x, ghost_y, ax=ax, c='#ff4444', marker='x', s=50, alpha=0.8)

    # Plot A: The Lie
    pitch.draw(ax=axs[0])
    # EXTENT must match the grid: -52.5 to 52.5
    axs[0].imshow(deterministic_surface, extent=(-52.5, 52.5, -34, 34), origin='lower', cmap='viridis', alpha=0.7,
                  vmin=0, vmax=1)
    draw_markers(axs[0])
    axs[0].set_title("A. Deterministic Control (Standard)", color='white', fontsize=14)

    # Plot B: The Truth
    pitch.draw(ax=axs[1])
    axs[1].imshow(probabilistic_surface, extent=(-52.5, 52.5, -34, 34), origin='lower', cmap='viridis', alpha=0.7,
                  vmin=0, vmax=1)
    draw_markers(axs[1])
    axs[1].set_title("B. Stochastic Control (Ours)", color='white', fontsize=14)

    # Plot C: The Difference
    pitch.draw(ax=axs[2])
    img = axs[2].imshow(delta_surface, extent=(-52.5, 52.5, -34, 34), origin='lower', cmap='coolwarm', vmin=-0.2,
                        vmax=0.2, alpha=0.9)

    axs[2].set_title(f"C. Ghost Gain: +{ghost_gain:.1f} Units", color='#ff4444', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(img, ax=axs[2], fraction=0.046, pad=0.04)
    cbar.set_label('Control Delta (Red = Hidden Threat)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.suptitle(f"Quantifying the 'Ghost Effect': Frame {TARGET_FRAME}", color='white', fontsize=20, y=0.98)
    plt.show()


if __name__ == "__main__":
    run_control_metric()