import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from mplsoccer import Pitch
from matplotlib.path import Path
import seaborn as sns
from matplotlib.lines import Line2D
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


class GhostPlayer:
    """
    Simulates the movement of an occluded player ('Ghost') using a Stochastic
    Particle Filter governed by Langevin Dynamics.
    """

    def __init__(self, start_x, start_y, v_x, v_y):
        """
        Initialize 1000 particles at the last known location.
        """
        self.n_particles = 1000
        self.particles = np.zeros((self.n_particles, 4))
        self.particles[:, 0], self.particles[:, 1] = start_x, start_y
        self.particles[:, 2], self.particles[:, 3] = v_x, v_y

    def predict(self, iterations=25, dt=0.1):
        """
        Propagate particles forward in time using Anisotropic Diffusion.

        Physics Justification:
        We apply more noise longitudinally (0.7) than laterally (0.25) because
        players can accelerate/decelerate significantly faster than they can
        strafe sideways at high speeds.
        """
        for _ in range(iterations):
            # 1. Deterministic Update (Velocity)
            self.particles[:, 0] += self.particles[:, 2] * dt
            self.particles[:, 1] += self.particles[:, 3] * dt

            # 2. Apply Drag (Bio-mechanical friction)
            self.particles[:, 2] *= 0.95
            self.particles[:, 3] *= 0.95

            # 3. Anisotropic Stochastic Injection
            speed = np.linalg.norm(self.particles[:, 2:], axis=1) + 0.01
            ux, uy = self.particles[:, 2] / speed, self.particles[:, 3] / speed
            px, py = -uy, ux  # Perpendicular vector

            long_noise = np.random.normal(0, 0.7, self.n_particles)
            lat_noise = np.random.normal(0, 0.25, self.n_particles)

            self.particles[:, 0] += (long_noise * ux) + (lat_noise * px)
            self.particles[:, 1] += (long_noise * uy) + (lat_noise * py)

            # 4. Boundary Clamp (Pitch dimensions)
            self.particles[:, 0] = np.clip(self.particles[:, 0], 0, 105)
            self.particles[:, 1] = np.clip(self.particles[:, 1], 0, 68)

    def apply_camera_constraint(self, camera_poly_coords):
        """
        Schrödinger's Constraint: If a particle wanders INTO the camera view
        but the player is not detected, that particle is impossible. Kill it.
        """
        if not camera_poly_coords: return
        camera_path = Path(camera_poly_coords)
        inside_mask = camera_path.contains_points(self.particles[:, :2])
        self.particles[inside_mask, :] = np.nan


def get_player_velocity(df, player_id, current_frame_idx):
    """
    Calculates the velocity vector (vx, vy) of a player at the moment of occlusion.
    """
    try:
        curr = df.iloc[current_frame_idx]
        past = df.iloc[max(0, current_frame_idx - 5)]
        p_curr = next((p for p in curr['player_data'] if p['player_id'] == player_id), None)
        p_past = next((p for p in past['player_data'] if p['player_id'] == player_id), None)
        if p_curr and p_past:
            return (p_curr['x'] - p_past['x']) / 0.5, (p_curr['y'] - p_past['y']) / 0.5
    except:
        pass
    return 0, 0


def run_visualization():
    """
    Main execution function. Loads data, runs the particle filter, and generates
    the 'Stochastic Reconstruction' visualization.
    """
    print(f"👻 Running Ghost Model Visualization for Frame {TARGET_FRAME}...")
    df = load_match_data()

    target_row_idx = df.index[df['frame'] == TARGET_FRAME][0]
    target_frame = df.iloc[target_row_idx]

    cam_data = target_frame.get('image_corners_projection')
    camera_poly = []
    if isinstance(cam_data, dict):
        camera_poly = [[cam_data['x_top_left'], cam_data['y_top_left']],
                       [cam_data['x_bottom_left'], cam_data['y_bottom_left']],
                       [cam_data['x_bottom_right'], cam_data['y_bottom_right']],
                       [cam_data['x_top_right'], cam_data['y_top_right']]]

    # --- PLOTTING ---
    pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68,
                  pitch_color='#22312b', line_color='#c7d5cc', linewidth=1.5)
    fig, ax = pitch.draw(figsize=(16, 10))

    all_ghost_x, all_ghost_y = [], []
    eigen_ellipses = []

    for player in target_frame['player_data']:
        pid = player['player_id']
        x, y = player['x'], player['y']

        # Identify Ghosts (Not Detected)
        if not player.get('is_detected', True):
            vx, vy = get_player_velocity(df, pid, target_row_idx)
            ghost = GhostPlayer(x, y, vx, vy)
            ghost.predict()
            ghost.apply_camera_constraint(camera_poly)
            valid = ghost.particles[~np.isnan(ghost.particles).any(axis=1)]

            if len(valid) > 10:
                all_ghost_x.extend(valid[:, 0])
                all_ghost_y.extend(valid[:, 1])

                # Calculate Covariance for Eigen-Ellipses
                cov = np.cov(valid[:, 0], valid[:, 1])
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                eigen_ellipses.append({
                    'xy': np.mean(valid[:, :2], axis=0),
                    'w': 4 * np.sqrt(vals[0]),
                    'h': 4 * np.sqrt(vals[1]),
                    'angle': angle
                })

    # A. Plot Uncertainty Cloud (PDF)
    if len(all_ghost_x) > 0:
        sns.kdeplot(x=all_ghost_x, y=all_ghost_y, ax=ax, fill=True,
                    cmap='viridis', alpha=0.6, thresh=0.02, levels=15, zorder=2)

    # B. Plot 2-Sigma Ellipses
    for e in eigen_ellipses:
        ell = Ellipse(xy=e['xy'], width=e['w'], height=e['h'], angle=e['angle'],
                      facecolor='none', edgecolor='white', lw=1.5, linestyle='--', alpha=0.8, zorder=3)
        ax.add_patch(ell)

    # C. Plot Camera Field of View
    if camera_poly:
        ax.add_patch(Polygon(camera_poly, closed=True, facecolor='white', alpha=0.05, zorder=1))
        ax.add_patch(
            Polygon(camera_poly, closed=True, facecolor='none', edgecolor='white', lw=1, linestyle=':', zorder=2))

    # D. Plot Players
    for player in target_frame['player_data']:
        if player.get('is_detected', True):
            pitch.scatter(player['x'], player['y'], ax=ax, c='#00f2ff', s=80, edgecolors='none', zorder=5)
        else:
            pitch.scatter(player['x'], player['y'], ax=ax, c='#ff4444', marker='x', s=50, alpha=0.8, zorder=4)

    # E. Legend & Titles
    legend_elements = [
        Line2D([0], [0], color='white', lw=1.5, linestyle='--', label='2$\sigma$ Eigen-Ellipse (Covariance)'),
        Line2D([0], [0], marker='o', color='w', label='Observed Player', markerfacecolor='#00f2ff', markersize=8),
        plt.Rectangle((0, 0), 1, 1, facecolor='#2ecc71', alpha=0.6, label='Uncertainty PDF (Particle Cloud)'),
        Line2D([0], [0], marker='x', color='#ff4444', label='Deterministic Extrapolation', markersize=8,
               linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=10, frameon=False, labelcolor='white')

    plt.suptitle("Schrödinger's Pitch Control: Stochastic Reconstruction", fontsize=20, color='white',
                 fontweight='bold', y=0.99)
    plt.title(f"Uncertainty Quantification via Anisotropic Particle Filtering | Frame {TARGET_FRAME}", fontsize=13,
              color='#aaaaaa', y=1.01)
    plt.subplots_adjust(top=0.85, bottom=0.1)
    fig.set_facecolor('#22312b')
    plt.show()


if __name__ == "__main__":
    run_visualization()