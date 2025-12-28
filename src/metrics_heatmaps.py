import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os
import scipy.ndimage
import sys

# Ensure we can find data_loader.py
try:
    from data_loader import load_match_data
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_match_data


# --- HELPER: ROBUST VELOCITY ---
def calculate_speeds(df):
    """
    Vectorized velocity calculation for the entire match.
    """
    sprints = []

    # Iterate through frames
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        dt = 0.1  # 10 FPS assumption

        curr_players = {p['player_id']: p for p in curr_row['player_data']}
        prev_players = {p['player_id']: p for p in prev_row['player_data']}

        for pid, p_curr in curr_players.items():
            if pid in prev_players:
                p_prev = prev_players[pid]

                # Calculate Speed
                vx = (p_curr['x'] - p_prev['x']) / dt
                vy = (p_curr['y'] - p_prev['y']) / dt
                speed = np.sqrt(vx ** 2 + vy ** 2)

                # Filter implausible tracking glitches and walking
                if 4.0 < speed < 13.0:
                    sprints.append({
                        'x': p_curr['x'],
                        'y': p_curr['y'],
                        'speed': speed,
                        # If 'is_detected' is missing, assume True. We want Ghosts (False).
                        'is_ghost': not p_curr.get('is_detected', True)
                    })
    return sprints


def run_audit():
    """
    Generates the Blindside Risk Heatmap.
    """
    match_id = '1886347'
    print(f"📊 Auditing Broadcast Reliability for {match_id}...")

    # Load Data
    df = load_match_data()

    # Calc Speeds
    sprint_data = calculate_speeds(df)
    print(f"   Analyzed {len(sprint_data)} high-speed events (>4m/s).")

    # Separate Lists
    total_x = [d['x'] for d in sprint_data]
    total_y = [d['y'] for d in sprint_data]

    ghost_x = [d['x'] for d in sprint_data if d['is_ghost']]
    ghost_y = [d['y'] for d in sprint_data if d['is_ghost']]

    if not total_x:
        print("❌ No high-speed sprints found.")
        return

    # --- VISUALIZATION ---
    # Use 'skillcorner' pitch type which handles the -52.5 to 52.5 coordinates automatically for drawing
    pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68,
                  pitch_color='#1a1a1a', line_color='#444d56', linewidth=1.5)
    fig, ax = pitch.draw(figsize=(14, 9))

    # 1. Compute 2D Histograms
    bins = (50, 32)

    # --- CRITICAL FIX: MATCH SKILLCORNER COORDINATES ---
    # SkillCorner uses centered coordinates: X [-52.5, 52.5], Y [-34, 34]
    pitch_range = [[-52.5, 52.5], [-34, 34]]

    hist_total, x_edge, y_edge = np.histogram2d(total_x, total_y, bins=bins, range=pitch_range)
    hist_ghost, _, _ = np.histogram2d(ghost_x, ghost_y, bins=bins, range=pitch_range)

    # 2. Compute The Ratio (Risk %)
    with np.errstate(divide='ignore', invalid='ignore'):
        risk_map = hist_ghost / hist_total
        risk_map = np.nan_to_num(risk_map)

    # 3. Smooth
    risk_map_smooth = scipy.ndimage.gaussian_filter(risk_map, sigma=1.5)

    # 4. Plot
    # --- CRITICAL FIX: MATCH EXTENT TO COORDINATES ---
    img = ax.imshow(
        risk_map_smooth.T,
        extent=(-52.5, 52.5, -34, 34),
        origin='lower',
        cmap='inferno',
        vmin=0, vmax=0.8,
        alpha=0.9
    )

    # 5. Labels & Polish
    cbar = plt.colorbar(img, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Blindside Risk % (Prob. of Unseen Sprint)', color='white', fontsize=12, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#444d56')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Calculate Global Stat
    avg_risk = np.mean(risk_map[hist_total > 5]) * 100 if np.any(hist_total > 5) else 0

    plt.suptitle("The Broadcast Reliability Audit", fontsize=22, color='white', fontweight='bold', y=0.98)
    plt.title(
        f"Match {match_id} | Color = Probability that a Sprint (>4m/s) is Invisible | Global Avg Risk: {avg_risk:.1f}%",
        fontsize=14, color='#ff4444', fontweight='bold', y=1.01)

    fig.set_facecolor('#1a1a1a')
    plt.subplots_adjust(top=0.85)

    plt.show()


if __name__ == "__main__":
    run_audit()