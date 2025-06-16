import numpy as np
import matplotlib.pyplot as plt

# === SETTINGS ===
title_text = "Version 1.0.56 (Labeled Peaks, Base 1.0)"
num_points = 2000            # Number of points per circle
num_circles = 10             # Number of concentric circles
base_radius = 1.0            # Base radius for all circles
power = 7                    # Steepness of each peak
num_peaks = 8                # Number of total peaks
min_peak_size = 80           # Minimum peak width (in points)
max_peak_size = 160          # Maximum peak width (in points)
min_peak_factor = 1.05       # Minimum peak height (1.05x circle)
max_peak_factor = 2.0        # Maximum peak height (2.0x circle)

theta = np.linspace(0, 2 * np.pi, num_points)

# --- Function to generate noisy base for each circle ---
def independent_variance(base_radius, num_points, min_var=0.01, max_var=0.10):
    # Adds a small phase-shifted, variable random "wobble" to the base circle
    phase = np.random.uniform(0, 2 * np.pi)
    maxv = np.random.uniform(min_var, 0.05) if np.random.rand() < 0.75 else np.random.uniform(0.05, max_var)
    climb = np.linspace(min_var, maxv, num_points // 2)
    recede = np.linspace(maxv, min_var, num_points - num_points // 2)
    vpat = np.concatenate((climb, recede))
    vpat = np.roll(vpat, int(phase * num_points / (2 * np.pi)))
    # Random, symmetrical deviation about the circle
    return base_radius * (1 + vpat * (2 * np.random.rand(num_points) - 1))

# --- Generate the noisy circles ---
radii = [independent_variance(base_radius, num_points) for _ in range(num_circles)]

# --- Create and store all peaks ---
peak_indices = [[] for _ in range(num_circles)]
peaks = []          # To store all peak parameters
peak_labels = []    # To store tip position and label for each peak

for _ in range(num_peaks):
    circle = np.random.randint(0, num_circles)
    peak_size = np.random.randint(min_peak_size, max_peak_size)
    start = np.random.randint(0, num_points - peak_size)
    end = start + peak_size
    peak_factor = np.random.uniform(min_peak_factor, max_peak_factor)
    peaks.append({'circle': circle, 'start': start, 'end': end, 'peak_factor': peak_factor, 'peak_size': peak_size})

# Print the parameters of each peak for reference
print("Peak array used for plotting:")
for idx, peak in enumerate(peaks):
    print(f"Peak {idx+1}: {peak}")

# --- Apply each peak's ramp to the correct region of its circle ---
for idx, p in enumerate(peaks):
    c = p['circle']
    start = p['start']
    end = p['end']
    peak_factor = p['peak_factor']
    peak_size = p['peak_size']
    half = (peak_size + 1) // 2
    up = 1 + (peak_factor - 1) * (np.linspace(0, 1, half) ** power)
    down = 1 + (peak_factor - 1) * (np.linspace(1, 0, peak_size // 2) ** power)
    ramp = np.concatenate([up, down])
    radii[c][start:end] *= ramp
    peak_indices[c].append((start, end))
    # Find and store label position (at tip of peak)
    peak_r = radii[c][start:end]
    peak_theta = theta[start:end]
    tip_idx = np.argmax(peak_r)
    x_tip = peak_r[tip_idx] * np.cos(peak_theta[tip_idx])
    y_tip = peak_r[tip_idx] * np.sin(peak_theta[tip_idx])
    peak_labels.append((x_tip, y_tip, f"peak {idx+1}"))

# --- Convert to x/y for plotting, split into base and peaks ---
circle_coordinates = []
for i in range(num_circles):
    x_peak, y_peak, x_base, y_base = [], [], [], []
    for j in range(num_points):
        if any(start <= j < end for start, end in peak_indices[i]):
            x_peak.append(radii[i][j] * np.cos(theta[j]))
            y_peak.append(radii[i][j] * np.sin(theta[j]))
        elif np.random.rand() < 0.1:  # Plot 10% of base dots for visual clarity
            x_base.append(radii[i][j] * np.cos(theta[j]))
            y_base.append(radii[i][j] * np.sin(theta[j]))
    circle_coordinates.append((x_peak, y_peak, x_base, y_base))

# --- Plot everything ---
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
distinct_gray_colors_10 = [plt.cm.Greys(i / 10) for i in range(num_circles)]
for i, (x_peak, y_peak, x_base, y_base) in enumerate(circle_coordinates):
    ax.scatter(x_base, y_base, color=distinct_gray_colors_10[i], s=1)
    ax.scatter(x_peak, y_peak, color='darkgray', s=1)

# Add labels at each peak tip
for x, y, label in peak_labels:
    ax.text(x, y, label, fontsize=9, color='black', ha='center', va='center')

ax.axis('off')
ax.set_aspect('equal', adjustable='box')
fig.text(0.01, 0.99, title_text, fontsize=12, ha='left', va='top', color='black')
plt.show()

