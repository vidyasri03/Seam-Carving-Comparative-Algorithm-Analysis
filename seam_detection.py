import numpy as np

def detect_seams(energy_map):
    if energy_map is None or energy_map.size == 0:
        raise ValueError("The energy map is empty. Please check the image or energy map generation.")

    rows, cols = energy_map.shape
    dp = np.zeros_like(energy_map, dtype=float)  # DP table to store cumulative energy
    seam = np.zeros((rows,), dtype=int)  # Array to store the seam path for columns

    # First row of dp is just the energy map
    dp[0] = energy_map[0]

    # Compute the dp table with cumulative energy
    for i in range(1, rows):
        for j in range(cols):
            left = dp[i - 1, j - 1] if j - 1 >= 0 else float('inf')
            up = dp[i - 1, j]
            right = dp[i - 1, j + 1] if j + 1 < cols else float('inf')

            dp[i, j] = energy_map[i, j] + min(left, up, right)

    # Backtrack to find the seam
    seam[rows - 1] = np.argmin(dp[rows - 1])  # Start with the minimum energy in the last row
    for i in range(rows - 2, -1, -1):
        x = seam[i + 1]
        start = max(x - 1, 0)
        end = min(x + 2, cols)

        seam[i] = start + np.argmin(dp[i, start:end])

    # Return the seam as a list of (row, col) tuples
    seam_coords = [(i, seam[i]) for i in range(rows)]
    return seam_coords
