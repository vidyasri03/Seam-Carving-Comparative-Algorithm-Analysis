import cv2
import numpy as np
import heapq  # For implementing Dijkstra's algorithm


# Energy Map Calculation
def calculate_energy_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.sqrt(sobel_x**2 + sobel_y**2)
    return energy_map


# Seam Removal Logic
def remove_vertical_seam(image, seam):
    rows, cols, _ = image.shape
    output = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
    for i in range(rows):
        col = seam[i]
        output[i, :, :] = np.delete(image[i, :, :], col, axis=0)
    return output


# Greedy Seam Carving
def seam_carving_greedy(image, num_seams):
    for _ in range(num_seams):
        energy_map = calculate_energy_map(image)
        seam = np.argmin(np.sum(energy_map, axis=0))
        seam = np.array([seam] * image.shape[0])
        image = remove_vertical_seam(image, seam)
    return image


# Recursive Seam Carving
def calculate_recursive_seam(energy_map, row, col, memo):
    if row == len(energy_map):
        return 0
    if (row, col) in memo:
        return memo[(row, col)]

    min_energy = energy_map[row, col] + min(
        calculate_recursive_seam(energy_map, row + 1, col, memo) if col >= 0 else float('inf'),
        calculate_recursive_seam(energy_map, row + 1, col - 1, memo) if col > 0 else float('inf'),
        calculate_recursive_seam(energy_map, row + 1, col + 1, memo) if col < energy_map.shape[1] - 1 else float('inf')
    )
    memo[(row, col)] = min_energy
    return min_energy


def seam_carving_recursive(image, num_seams):
    for _ in range(num_seams):
        energy_map = calculate_energy_map(image)
        rows, cols = energy_map.shape
        memo = {}
        seam = np.zeros(rows, dtype=np.int32)
        min_col = np.argmin([calculate_recursive_seam(energy_map, 0, col, memo) for col in range(cols)])
        seam[0] = min_col

        for row in range(1, rows):
            col = seam[row - 1]
            offsets = [-1, 0, 1]
            min_offset = min(offsets, key=lambda d: memo.get((row, col + d), float('inf')))
            seam[row] = col + min_offset if 0 <= col + min_offset < cols else col

        image = remove_vertical_seam(image, seam)
    return image


# Dynamic Programming Seam Carving
def seam_carving_dynamic(image, num_seams):
    for _ in range(num_seams):
        energy_map = calculate_energy_map(image)
        rows, cols = energy_map.shape
        seam = np.zeros(rows, dtype=np.int32)
        cumulative_map = energy_map.copy()

        for i in range(1, rows):
            for j in range(cols):
                min_val = cumulative_map[i-1, j]
                if j > 0:
                    min_val = min(min_val, cumulative_map[i-1, j-1])
                if j < cols - 1:
                    min_val = min(min_val, cumulative_map[i-1, j+1])
                cumulative_map[i, j] += min_val

        seam[-1] = np.argmin(cumulative_map[-1])
        for i in range(rows - 2, -1, -1):
            prev_col = seam[i + 1]
            offset = np.argmin(cumulative_map[i, max(prev_col - 1, 0):min(prev_col + 2, cols)])
            seam[i] = prev_col + offset - 1 if prev_col > 0 else offset

        image = remove_vertical_seam(image, seam)
    return image


# Graph-Based (Dijkstra) Seam Carving
'''def seam_carving_graph(image, num_seams):
    for _ in range(num_seams):
        energy_map = calculate_energy_map(image)
        rows, cols = energy_map.shape
        cumulative_energy = np.copy(energy_map)
        parent = np.zeros_like(cumulative_energy, dtype=np.int32)
        pq = []

        for i in range(rows):
            heapq.heappush(pq, (cumulative_energy[i, 0], i, 0))

        while pq:
            energy, row, col = heapq.heappop(pq)
            if col == cols - 1:
                continue
            for d in [-1, 0, 1]:
                next_col = col + 1
                next_row = row + d
                if 0 <= next_row < rows:
                    next_energy = cumulative_energy[next_row, next_col]
                    if cumulative_energy[row, col] + energy_map[next_row, next_col] < next_energy:
                        cumulative_energy[next_row, next_col] = cumulative_energy[row, col] + energy_map[next_row, next_col]
                        parent[next_row, next_col] = row
                        heapq.heappush(pq, (cumulative_energy[next_row, next_col], next_row, next_col))

        seam = np.zeros(rows, dtype=int)
        seam[-1] = np.argmin(cumulative_energy[-1])
        for i in range(rows - 2, -1, -1):
            seam[i] = parent[seam[i + 1], i + 1]

        image = remove_vertical_seam(image, seam)
    return image'''

import numpy as np
import heapq

def seam_carving_graph(image, num_seams):
    """
    Perform seam carving using a graph-based approach (Dijkstra's algorithm) to remove vertical seams.
    
    Parameters:
        image (numpy.ndarray): Input image (2D or 3D array).
        num_seams (int): Number of vertical seams to remove.
    
    Returns:
        numpy.ndarray: Image with specified seams removed.
    """
    for _ in range(num_seams):
        # Calculate the energy map for the current image
        energy_map = calculate_energy_map(image)
        rows, cols = energy_map.shape

        # Initialize cumulative energy and parent arrays
        cumulative_energy = np.full((rows, cols), float('inf'))
        parent = np.full((rows, cols), -1, dtype=int)

        # Start from the first column of the energy map
        for i in range(rows):
            cumulative_energy[i, 0] = energy_map[i, 0]

        # Priority queue for Dijkstra's algorithm
        pq = []
        for i in range(rows):
            heapq.heappush(pq, (cumulative_energy[i, 0], i, 0))

        # Perform Dijkstra's algorithm
        while pq:
            energy, row, col = heapq.heappop(pq)

            if col == cols - 1:
                continue  # Skip processing if at the last column

            for d in [-1, 0, 1]:  # Possible row offsets for neighbors
                next_row = row + d
                next_col = col + 1
                if 0 <= next_row < rows and next_col < cols:
                    new_energy = energy + energy_map[next_row, next_col]
                    if new_energy < cumulative_energy[next_row, next_col]:
                        cumulative_energy[next_row, next_col] = new_energy
                        parent[next_row, next_col] = row
                        heapq.heappush(pq, (new_energy, next_row, next_col))

        # Trace back the seam from the last column
        seam = np.zeros(cols, dtype=int)
        seam[-1] = np.argmin(cumulative_energy[:, -1])  # Start from the minimum energy at the last column
        for col in range(cols - 2, -1, -1):
            seam[col] = parent[seam[col + 1], col + 1]

        # Remove the seam from the image
        image = remove_vertical_seam(image, seam)

    return image
