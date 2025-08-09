# Redefine the function before plotting, as it was not retained in the environment.

import numpy as np
import matplotlib.pyplot as plt

def generate_positions(center, num_points=100, grid_size=(1220, 370)):
    """
    Generate random positions with probabilities inversely proportional to the distance from a given position.

    Parameters:
    - center: A tuple (x, y) representing the center position.
    - num_points: The number of positions to generate (default is 100).
    - grid_size: The size of the grid (width, height) where positions are generated (default is 100x100).

    Returns:
    - A numpy array of shape (num_points, 2) with the generated positions.
    """
    num_candidates = 10000
    candidates = np.random.rand(num_candidates, 2) * np.array(grid_size)

    distances = np.linalg.norm(candidates - np.array(center), axis=1)
    
    epsilon = 1e-6
    inverse_distances = 1 / (distances + epsilon)

    probabilities = inverse_distances / np.sum(inverse_distances)

    indices = np.random.choice(np.arange(num_candidates), size=num_points, p=probabilities, replace=False)
    sampled_positions = candidates[indices]

    return sampled_positions

def plot_generated_positions(center, generated_positions, grid_size=(1220, 370)):
    """
    Plot the generated positions on a 2D image.

    Parameters:
    - center: A tuple (x, y) representing the center position.
    - generated_positions: A numpy array of shape (num_points, 2) with the generated positions.
    - grid_size: The size of the grid (width, height) for visualization (default is 100x100).
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(generated_positions[:, 0], generated_positions[:, 1], c='blue', label='Generated Positions')
    plt.scatter(center[0], center[1], c='red', marker='x', s=100, label='Center Position')
    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])
    plt.title('Generated Positions with Inverse Distance Probability')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.savefig('./vis.png')

# Example usage
center_position = (1220//2, 370//2)  # Center of the grid
generated_positions = generate_positions(center_position)
generated_positions = np.array(generated_positions) / (1220, 370)
np.save('./vp_sample.npy', generated_positions)
# plot_generated_positions(center_position, generated_positions)
