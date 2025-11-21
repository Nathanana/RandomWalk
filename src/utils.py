from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from numpy.fft import fft, ifft

def generate_srw(num_steps: int, start_position: Union[int, list, np.ndarray] = 0, dim: int = 1, seed: int = None) -> np.ndarray:
    """
    Generate a simple random walk

    Parameters:
    - num_steps: Number of steps in the random walk
    - start_position: Starting position of the walk (Can be an integer for 1D, or a list/np.array for D > 1)
    - dim: Number of dimensions of the walk
    - seed: Random seed for reproducibility

    Returns:
    - positions: An array of positions at each step of the walk. The shape will be (num_steps + 1, dim)
    """
    
    if seed is not None:
        np.random.seed(seed)

    # Process the start_position to ensure it is an array of shape (1, dim)
    start_pos_arr = np.asarray(start_position)

    if start_pos_arr.ndim == 0:
        start_pos_arr = np.full(dim, start_pos_arr)
    elif start_pos_arr.ndim == 1 and start_pos_arr.size == dim:
        pass
    else:
        raise ValueError("start_position must be a scalar or a 1D sequence of length 'dim'.")

    start_pos_arr = start_pos_arr.reshape(1, dim)
        
    # Create the list of steps, where each element is a random +1 or -1 in one of the dimensions
    steps = np.zeros((num_steps, dim))
    moving_dimension_indices = np.random.randint(0, dim, size=num_steps)
    directions = np.random.choice([-1, 1], size=num_steps)
    steps[np.arange(num_steps), moving_dimension_indices] = directions
    
    # Turn the steps into positions by cumulative sum, adding each step to the previous position and then displacing every position by the start_position
    displacements = np.cumsum(steps, axis=0)
    initial_position = np.zeros((1, dim)) 
    cum_displacements = np.concatenate((initial_position, displacements), axis=0) 
    positions = start_pos_arr + cum_displacements
    
    # Output a scalar if dim == 1, for cleanliness
    if dim == 1:
        positions = positions.squeeze()
        
    return positions

def generate_ensemble(num_walks: int, num_steps: int, dim: int = 1, 
                                 start_position: int | np.ndarray = 0, initial_seed: int = 42) -> np.ndarray:
    """
    Generates an ensemble of independent Simple Random Walk walks

    Args:
        num_walks (int): The number of independent walks (M) in the ensemble.
        num_steps (int): The number of steps (N) in each walk
        dim (int): The dimensionality of the walk
        start_position (int | np.ndarray): Starting position for all walks
        initial_seed (int): Base seed for simulation reproducibility

    Returns:
        np.ndarray: An array of shape (num_steps + 1, num_walks, dim)
                    containing the position of all walkers over time
    """
    np.random.seed(initial_seed)

    step_vectors = _get_step_vectors(dim)
    choice_indices = np.random.randint(
        0, 
        2 * dim, 
        size=(num_steps, num_walks)
    )
    all_steps = step_vectors[choice_indices]
    
    cumulative_displacement = np.cumsum(all_steps, axis=0)
    zeros_displacement = np.zeros((1, num_walks, dim), dtype=np.int64)
    walk_from_origin = np.concatenate((zeros_displacement, cumulative_displacement), axis=0)

    if np.isscalar(start_position):
        offset = start_position
    else:
        offset = np.asarray(start_position).reshape(1, 1, dim)
        
    final_positions = walk_from_origin + offset

    return final_positions

def plot_1d_walk(walk: np.ndarray, title: str = '1D Simple Random Walk'):
    """
    Plots a 1D walk versus time
    """
    plt.figure(figsize=(10, 4))
    plt.plot(walk, marker='o', markersize=0.3, linestyle='-', color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_2d_walk(walk: np.ndarray, title: str = '2D Simple Random Walk'):
    """
    Plots a 2D walk in the XY-plane with coloring to represent time
    """
    plt.figure(figsize=(6, 6))

    plt.scatter(walk[:, 0], walk[:, 1], c=np.arange(len(walk)), cmap='viridis', s=5, zorder=1)
    plt.plot(walk[:, 0], walk[:, 1], color='gray', linewidth=1, alpha=0.6, zorder=2)
    plt.plot(walk[0, 0], walk[0, 1], 'go', markersize=10, label='Start (Step 0)', zorder=3)
    plt.plot(walk[-1, 0], walk[-1, 1], 'ro', markersize=10, label='End (Step N)', zorder=3)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.colorbar(label='Time Step')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.legend()
    plt.show()
    
def plot_distribution(data: np.ndarray, bins: int = 30, title: str = 'Distribution', xlabel: str = 'Value', ylabel: str = 'Frequency'):
    """
    Plots the distribution of a 1D NumPy array with mean, median, and mode indicated
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_val = stats.mode(data).mode
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.axvline(mode_val, color='orange', linestyle=':', linewidth=2, label=f'Mode: {mode_val:.2f}')

    plt.legend(loc='best') 
    
    plt.show()
    
def plot_normality_check(data: np.ndarray, title: str = 'Normality Check', bins: int = 30):
    """
    Performs a visual and statistical check for normality on a 1D NumPy array
    Plots a distribution histogram with normal PDF overlay and a Q-Q plot
    """
    skewness = stats.skew(data)
    kurt = stats.kurtosis(data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Left Plot: Histogram with Normal PDF ---
    mu, sigma = stats.norm.fit(data)

    sns.histplot(data, bins=bins, kde=False, ax=axes[0], 
                 color='skyblue', edgecolor='black', alpha=0.7, 
                 stat='density')

    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, sigma)
    axes[0].plot(x, p, 'r', linewidth=2, label='Normal PDF')

    axes[0].set_title(f'{title} - Distribution vs. Normal PDF', fontsize=16)
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].text(0.95, 0.95, f'Skewness: {skewness:.2f}\nKurtosis: {kurt:.2f}', 
                 transform=axes[0].transAxes, 
                 fontsize=12, verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    axes[0].legend(loc='upper left')
    
    # --- Right Plot: Q-Q Plot ---
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Quantile-Quantile)', fontsize=16)
    axes[1].get_lines()[0].set_markerfacecolor('blue')
    axes[1].get_lines()[0].set_markeredgecolor('blue')
    axes[1].get_lines()[0].set_markersize(5)
    
    plt.tight_layout()
    plt.show()
    
def calculate_msd_fft(walk: np.ndarray) -> np.ndarray:
    """
    Calculates the time-averaged Mean Squared Displacement (MSD) of a 
    walk for any dimension using the Fast Fourier Transform (FFT) for 
    O(N log N) performance, summing the contributions from each dimension

    Args:
        walk (np.ndarray): The trajectory array. Can be 1D (N+1,) or 
                           2D (N+1, dim)

    Returns:
        np.ndarray: The total MSD values as a function of time lag (up to N/4)
    """

    def _calculate_msd_single_dim(trajectory_1d: np.ndarray) -> np.ndarray:
        N = len(trajectory_1d)

        fk = fft(trajectory_1d, n=2 * N) 
        power = fk * fk.conjugate()
        res = ifft(power)[:N].real

        n_minus_m = (N * np.ones(N) - np.arange(0, N))
        s2 = res / n_minus_m

        x2 = trajectory_1d ** 2
        s1 = np.zeros(N)
        s1[0] = np.average(x2) * 2.0
        for m in range(1, N):
            s1[m] = np.average(x2[m:] + x2[:-m])

        return s1 - 2 * s2

    
    if walk.ndim == 1:
        msd_total = _calculate_msd_single_dim(walk)
    elif walk.ndim == 2:
        num_dimensions = walk.shape[1]
        msd_sum = None
        
        for d in range(num_dimensions):
            dim_trajectory = walk[:, d]
            msd_d = _calculate_msd_single_dim(dim_trajectory)
            
            if msd_sum is None:
                msd_sum = msd_d
            else:
                msd_sum += msd_d
        
        if msd_sum is None:
            return np.array([0.0])
            
        msd_total = msd_sum
    else:
        raise ValueError("Input walk must be 1D or 2D (time, dimension).")
    
    return msd_total[:len(msd_total) // 4]
    
# ----- HELPER FUNCTIONS -----

def _get_step_vectors(dim: int) -> np.ndarray:
    vectors = np.zeros((2 * dim, dim), dtype=np.int64)
    for i in range(dim):
        vectors[2 * i, i] = 1
        vectors[2 * i + 1, i] = -1
    return vectors