from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from numpy.fft import fft, ifft

# ==================== RANDOM WALK GENERATION ====================

def generate_srw(num_steps: int, start_position: Union[int, list, np.ndarray] = 0, 
                 dim: int = 1, seed: int = None) -> np.ndarray:
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

    start_pos_arr = np.asarray(start_position)

    if start_pos_arr.ndim == 0:
        start_pos_arr = np.full(dim, start_pos_arr)
    elif start_pos_arr.ndim == 1 and start_pos_arr.size == dim:
        pass
    else:
        raise ValueError("start_position must be a scalar or a 1D sequence of length 'dim'.")

    start_pos_arr = start_pos_arr.reshape(1, dim)
        
    steps = np.zeros((num_steps, dim))
    moving_dimension_indices = np.random.randint(0, dim, size=num_steps)
    directions = np.random.choice([-1, 1], size=num_steps)
    steps[np.arange(num_steps), moving_dimension_indices] = directions
    
    displacements = np.cumsum(steps, axis=0)
    initial_position = np.zeros((1, dim)) 
    cum_displacements = np.concatenate((initial_position, displacements), axis=0) 
    positions = start_pos_arr + cum_displacements
    
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
    choice_indices = np.random.randint(0, 2 * dim, size=(num_steps, num_walks))
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


# ==================== STATISTICAL ANALYSIS ====================

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


def calculate_ensemble_msd(ensemble: np.ndarray) -> np.ndarray:
    """
    Calculate ensemble-averaged MSD from multiple walks
    
    Args:
        ensemble: Array of shape (num_steps+1, num_walks, dim)
        
    Returns:
        np.ndarray: Ensemble-averaged MSD
    """
    num_steps, num_walks, dim = ensemble.shape

    squared_displacements = np.sum(ensemble**2, axis=2) 

    msd = np.mean(squared_displacements, axis=1)
    
    return msd


def calculate_diffusion_coefficient(msd: np.ndarray, time_steps: np.ndarray, 
                                    dim: int, fit_range: Tuple[int, int] = None) -> Tuple[float, dict]:
    """
    Calculate diffusion coefficient from MSD vs time data
    For d-dimensional random walk: MSD = 2*d*D*t
    
    Args:
        msd: Mean squared displacement values
        time_steps: Corresponding time steps
        dim: Dimensionality of the walk
        fit_range: Optional tuple (start_idx, end_idx) for linear fit
        
    Returns:
        D: Diffusion coefficient
        fit_info: Dictionary with slope, intercept, r_squared
    """
    if fit_range is None:
        start_idx = len(msd) // 4
        end_idx = 3 * len(msd) // 4
    else:
        start_idx, end_idx = fit_range
    
    x = time_steps[start_idx:end_idx]
    y = msd[start_idx:end_idx]
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # D = slope / (2*d)
    D = slope / (2 * dim)
    
    fit_info = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    return D, fit_info


def calculate_return_probability(ensemble: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Calculate probability of returning to origin (within threshold) over time
    
    Args:
        ensemble: Array of shape (num_steps+1, num_walks, dim)
        threshold: Distance threshold for considering "at origin"
        
    Returns:
        np.ndarray: Return probability at each time step
    """
    num_steps, num_walks, dim = ensemble.shape

    distances = np.sqrt(np.sum(ensemble**2, axis=2))

    at_origin = distances <= threshold
    
    return_prob = np.mean(at_origin, axis=1)
    
    return return_prob


def analyze_first_passage_time(ensemble: np.ndarray, target: float) -> dict:
    """
    Analyze first passage time statistics to reach target distance from origin
    
    Args:
        ensemble: Array of shape (num_steps+1, num_walks, dim)
        target: Target distance from origin
        
    Returns:
        dict: Statistics including mean, median, std of first passage times
    """
    num_steps, num_walks, dim = ensemble.shape

    distances = np.sqrt(np.sum(ensemble**2, axis=2))

    first_passage_times = []
    for walk_idx in range(num_walks):
        reached = np.where(distances[:, walk_idx] >= target)[0]
        if len(reached) > 0:
            first_passage_times.append(reached[0])
    
    if len(first_passage_times) == 0:
        return {'mean': None, 'median': None, 'std': None, 'success_rate': 0.0}
    
    fpt_array = np.array(first_passage_times)
    
    return {
        'mean': np.mean(fpt_array),
        'median': np.median(fpt_array),
        'std': np.std(fpt_array),
        'success_rate': len(first_passage_times) / num_walks,
        'data': fpt_array
    }


# ==================== VISUALIZATION ====================

def plot_1d_walk(walk: np.ndarray, title: str = '1D Simple Random Walk'):
    """Plots a 1D walk versus time"""
    plt.figure(figsize=(10, 4))
    plt.plot(walk, marker='o', markersize=0.3, linestyle='-', color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_2d_walk(walk: np.ndarray, title: str = '2D Simple Random Walk'):
    """Plots a 2D walk in the XY-plane with coloring to represent time"""
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


def plot_3d_walk(walk: np.ndarray, title: str = '3D Simple Random Walk'):
    """
    Plots a 3D walk with coloring to represent time
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.arange(len(walk))
    scatter = ax.scatter(walk[:, 0], walk[:, 1], walk[:, 2], 
                        c=colors, cmap='viridis', s=2, alpha=0.6)

    ax.plot(walk[:, 0], walk[:, 1], walk[:, 2], 
           color='gray', linewidth=0.5, alpha=0.3)
    
    ax.scatter(walk[0, 0], walk[0, 1], walk[0, 2], 
              color='green', s=100, marker='o', label='Start', zorder=10)
    ax.scatter(walk[-1, 0], walk[-1, 1], walk[-1, 2], 
              color='red', s=100, marker='o', label='End', zorder=10)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(title)
    fig.colorbar(scatter, label='Time Step', shrink=0.5)
    ax.legend()
    plt.show()


def plot_ensemble_trajectories(ensemble: np.ndarray, max_walks: int = 100, 
                               title: str = 'Ensemble of Random Walks', alpha: float = 0.3):
    """
    Plot multiple walks from an ensemble
    
    Args:
        ensemble: Array of shape (num_steps+1, num_walks, dim)
        max_walks: Maximum number of walks to plot
        title: Plot title
        alpha: Transparency of individual walks
    """
    num_steps, num_walks, dim = ensemble.shape
    
    if dim == 1:
        plt.figure(figsize=(12, 5))
        for i in range(min(num_walks, max_walks)):
            plt.plot(ensemble[:, i, 0], alpha=alpha, linewidth=0.5)
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    elif dim == 2:
        plt.figure(figsize=(8, 8))
        for i in range(min(num_walks, max_walks)):
            plt.plot(ensemble[:, i, 0], ensemble[:, i, 1], alpha=alpha, linewidth=0.5)
        plt.scatter(0, 0, color='green', s=100, marker='o', label='Origin', zorder=5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(min(num_walks, max_walks)):
            ax.plot(ensemble[:, i, 0], ensemble[:, i, 1], ensemble[:, i, 2], 
                   alpha=alpha, linewidth=0.5)
        
        ax.scatter(0, 0, 0, color='green', s=100, marker='o', label='Origin', zorder=10)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(title)
        ax.legend()
        plt.show()


def plot_msd_analysis(msd: np.ndarray, time_steps: np.ndarray = None, 
                     theoretical_slope: float = None, dim: int = 1):
    """
    Plot MSD with theoretical prediction and log-log analysis
    
    Args:
        msd: Mean squared displacement values
        time_steps: Time steps (if None, uses indices)
        theoretical_slope: Theoretical slope (2*dim for random walk)
        dim: Dimensionality
    """
    if time_steps is None:
        time_steps = np.arange(len(msd))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear plot
    ax1.plot(time_steps, msd, 'b-', linewidth=2, label='Simulated MSD')
    if theoretical_slope is None:
        theoretical_slope = 2 * dim
    ax1.plot(time_steps, theoretical_slope * time_steps, 'r--', 
             linewidth=2, label=f'Theory: MSD = {theoretical_slope}t')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('MSD', fontsize=12)
    ax1.set_title('Mean Squared Displacement vs Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot
    valid_idx = (time_steps > 0) & (msd > 0)
    ax2.loglog(time_steps[valid_idx], msd[valid_idx], 'b-', linewidth=2, label='Simulated')
    ax2.loglog(time_steps[valid_idx], theoretical_slope * time_steps[valid_idx], 
               'r--', linewidth=2, label='Theory (slope=1)')
    ax2.set_xlabel('Time Step (log)', fontsize=12)
    ax2.set_ylabel('MSD (log)', fontsize=12)
    ax2.set_title('Log-Log Plot (Scaling Analysis)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()


def plot_distribution(data: np.ndarray, bins: int = 30, title: str = 'Distribution', 
                     xlabel: str = 'Value', ylabel: str = 'Frequency'):
    """Plots the distribution of a 1D NumPy array with mean, median, and mode indicated"""
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_val = stats.mode(data, keepdims=True).mode[0]
    
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
    
    # Histogram with Normal PDF
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
    
    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Quantile-Quantile)', fontsize=16)
    axes[1].get_lines()[0].set_markerfacecolor('blue')
    axes[1].get_lines()[0].set_markeredgecolor('blue')
    axes[1].get_lines()[0].set_markersize(5)
    
    plt.tight_layout()
    plt.show()


# ==================== HELPER FUNCTIONS ====================

def _get_step_vectors(dim: int) -> np.ndarray:
    """Generate all possible step vectors for d-dimensional lattice walk"""
    vectors = np.zeros((2 * dim, dim), dtype=np.int64)
    for i in range(dim):
        vectors[2 * i, i] = 1
        vectors[2 * i + 1, i] = -1
    return vectors