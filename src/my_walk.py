from utils import generate_srw, generate_ensemble
from utils import plot_1d_walk, plot_distribution

def main():
  #-Make one random walk
  walk = generate_srw(num_steps=100, dim=1)
  plot_1d_walk(walk)
  
  #-Simulate multiple random walks-
  ensemble = generate_ensemble(
    num_walks=200,
    num_steps=500,
    dim=1
  )

  #-final position of all walks-
  final_positions = ensemble[-1, :, 0]
  plot_distribution(final_positions, title="Final Position Distribution")

if __name__ == "__main__":
  main()
