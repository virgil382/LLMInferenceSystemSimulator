import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulator import CommNetworkSimulator

class ParameterSweepVisualizer:
    def __init__(self, system_cls, base_config, t_range, m_start=None, m_step=64, m_end=None):
        """
        Initialize the visualizer.
        :param system_cls: The class of the system to simulate (e.g. DisaggregatedPDSystem)
        :param base_config: Dictionary of arguments to pass to system_cls constructor (excluding T and M)
        :param t_range: List or Range of T values to sweep (e.g., range(128, 4096, 128))
        :param m_start: Start value for M (chunk size). If None, defaults to m_step.
        :param m_step: Step size for M (chunk size)
        :param m_end: Optional max value for M. If None, sweeps up to T.
        """
        self.system_cls = system_cls
        self.base_config = base_config
        self.t_range = t_range
        self.m_step = m_step
        self.m_start = m_start if m_start is not None else m_step
        self.m_end = m_end

    def run_sweep(self):
        results = []

        print(f"Starting parameter sweep along T={self.t_range} with M start={self.m_start} step={self.m_step}...")

        for t in self.t_range:
            # Determine the upper bound for this specific T
            current_m_end = t
            if self.m_end is not None:
                current_m_end = min(t, self.m_end)

            # For each T, sweep M from m_start up to current_m_end
            # Ensure at least one value for M is tested
            m_values = list(range(self.m_start, current_m_end + 1, self.m_step))
            if not m_values:
                m_values = [t]
            
            for m in m_values:
                try:
                    # Create configuration for this run
                    config = self.base_config.copy()
                    config['T'] = t
                    config['M'] = m
                    
                    # Instantiate system
                    pd_system = self.system_cls(**config)

                    sim = CommNetworkSimulator()
                    pd_system.start(sim)
                    sim.run(pd_system)

                    ttft = pd_system.calculate_ttft(sim)
                    
                    if ttft is not None:
                        results.append((t, m, ttft))
                    else:
                        results.append((t, m, np.nan)) 
                        
                except ValueError as e:
                    # Likely OOM or invalid configuration
                    # print(f"Skipping T={t}, M={m} due to error: {e}")
                    results.append((t, m, np.nan))
                    continue

        return results

    def plot_3d(self, results, output_file="ttft_sweep_3d.png"):
        """
        Generates a 3D surface plot of (T, M) -> TTFT
        """
        if not results:
            print("No results to plot.")
            return

        # Filter out NaNs
        valid_results = [r for r in results if not np.isnan(r[2])]
        if not valid_results:
             print("No valid results (all NaN) to plot.")
             return

        ts = [r[0] for r in valid_results]
        ms = [r[1] for r in valid_results]
        ttfts = [r[2] for r in valid_results]

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for data points
        sc = ax.scatter(ts, ms, ttfts, c=ttfts, cmap='viridis', marker='o')
        
        # Try to plot a surface as well for better visualization if grid is regular enough
        try:
            ax.plot_trisurf(ts, ms, ttfts, cmap='viridis', alpha=0.5, linewidth=0.2, edgecolor='gray')
        except Exception as e:
            print(f"Could not plot surface: {e}")

        # Labels
        ax.set_xlabel('Context Length (T)')
        ax.set_ylabel('Prefill Chunk Size (M)')
        ax.set_zlabel('TTFT (seconds)')
        ax.set_title('Impact of Context Length (T) and Chunk Size (M) on TTFT')

        # Add a color bar which maps values to colors.
        fig.colorbar(sc, shrink=0.5, aspect=5, label='TTFT (s)')

        # Rotate the chart clockwise on the vertical axis (adjust azimuth)
        # Default is usually azim=-60. We'll rotate it by -20 degrees.
        ax.view_init(elev=45, azim=-20) 

        plt.tight_layout()
        plt.savefig(output_file)
        print(f"3D Sweep Chart saved to {output_file}")
