import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulator import CommNetworkSimulator

class PPTSweepVisualizer:
    def __init__(self, system_cls, base_config, pp_range, t_range, m_value=64):
        """
        Initialize the visualizer for PP and T sweep.
        :param system_cls: The class of the system to simulate (e.g. DisaggregatedPDSystem)
        :param base_config: Dictionary of arguments to pass to system_cls constructor (excluding PP, T, M)
        :param pp_range: List or Range of PP values to sweep (e.g., range(1, 9))
        :param t_range: List or Range of T values to sweep (e.g., range(128, 4096, 128))
        :param m_value: Fixed value for M (chunk size), default is 64
        """
        self.system_cls = system_cls
        self.base_config = base_config
        self.pp_range = pp_range
        self.t_range = t_range
        self.m_value = m_value

    def run_sweep(self):
        results = []
        print(f"Starting parameter sweep along PP={self.pp_range} and T={self.t_range} with fixed M={self.m_value}...")
        for pp in self.pp_range:
            for t in self.t_range:
                try:
                    config = self.base_config.copy()
                    config['pp_degree'] = pp
                    config['T'] = t
                    config['M'] = self.m_value
                    pd_system = self.system_cls(**config)
                    sim = CommNetworkSimulator()
                    pd_system.start(sim)
                    sim.run(pd_system)
                    ttft = pd_system.calculate_ttft(sim)
                    if ttft is not None:
                        results.append((pp, t, ttft))
                    else:
                        results.append((pp, t, np.nan))
                except ValueError as e:
                    results.append((pp, t, np.nan))
                    continue
        return results

    def plot_3d(self, results, output_file="PP_T_TTFT_sweep_3d.png"):
        """
        Generates a 3D surface plot of (PP, T) -> TTFT
        """
        if not results:
            print("No results to plot.")
            return
        valid_results = [r for r in results if not np.isnan(r[2])]
        if not valid_results:
            print("No valid results (all NaN) to plot.")
            return
        pps = [r[0] for r in valid_results]
        ts = [r[1] for r in valid_results]
        ttfts = [r[2] for r in valid_results]
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        # type: ignore[arg-type] -- ttfts is a list of floats, which is valid for matplotlib's zs argument
        sc = ax.scatter(pps, ts, ttfts, c=ttfts, cmap='viridis', marker='o')  # type: ignore[arg-type] 
        try:
            ax.plot_trisurf(pps, ts, ttfts, cmap='viridis', alpha=0.5, linewidth=0.2, edgecolor='gray')
        except Exception as e:
            print(f"Could not plot surface: {e}")
        ax.set_xlabel('Pipeline Parallelism (PP)')
        ax.set_ylabel('Context Length (T)')
        ax.set_zlabel('TTFT (seconds)')
        ax.set_title('Impact of PP and T on TTFT (M=64)')
        fig.colorbar(sc, shrink=0.5, aspect=5, label='TTFT (s)')

        # Rotate the chart clockwise on the vertical axis (adjust azimuth)
        # Default is usually azim=-60. We'll rotate it by -20 degrees.
        ax.view_init(elev=20, azim=-20) 


        plt.tight_layout()
        plt.savefig(output_file)
        print(f"3D Sweep Chart saved to {output_file}")
