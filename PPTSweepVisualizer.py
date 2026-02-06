import numpy as np
import plotly.graph_objects as go
from utils_env import show_or_save_plotly_figure
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
                    ttds = pd_system.calculate_ttds(sim)
                    if ttds is not None:
                        results.append((pp, t, ttds))
                    else:
                        results.append((pp, t, np.nan))
                except ValueError as e:
                    results.append((pp, t, np.nan))
                    continue
        return results

    def plot_3d(self, results, output_file="PP_T_TTDS_sweep_3d.html"):
        """
        Generates a 3D surface plot of (PP, T) -> TTDS
        """
        global np
        if not results:
            print("No results to plot.")
            return
        valid_results = [r for r in results if not np.isnan(r[2])]
        if not valid_results:
            print("No valid results (all NaN) to plot.")
            return
        pps = [r[0] for r in valid_results]
        ts = [r[1] for r in valid_results]
        ttds = [r[2] for r in valid_results]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=pps, y=ts, z=ttds,
            mode='markers',
            marker=dict(size=5, color=ttds, colorscale='Viridis', colorbar=dict(title='TTDS (s)')),
            name='TTDS Data'
        ))
        # Try to plot a surface for regular grid data
        try:
            # Reshape data into grid
            unique_pps = np.unique(pps)
            unique_ts = np.unique(ts)
            grid_pps, grid_ts = np.meshgrid(unique_pps, unique_ts, indexing='ij')
            grid_ttds = np.full(grid_pps.shape, np.nan)
            for i, pp in enumerate(unique_pps):
                for j, t in enumerate(unique_ts):
                    for k in range(len(pps)):
                        if pps[k] == pp and ts[k] == t:
                            grid_ttds[i, j] = ttds[k]
            # Only plot if grid is fully populated (no nans)
            if not np.isnan(grid_ttds).any():
                fig.add_trace(go.Surface(
                    x=grid_pps,
                    y=grid_ts,
                    z=grid_ttds,
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=False,
                    name='Surface'
                ))
            else:
                print("Grid has missing values, skipping surface plot.")
        except Exception as e:
            print(f"Could not plot surface: {e}")
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Pipeline Parallelism, PP)',
                yaxis_title='Y (Context Length, T)',
                zaxis_title='Z (TTDS, seconds)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            title=f'Impact of PP and T on Time to Decode Start (TTDS) | M={self.m_value}',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        show_or_save_plotly_figure(fig, output_file)
