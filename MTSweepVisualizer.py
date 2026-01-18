import numpy as np
import plotly.graph_objects as go
from utils_env import show_or_save_plotly_figure
from simulator import CommNetworkSimulator

class MTSweepVisualizer:
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

                    ttds = pd_system.calculate_ttds(sim)
                    
                    if ttds is not None:
                        results.append((t, m, ttds))
                    else:
                        results.append((t, m, np.nan)) 
                        
                except ValueError as e:
                    # Likely OOM or invalid configuration
                    # print(f"Skipping T={t}, M={m} due to error: {e}")
                    results.append((t, m, np.nan))
                    continue

        return results

    def plot_3d(self, results, output_file="M_T_TTDS_sweep_3d.html"):
        """
        Generates a 3D surface plot of (T, M) -> TTDS
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
        ttdss = [r[2] for r in valid_results]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=ts, y=ms, z=ttdss,
            mode='markers',
            marker=dict(size=5, color=ttdss, colorscale='Viridis', colorbar=dict(title='TTDS (s)')),
            name='TTDS Data'
        ))
        # Try to plot a surface for regular grid data
        surface_drawn = False
        try:
            # Reshape data into grid
            unique_ts = np.unique(ts)
            unique_ms = np.unique(ms)
            grid_ts, grid_ms = np.meshgrid(unique_ts, unique_ms, indexing='ij')
            grid_ttdss = np.full(grid_ts.shape, np.nan)
            for i, t in enumerate(unique_ts):
                for j, m in enumerate(unique_ms):
                    for k in range(len(ts)):
                        if ts[k] == t and ms[k] == m:
                            grid_ttdss[i, j] = ttdss[k]
            # Only plot if grid is fully populated (no nans)
            if not np.isnan(grid_ttdss).any():
                fig.add_trace(go.Surface(
                    x=grid_ts,
                    y=grid_ms,
                    z=grid_ttdss,
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=False,
                    name='Surface'
                ))
                surface_drawn = True
            else:
                print("Grid has missing values, skipping surface plot.")
        except Exception as e:
            print(f"Could not plot surface: {e}")
        # Fallback to trisurf if surface not drawn
        if not surface_drawn:
            try:
                from scipy.spatial import Delaunay
                import plotly.figure_factory as ff
                points2d = np.column_stack((ts, ms))
                tri = Delaunay(points2d)
                fig_mesh = ff.create_trisurf(x=ts, y=ms, z=ttdss, simplices=tri.simplices, colormap='Viridis', show_colorbar=True)
                for trace in fig_mesh.data:
                    if trace.type == 'mesh3d':  # type: ignore[attr-defined]
                        trace.opacity = 0.7  # type: ignore[attr-defined]
                        trace.showscale = True  # type: ignore[attr-defined]
                        trace.colorbar = dict(title='TTDS (s)')  # type: ignore[attr-defined]
                        fig.add_trace(trace)
            except Exception as e:
                print(f"Could not plot trisurf fallback: {e}")
        pp_degree = self.base_config.get('pp_degree', 'N/A')
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Context Length, T)',
                yaxis_title='Y (Prefill Chunk Size, M)',
                zaxis_title='Z (TTDS, seconds)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            title=f'Impact of Context Length (T) and Chunk Size (M) on Time to Decode Start (TTDS) | PP={pp_degree}',
            margin=dict(l=0, r=0, b=0, t=40)
        )
        show_or_save_plotly_figure(fig, output_file)

