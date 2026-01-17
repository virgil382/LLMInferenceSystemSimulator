

from Gantt5DVisualizer import Gantt5DVisualizer
from simulator import CommNetworkSimulator, GPU, LLM
from GanttVisualizer import GanttVisualizer
from MTSweepVisualizer import MTSweepVisualizer
from PPTSweepVisualizer import PPTSweepVisualizer
from DisaggregatedPDSystemPP import DisaggregatedPDSystemPP
from utils_env import detect_environment

# --- Setup and Run ---

# The system works best with PP=8, T=8192, M=128 for large models like LLaMA-3.1-70B.

if __name__ == "__main__":
    # Common Configuration
    # We use H100s to ensure we have enough VRAM for larger contexts in the sweep
    system_config = {
        "llm": LLM.from_name("LLaMA-3.1-70B"),
        "prefill_gpu": GPU.from_name("NVIDIA H100"),
        "decode_gpu": GPU.from_name("NVIDIA RTX PRO 6000"),
        "pp_degree": 8,  # Increased PP degree to fit model
        "num_prefill_ib_cards": 1,
        "N": 1,
        "vram_limit_ratio": 0.95 
    }

    # 1. Single Run
    # Override T and M for specific scenario
    run_config = system_config.copy()
    run_config.update({"T": 8192, "M": 128})

    pd_system = DisaggregatedPDSystemPP(**run_config)

    # --- TTFT and TPOT extraction and Gantt visualization ---
    sim = CommNetworkSimulator()
    pd_system.start(sim)
    sim.run(pd_system)

    ttft = pd_system.calculate_ttft(sim)
    tpot = pd_system.calculate_tpot(sim)

    if ttft is not None and tpot is not None:
        print(f"TTFT (Time To First Token): {ttft:.6f} seconds")
        print(f"TPOT (Time Per Output Token): {tpot:.6f} seconds")
    else:
        print("No decode jobs found for TTFT/TPOT calculation.")

    print(f"Total Inference Latency: {sim.current_time:.4f} seconds")
    print(f"Prefill VRAM Utilization: {pd_system.prefill_vram_util:.2f}%")
    print(f"Decode VRAM Utilization: {pd_system.decode_vram_util:.2f}%")

    visualizer = GanttVisualizer(pd_system, 1.0)
    visualizer.generate(sim)

    # --- Parameter Sweep (M and T)---
    print("\n--- Running Parameter Sweep (3D Plot) ---")
    
    # Sweep T from 1024 to 8192 in steps of 1024
    # M varies from 64 up to T
    t_sweep_range = range(1024, 8192 + 1, 1024)
    
    sweeper = MTSweepVisualizer(
        system_cls=DisaggregatedPDSystemPP,
        base_config=system_config,
        t_range=t_sweep_range,
        m_start=64,  # Start value for M
        m_step=128,  # Step size for M
        m_end=2048   # Optional max value for M
    )
    
    results = sweeper.run_sweep()
    sweeper.plot_3d(results, output_file="M_T_TTFT_sweep_3d.html")

    # --- Parameter Sweep (PP and T)---
    print("\n--- Running PP-T Sweep (3D Plot) ---")
    pp_sweep_range = range(1, 9)  # Example: PP from 1 to 8
    t_sweep_range = range(1024, 8192 + 1, 1024)
    ppt_sweeper = PPTSweepVisualizer(
        system_cls=DisaggregatedPDSystemPP,
        base_config=system_config,
        pp_range=pp_sweep_range,
        t_range=t_sweep_range,
        m_value=64
    )
    ppt_results = ppt_sweeper.run_sweep()
    ppt_sweeper.plot_3d(ppt_results, output_file="PP_T_TTFT_sweep_3d.html")

    # --- 5D Interactive Gantt Visualizer (PP, M, T, N) ---
    print("\n--- Starting Interactive 5D Gantt Visualizer ---")
    env = detect_environment()
    if env == 'jupyter' or env == 'colab':
        # Jupyter or Colab
        try:
            from jupyter_dash import JupyterDash  # type: ignore[import-unresolved]
            import Gantt5DVisualizer as gantt5d_mod
            # Patch the class to use JupyterDash
            class JupyterGantt5DVisualizer(gantt5d_mod.Gantt5DVisualizer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.app = JupyterDash(__name__)
                    # Re-inject layout and callbacks after replacing app
                    self._setup_layout()
                    self._setup_callbacks()
                def run(self, debug=False, port=8050):
                    self.app.run_server(mode='inline', debug=debug, port=port)
            visualizer_5d = JupyterGantt5DVisualizer(system_config)
            visualizer_5d.run(debug=False)
        except ImportError:
            print("JupyterDash is not installed. Please install it with 'pip install jupyter-dash'.")
    else:
        # VS Code
        print("Point your browser to http://127.0.0.1:8050")
        visualizer_5d = Gantt5DVisualizer(system_config)
        visualizer_5d.run(debug=False)