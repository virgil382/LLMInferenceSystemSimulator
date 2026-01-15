
from simulator import CommNetworkSimulator, GPU, LLM
from GanttVisualizer import GanttVisualizer
from MTSweepVisualizer import MTSweepVisualizer
from PPTSweepVisualizer import PPTSweepVisualizer
from DisaggregatedPDSystemPP import DisaggregatedPDSystemPP

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

    sim = CommNetworkSimulator()
    pd_system.start(sim)
    sim.run(pd_system)

    # --- TTFT and TPOT extraction ---
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

    # Visualization
    visualizer = GanttVisualizer(pd_system, 1.0)
    visualizer.generate(sim)

    # --- Parameter Sweep ---
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
    sweeper.plot_3d(results, output_file="M_T_TTFT_sweep_3d.png")

    # --- PP-T Sweep ---
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
    ppt_sweeper.plot_3d(ppt_results, output_file="PP_T_TTFT_sweep_3d.png")

