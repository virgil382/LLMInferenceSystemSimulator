from simulator import CommNetworkSimulator, DataBatch, ComputeJob, SimulatedSystem, CommChannel, GPU, LLM

class DisaggregatedPDSystem(SimulatedSystem):
    def __init__(self, llm: LLM, gpu: GPU, pp_prefill: int, pp_decode: int, 
                 num_ib_cards: int, n: int, t: int, m: int):
        self.llm = llm
        self.gpu = gpu
        self.pp_p = pp_prefill
        self.pp_d = pp_decode
        self.num_ib = num_ib_cards
        self.N = n  # Batch size
        self.T = t  # Context length
        self.M = m  # Prefill chunk size
        
        # Internal state tracking
        self.current_prefill_token_idx = 0
        self.prefill_complete = False
        
        # Hardware setup
        self.p_gpu_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(pp_prefill)]
        self.d_gpu_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(pp_decode)]
        self.p_ib_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(num_ib_cards)]
        self.p_ib_cables = [CommChannel("Infiniband NDR") for _ in range(num_ib_cards)]
        self.d_ib_cables = [CommChannel("Infiniband NDR") for _ in range(pp_decode)]
        self.d_eth_lanes = [CommChannel("PCIe Gen4") for _ in range(pp_decode)]
        self.d_eth_cables = [CommChannel("Ethernet 100G") for _ in range(pp_decode)]

    def _get_activation_size(self):
        # N * M tokens * Hidden Size * Bytes per param
        return self.N * self.M * self.llm.H_model * self.llm.B

    def _get_kv_share_size(self):
        # KV cache for N * M tokens for a single pipeline rank
        return self.llm.KV(self.N, self.M) // self.pp_p

    def T_prefill(self, spec_prefill_enabled: bool = False):
        """
        More accurate prefill time calculation, including quadratic attention FLOPS and memory bandwidth.
        """
        N = self.N
        T = self.M  # Chunk size for prefill
        H_model = self.llm.H_model
        L_rank = self.llm.L // self.pp_p

        # 1. Compute Time (FLOPS)
        # Projections: Q, K, V, and O projections + MLP (up/down/gate)
        # Most models approximate total linear FLOPS as 2 * N * T * W_per_rank
        # But to be explicit about the H^2 in the attention block projections:
        # flops_attn_projections = L_rank * (8 * N * T * H_model**2)

        # Linear FLOPS (MLP + Attention Projections)
        flops_linear = 2 * N * T * ((self.llm.W // self.llm.L) * L_rank)

        # Quadratic Attention FLOPS (The QK^T and Score * V part)
        # Standard formula: 2 * N * L_rank * H_model * T^2
        flops_quadratic = 2 * N * L_rank * H_model * (T ** 2)

        # A constant factor representing the reduction in attention complexity
        spec_prefill_multiplier = 0.4 if spec_prefill_enabled else 1.0
        flops_quadratic *= spec_prefill_multiplier

        total_flops = flops_linear + flops_quadratic
        t_compute = total_flops / self.gpu.flops

        # 2. Memory Time (VRAM BW)
        # Must read weights + write the resulting KV cache
        weight_bytes = ((self.llm.W // self.llm.L) * L_rank) * self.llm.B
        kv_bytes = self.llm.KV(N, T) // self.pp_p
        vram_util = 0.80
        t_memory = (weight_bytes + kv_bytes) / (self.gpu.vram_bw_bps * vram_util)

        return max(t_compute, t_memory)

    def start(self, simulator):
        # Trigger the first prefill rank compute
        compute_time = self.T_prefill()
        simulator.add_compute(ComputeJob(f"P_Rank_0_Chunk_0", compute_time))

    def on_compute_complete(self, simulator, job):
        parts = job.name.split("_")
        if "P_Rank" in job.name:
            r_idx = int(parts[2])
            c_idx = int(parts[4])
            self.on_prefill_compute_complete(simulator, r_idx, c_idx)
        elif "D_Rank" in job.name:
            r_idx = int(parts[2])
            self.on_decode_compute_complete(simulator, r_idx)

    def on_prefill_compute_complete(self, simulator, r_idx, c_idx):
        # 1. Start Handoff to Decode Cluster (Path 2)
        ib_idx = r_idx % self.num_ib
        path2 = [
            self.p_gpu_lanes[r_idx],    # GPU -> PLX
            self.p_ib_lanes[ib_idx],    # PLX -> IB Card
            self.p_ib_cables[ib_idx],   # Prefill IB Card -> IB Switch
            self.d_ib_cables[r_idx],    # IB Switch -> Decode IB Card
            self.d_gpu_lanes[r_idx]     # IB Card -> Decode GPU
        ]
        simulator.add_batch(DataBatch(f"Handoff_Rank_{r_idx}_Chunk_{c_idx}", 
                                      self._get_kv_share_size(), path2))

        # 2. Start Inter-rank Activation (Path 1)
        if r_idx + 1 < self.pp_p:
            path1 = [self.p_gpu_lanes[r_idx], self.p_gpu_lanes[r_idx+1]]  # GPU -> PLX -> Next GPU
            simulator.add_batch(DataBatch(f"Prefill_Act_Rank_{r_idx}_Chunk_{c_idx}", 
                                          self._get_activation_size(), path1))
        
        # 3. If rank 0, check if we need to start next prefill chunk (Pipeline overlap)
        if r_idx == 0 and (c_idx + 1) * self.M < self.T:
            comp_time = self.T_prefill()
            simulator.add_compute(ComputeJob(f"P_Rank_0_Chunk_{c_idx+1}", comp_time))

    def on_decode_compute_complete(self, simulator, r_idx):
        # Decode Logic: Path 3 (Inter-server Ethernet)
        if r_idx + 1 < self.pp_d:
            path3 = [
                self.d_gpu_lanes[r_idx], 
                self.d_eth_lanes[r_idx], 
                self.d_eth_cables[r_idx], 
                self.d_eth_cables[r_idx+1], 
                self.d_eth_lanes[r_idx+1], 
                self.d_gpu_lanes[r_idx+1]
            ]
            # Activation size for 1 token decode
            dec_act_size = self.N * 1 * self.llm.H_model * self.llm.B
            simulator.add_batch(DataBatch(f"Decode_Act_Rank_{r_idx}", dec_act_size, path3))

    def on_data_transfer_complete(self, simulator, batch):
        # Logic to trigger next compute rank or handoff
        parts = batch.name.split("_")
        rank_idx = int(parts[2])
        chunk_idx = int(parts[4])

        if "Prefill_Act" in batch.name:
            # Move to next prefill compute
            if rank_idx + 1 < self.pp_p:
                comp_time = self.T_prefill()
                return ComputeJob(f"P_Rank_{rank_idx+1}_Chunk_{chunk_idx}", comp_time)
            else:
                # Last prefill rank finished a chunk
                self.current_prefill_token_idx += self.M
                if self.current_prefill_token_idx >= self.T:
                    self.prefill_complete = True
                    # Start Decode Phase
                    return ComputeJob(f"D_Rank_0_Step_0", 0.001) # Trigger first decode
        
        return None


# --- Setup and Run ---
my_llm = LLM.from_name("LLaMA-3.1-70B")
my_gpu = GPU.from_name("NVIDIA H100")

# System: 8 Prefill Ranks, 4 Decode Ranks, 2 IB Cards in Prefill Server
# N=32, Context=2048, Chunk Size=512
pd_system = DisaggregatedPDSystem(
    llm=my_llm,
    gpu=my_gpu,
    pp_prefill=8,
    pp_decode=4,
    num_ib_cards=2,
    n=32,
    t=2048,
    m=512
)

sim = CommNetworkSimulator()
pd_system.start(sim)
sim.run(pd_system)

# --- TTFT and TPOT extraction ---

# Collect decode and prefill jobs.
decode_jobs = [job for job in sim.completed_compute if job.name.startswith("D_Rank")]
decode_jobs_sorted = sorted(decode_jobs, key=lambda j: j.end_time)

prefill_jobs = [job for job in sim.completed_compute if job.name.startswith("P_Rank")]
prefill_jobs_sorted = sorted(prefill_jobs, key=lambda j: j.end_time)

# Assign jobs to well-named variables for clarity.
first_prefill_job = prefill_jobs_sorted[0] if prefill_jobs_sorted else None
first_decode_job = decode_jobs_sorted[0] if decode_jobs_sorted else None
last_decode_job = decode_jobs_sorted[-1] if decode_jobs_sorted else None

# Calculate TTFT and TPOT using extracted jobs.
if first_decode_job and last_decode_job:
    ttft = first_decode_job.start_time
    print(f"TTFT (Time To First Token): {ttft:.6f} seconds")

    tpot = last_decode_job.end_time - first_decode_job.start_time
    print(f"TPOT (Time Per Output Token): {tpot:.6f} seconds")
else:
    print("No decode jobs found for TTFT/TPOT calculation.")

print(f"Total Inference Latency: {sim.current_time:.4f} seconds")
