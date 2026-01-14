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
        self.p_gpu_lanes = [CommChannel("PCI Express 5.0 x16") for _ in range(pp_prefill)]
        self.d_gpu_lanes = [CommChannel("PCI Express 5.0 x16") for _ in range(pp_decode)]
        self.p_ib_lanes = [CommChannel("PCI Express 5.0 x16") for _ in range(num_ib_cards)]
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

    def start(self, simulator):
        # Trigger the first prefill rank compute
        # Duration estimated as (FLOPs needed / GPU FLOPS)
        compute_time = (2 * self.N * self.M * (self.llm.W / self.pp_p)) / self.gpu.flops
        simulator.add_compute(ComputeJob(f"P_Rank_0_Chunk_0", compute_time))

    def on_data_transfer_complete(self, simulator, batch):
        # Logic to trigger next compute rank or handoff
        parts = batch.name.split("_")
        rank_idx = int(parts[2])
        chunk_idx = int(parts[4])

        if "Prefill_Act" in batch.name:
            # Move to next prefill compute
            if rank_idx + 1 < self.pp_p:
                comp_time = (2 * self.N * self.M * (self.llm.W / self.pp_p)) / self.gpu.flops
                return ComputeJob(f"P_Rank_{rank_idx+1}_Chunk_{chunk_idx}", comp_time)
            else:
                # Last prefill rank finished a chunk
                self.current_prefill_token_idx += self.M
                if self.current_prefill_token_idx >= self.T:
                    self.prefill_complete = True
                    # Start Decode Phase
                    return ComputeJob(f"D_Rank_0_Step_0", 0.001) # Trigger first decode
        
        return None

    def on_compute_complete(self, simulator, job):
        parts = job.name.split("_")
        
        if "P_Rank" in job.name:
            r_idx = int(parts[2])
            c_idx = int(parts[4])
            
            # 1. Start Handoff to Decode Cluster (Path 2)
            ib_idx = r_idx % self.num_ib
            path2 = [
                self.p_gpu_lanes[r_idx], 
                self.p_ib_lanes[ib_idx], 
                self.p_ib_cables[ib_idx], 
                self.d_ib_cables[r_idx], 
                self.d_gpu_lanes[r_idx]
            ]
            simulator.add_batch(DataBatch(f"Handoff_Rank_{r_idx}_Chunk_{c_idx}", 
                                          self._get_kv_share_size(), path2))

            # 2. Start Inter-rank Activation (Path 1)
            if r_idx + 1 < self.pp_p:
                path1 = [self.p_gpu_lanes[r_idx], self.p_gpu_lanes[r_idx+1]]
                simulator.add_batch(DataBatch(f"Prefill_Act_Rank_{r_idx}_Chunk_{c_idx}", 
                                              self._get_activation_size(), path1))
            
            # 3. If rank 0, check if we need to start next prefill chunk (Pipeline overlap)
            if r_idx == 0 and (c_idx + 1) * self.M < self.T:
                comp_time = (2 * self.N * self.M * (self.llm.W / self.pp_p)) / self.gpu.flops
                simulator.add_compute(ComputeJob(f"P_Rank_0_Chunk_{c_idx+1}", comp_time))

        elif "D_Rank" in job.name:
            r_idx = int(parts[2])
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

# --- Setup and Run ---
my_llm = LLM.from_name("LLaMA-3.1-70B")
my_gpu = GPU.from_name("NVIDIA H100")

# System: 8 Prefill Ranks, 4 Decode Ranks, 2 IB Cards in Prefill Server
# N=32, Context=2048, Chunk Size=512
pd_system = DisaggregatedPDSystem(my_llm, my_gpu, 8, 4, 2, 32, 2048, 512)

sim = CommNetworkSimulator()
pd_system.start(sim)
sim.run(pd_system)

print(f"Total Inference Latency: {sim.current_time:.4f} seconds")
