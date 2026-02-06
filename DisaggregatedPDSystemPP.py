from simulator import DataBatch, ComputeJob, SimulatedSystem, CommChannel, GPU, LLM

class DisaggregatedPDSystemPP(SimulatedSystem):
    def __init__(self, llm: LLM, prefill_gpu: GPU, decode_gpu: GPU, pp_degree: int, 
                 num_prefill_ib_cards: int, N: int, T: int, M: int, vram_limit_ratio: float = 0.75):
        self.llm = llm
        self.prefill_gpu = prefill_gpu
        self.decode_gpu = decode_gpu
        self.pp_p = pp_degree
        self.pp_d = pp_degree
        self.num_prefill_ib_cards = num_prefill_ib_cards
        self.N = N  # Batch size
        self.T = T  # Context length
        self.M = M  # Prefill chunk size
        self.vram_limit_ratio = vram_limit_ratio
        
        # Internal state tracking
        self.current_prefill_token_idx = 0
        self.prefill_complete = False
        self.completed_transfers = set()
        
        # Hardware setup
        self.p_gpu_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.pp_p)]
        self.d_gpu_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.pp_d)]
        self.p_ib_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.num_prefill_ib_cards)]
        self.p_ib_cables = [CommChannel("Infiniband NDR") for _ in range(self.num_prefill_ib_cards)]
        self.d_ib_cables = [CommChannel("Infiniband NDR") for _ in range(self.pp_d)]
        self.d_ib_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.pp_d)]
        self.d_eth_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.pp_d)]
        self.d_eth_cables = [CommChannel("Ethernet 100G") for _ in range(self.pp_d)]

        # VRAM Validation
        # 1. Weights
        # Assuming parameters are evenly distributed across layers, and layers are distributed across ranks.
        # Note: If L is not divisible by pp_degree, integer division floors it, which is approximate but valid for checks.
        params_per_rank = (self.llm.W // self.llm.L) * (self.llm.L // self.pp_p)
        weight_bytes = params_per_rank * self.llm.B
        
        # 2. KV Cache (at full context T)
        kv_bytes = self.llm.KV(self.N, self.T) // self.pp_p
        
        total_mem_b = weight_bytes + kv_bytes
        
        # Check Prefill
        prefill_limit = self.prefill_gpu.vram_b * self.vram_limit_ratio
        if total_mem_b > prefill_limit:
            raise ValueError(f"Prefill GPU OOM (> {self.vram_limit_ratio*100:.1f}%): Rank needs {total_mem_b/1e9:.2f} GB, but allowed is {prefill_limit/1e9:.2f} GB (Total: {self.prefill_gpu.vram_b/1e9:.2f} GB)")
        self.prefill_vram_util = (total_mem_b / self.prefill_gpu.vram_b) * 100
        
        # Check Decode
        decode_limit = self.decode_gpu.vram_b * self.vram_limit_ratio
        if total_mem_b > decode_limit:
             raise ValueError(f"Decode GPU OOM (> {self.vram_limit_ratio*100:.1f}%): Rank needs {total_mem_b/1e9:.2f} GB, but allowed is {decode_limit/1e9:.2f} GB (Total: {self.decode_gpu.vram_b/1e9:.2f} GB)")
        self.decode_vram_util = (total_mem_b / self.decode_gpu.vram_b) * 100

    def start(self, simulator):
        # Trigger the first prefill rank compute
        compute_time = self.T_prefill()
        simulator.add_compute(ComputeJob(f"P_Rank_0_Chunk_0", compute_time))

    def A_prefill(self):
        """
        Return the size (in bytes) of the inter-rank activations for the prefill cluster
        """
        # N * M tokens * Hidden Size * Bytes per param
        return self.N * self.llm.B * self.M * self.llm.H_model

    def KV_handoff(self):
        """
        Return the size of the KV-cache share calculated by a prefill rank to be sent to the
        corresponding decode cluster rank.
        """
        # KV cache for N * M tokens for a single pipeline rank
        return self.llm.KV(self.N, self.M) // self.pp_p

    def T_prefill(self, spec_prefill_enabled: bool = False):
        """
        Calculates prefill time for a single chunk.
        Note: For small batch/chunk sizes, this is dominated by the time to stream 
        all model weights from VRAM (Memory Bound), which explains why it is much larger
        than the transfer time of just the activations.
        """
        N = self.N
        T = self.M  # Chunk size for prefill
        H_model = self.llm.H_model
        L_rank = self.llm.L // self.pp_p

        # 1. Compute Time (FLOPS)
        # Linear FLOPS (MLP + Attention Projections) - dominating term for compute
        flops_linear = 2 * N * T * ((self.llm.W // self.llm.L) * L_rank)

        # Quadratic Attention FLOPS 
        # Correct factor is 4 * N * L * H * T^2 (2 for QK^T + 2 for SV)
        flops_quadratic = 4 * N * L_rank * H_model * (T ** 2)

        # A constant factor representing the reduction in attention complexity
        spec_prefill_multiplier = 0.4 if spec_prefill_enabled else 1.0
        flops_quadratic *= spec_prefill_multiplier
        total_flops = flops_linear + flops_quadratic
        t_compute = total_flops / self.prefill_gpu.flops

        # 2. Memory Time (VRAM BW)
        # We must read ALL weights for every chunk because they don't fit in cache.
        weight_bytes = (self.llm.W  * self.llm.B) // self.pp_p
        kv_bytes = self.llm.KV(N, T) // self.pp_p
        vram_util = 0.80
        t_memory = (weight_bytes + kv_bytes) / (self.prefill_gpu.vram_bw_bps * vram_util)

        # Returns the bottleneck
        return max(t_compute, t_memory)

    def T_decode(self):
        """
        Estimate decode time for one token step on a single rank.
        Dominated by VRAM bandwidth (loading weights + KV cache).
        """
        
        # Compute Time
        flops = 2 * self.N * (self.llm.W // self.pp_d)  # Linear layers
        flops += (4 * self.N * self.T * (self.llm.L // self.pp_d) * self.llm.H_model) // self.pp_d  # Attention
        t_compute = flops / self.decode_gpu.flops
        
        # Memory Time
        weight_bytes = (self.llm.W * self.llm.B // self.pp_d)    # Weights
        kv_bytes = self.llm.KV(self.N, self.T) // self.pp_d      # KV-cache reading
        t_memory = (weight_bytes + kv_bytes) / (self.decode_gpu.vram_bw_bps * 0.8)
        
        return max(t_compute, t_memory)

    def A_decode(self):
        """
        Return the size (in bytes) of the inter-rank activations for the decode cluster (1 token)
        """
        # Activation size for 1 token decode
        return self.N * 1 * self.llm.H_model * self.llm.B

    def on_compute_complete(self, simulator, job):
        """
        Callback triggered when a compute task finishes.
        Delegates to specific prefill or decode handlers based on job name.
        """
        parts = job.name.split("_")
        if "P_Rank" in job.name:
            r_idx = int(parts[2])
            c_idx = int(parts[4])
            self._handle_prefill_compute_complete(simulator, r_idx, c_idx)
        elif "D_Rank" in job.name:
            r_idx = int(parts[2])
            s_idx = int(parts[4])
            self._handle_decode_compute_complete(simulator, r_idx, s_idx)

    def on_data_transfer_complete(self, simulator, batch):
        """
        Callback triggered when a data transfer finishes.

        Returns:
            ComputeJob | None: The next compute job to schedule immediately, or None.
        """
        self.completed_transfers.add(batch.name)

        if "Prefill_Act" in batch.name:
            # Format: Prefill_Act_Rank_X_Chunk_Y
            parts = batch.name.split("_")
            rank_idx = int(parts[3])
            chunk_idx = int(parts[5])
            return self._try_schedule_next_prefill_compute_job(simulator, rank_idx, chunk_idx)

        elif "Handoff" in batch.name:
            # Format: Handoff_Rank_X_Chunk_Y
            parts = batch.name.split("_")
            rank_idx = int(parts[2])
            chunk_idx = int(parts[4])
            return self._try_schedule_next_prefill_compute_job(simulator, rank_idx, chunk_idx)

        elif "Decode_Act" in batch.name:
            return self._handle_decode_transfer(batch.name)
        
        return None

    def _try_schedule_next_prefill_compute_job(self, simulator, rank_idx, chunk_idx):
        handoff_key = f"Handoff_Rank_{rank_idx}_Chunk_{chunk_idx}"
        act_key = f"Prefill_Act_Rank_{rank_idx}_Chunk_{chunk_idx}"
        
        handoff_done = handoff_key in self.completed_transfers
        act_done = act_key in self.completed_transfers
        
        # If not last rank, we need both Activations (next rank) and Handoff (decode) to complete
        if rank_idx + 1 < self.pp_p:
            if handoff_done and act_done:
                comp_time = self.T_prefill()
                return ComputeJob(f"P_Rank_{rank_idx+1}_Chunk_{chunk_idx}", comp_time)
        
        # If last rank, we only have Handoff (no next rank activation)
        else:
            if handoff_done:
                # Last prefill rank finished a chunk
                # Note: This simple counter assumes in-order completion of chunks at the last rank
                self.current_prefill_token_idx += self.M
                if self.current_prefill_token_idx >= self.T and not self.prefill_complete:
                    self.prefill_complete = True
                    # Start Decode Phase
                    return ComputeJob(f"D_Rank_0_Step_0", self.T_decode())
                    
        return None

    def _handle_prefill_compute_complete(self, simulator, r_idx, c_idx):
        # 1. Start Handoff to Decode Cluster (Path 2)
        ib_idx = r_idx % self.num_prefill_ib_cards
        path2 = [
            self.p_gpu_lanes[r_idx],    # GPU -> PLX
            self.p_ib_lanes[ib_idx],    # PLX -> IB Card
            self.p_ib_cables[ib_idx],   # Prefill IB Card -> IB Switch
            self.d_ib_cables[r_idx],    # IB Switch -> Decode IB Card
            self.d_ib_lanes[r_idx],     # Decode IB Card -> PLX
            self.d_gpu_lanes[r_idx]     # PLX -> Decode GPU
        ]
        kv_handoff = self.KV_handoff()
        simulator.add_batch(DataBatch(f"Handoff_Rank_{r_idx}_Chunk_{c_idx}", 
                                      kv_handoff, path2))

        # 2. Start Inter-rank Activation (Path 1)
        if r_idx + 1 < self.pp_p:
            path1 = [
                self.p_gpu_lanes[r_idx],    # GPU -> PLX
                self.p_gpu_lanes[r_idx+1]   # PLX -> Next GPU
            ]
            a_prefill = self.A_prefill()
            simulator.add_batch(DataBatch(f"Prefill_Act_Rank_{r_idx}_Chunk_{c_idx}", 
                                          a_prefill, path1))
        
        # 3. If rank 0, check if we need to start next prefill chunk (Pipeline overlap)
        if r_idx == 0 and (c_idx + 1) * self.M < self.T:
            comp_time = self.T_prefill()
            simulator.add_compute(ComputeJob(f"P_Rank_0_Chunk_{c_idx+1}", comp_time))

    def _handle_decode_compute_complete(self, simulator, r_idx, s_idx):
        # Decode Logic: Path 3 (Inter-server Ethernet)
        if r_idx + 1 < self.pp_d:
            path3 = [
                self.d_gpu_lanes[r_idx],      # GPU -> PLX
                self.d_eth_lanes[r_idx],      # PLX -> Eth Card
                self.d_eth_cables[r_idx],     # Eth Card -> Eth Switch
                self.d_eth_cables[r_idx+1],   # Eth Switch -> Next Eth Card
                self.d_eth_lanes[r_idx+1],    # Eth Card -> PLX
                self.d_gpu_lanes[r_idx+1]     # PLX -> Next GPU
            ]
            simulator.add_batch(DataBatch(f"Decode_Act_Rank_{r_idx}_Step_{s_idx}", self.A_decode(), path3))

    def _handle_decode_transfer(self, batch_name):
        parts = batch_name.split("_")
        rank_idx = int(parts[3])
        step_idx = int(parts[5])
        
        # Move to next decode compute rank
        if rank_idx + 1 < self.pp_d:
            return ComputeJob(f"D_Rank_{rank_idx+1}_Step_{step_idx}", self.T_decode())
        return None

    def calculate_ttds(self, simulator):
        """
        Returns the time to decode start (TTDS).
    
        This method searches the simulator's completed compute jobs for those whose names start with "D_Rank" (decode jobs).
        It sorts these jobs by their end time, then returns the start time of the job that finished first.
        This is useful for measuring when the decode phase began for the earliest completed decode job.
        If no decode jobs are found, returns 0.0.
        """
        decode_jobs = [job for job in simulator.completed_compute if job.name.startswith("D_Rank")]
        if not decode_jobs:
            return 0.0
        decode_jobs_sorted = sorted(decode_jobs, key=lambda j: j.end_time)
        return decode_jobs_sorted[0].start_time

    def calculate_tpot(self, simulator):
        decode_jobs = [job for job in simulator.completed_compute if job.name.startswith("D_Rank")]
        if not decode_jobs:
            return 0.0
        decode_jobs_sorted = sorted(decode_jobs, key=lambda j: j.end_time)
        first_decode_job = decode_jobs_sorted[0]
        last_decode_job = decode_jobs_sorted[-1]
        return last_decode_job.end_time - first_decode_job.start_time
