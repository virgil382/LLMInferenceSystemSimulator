from simulator import DataBatch, ComputeJob, SimulatedSystem, CommChannel, GPU, LLM
from collections import defaultdict

class DisaggregatedPDSystem2D(SimulatedSystem):

    def _init_tp_sync_gate(self):
        # Tracks how many TP jobs have finished for each (pp, chunk)
        self.tp_sync_gate = defaultdict(int)
        self.tp_sync_total = self.tp_degree

    def __init__(self, llm: LLM, prefill_gpu: GPU, decode_gpu: GPU, pp_degree: int, tp_degree: int = 1,
                 num_prefill_ib_cards: int = 1, N: int = 16, T: int = 4096, M: int = 128, vram_limit_ratio: float = 0.75):
        self.llm = llm
        self.prefill_gpu = prefill_gpu
        self.decode_gpu = decode_gpu
        self.pp_p = pp_degree
        self.pp_d = pp_degree
        self.tp_degree = tp_degree
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
        # 2D array: p_gpu_lanes[pp][tp] for 2D prefill cluster
        self.p_gpu_lanes = [[CommChannel("PCIe Gen5 x16") for _ in range(self.tp_degree)] for _ in range(self.pp_p)]
        self.d_gpu_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.pp_d)]
        self.p_ib_lanes = [CommChannel("PCIe Gen5 x16") for _ in range(self.num_prefill_ib_cards)]
        self.p_ib_cables = [CommChannel("Infiniband NDR") for _ in range(self.num_prefill_ib_cards)]
        self.d_ib_cables = [CommChannel("Infiniband NDR") for _ in range(self.pp_d)]
        self.d_eth_lanes = [CommChannel("PCIe Gen4") for _ in range(self.pp_d)]
        self.d_eth_cables = [CommChannel("Ethernet 100G") for _ in range(self.pp_d)]

        # VRAM Validation (Step 1: 2D Parallelism)
        # 1. Weights
        # Parameters are sharded across both PP and TP: (W // L) * (L // pp_degree) // tp_degree
        params_per_rank = ((self.llm.W // self.llm.L) * (self.llm.L // self.pp_p)) // self.tp_degree
        weight_bytes = params_per_rank * self.llm.B

        # 2. KV Cache (at full context T), sharded by both PP and TP
        kv_bytes = (self.llm.KV(self.N, self.T) // self.pp_p) // self.tp_degree

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
        Calculates prefill time for a single chunk, including TP All-Reduce sync overhead.
        """
        N = self.N
        T = self.M  # Chunk size for prefill
        H_model = self.llm.H_model
        L_rank = self.llm.L // self.pp_p
        TP = self.tp_degree

        # 1. Compute Time (FLOPS)
        # Linear FLOPS (MLP + Attention Projections) - dominating term for compute
        flops_linear = 2 * N * T * ((self.llm.W // self.llm.L) * L_rank) // TP

        # Quadratic Attention FLOPS 
        # Correct factor is 4 * N * L * H * T^2 (2 for QK^T + 2 for SV)
        flops_quadratic = 4 * N * L_rank * H_model * (T ** 2) // TP

        # A constant factor representing the reduction in attention complexity
        spec_prefill_multiplier = 0.4 if spec_prefill_enabled else 1.0
        flops_quadratic *= spec_prefill_multiplier
        total_flops = flops_linear + flops_quadratic
        t_compute = total_flops / self.prefill_gpu.flops

        # 2. Memory Time (VRAM BW)
        # We must read ALL weights for every chunk because they don't fit in cache.
        weight_bytes = ((self.llm.W // self.llm.L) * L_rank) * self.llm.B // TP
        kv_bytes = (self.llm.KV(N, T) // self.pp_p) // TP
        vram_util = 0.80
        t_memory = (weight_bytes + kv_bytes) / (self.prefill_gpu.vram_bw_bps * vram_util)

        # 3. TP All-Reduce Sync Overhead
        # Each layer: 2x All-Reduce (after Attention, after MLP)
        # All-Reduce size: activation size per TP GPU
        num_layers = L_rank
        act_size = N * T * H_model * self.llm.B // TP
        # Assume All-Reduce is bandwidth-bound, using NVLink (local server)
        # For now, use PCIe Gen5 x16 as a placeholder (update if NVLink available)
        ar_channel = CommChannel("PCIe Gen5 x16")
        ar_bw = ar_channel.bandwidth_bps
        # All-Reduce time for one op: (S * (TP-1)/TP) / BW (ring all-reduce)
        ar_time_per = (act_size * (TP - 1) / TP) / ar_bw if TP > 1 else 0
        tp_sync_time = 2 * num_layers * ar_time_per

        # Returns the bottleneck including TP sync
        return max(t_compute, t_memory) + tp_sync_time

    def T_decode(self):
        """
        Estimate decode time for one token step on a single rank.
        Dominated by VRAM bandwidth (loading weights + KV cache).
        """
        # Weights per rank
        weight_bytes = ((self.llm.W // self.llm.L) * (self.llm.L // self.pp_d)) * self.llm.B
        # KV cache estimate (full context for worst case)
        kv_bytes = self.llm.KV(self.N, self.T) // self.pp_d
        
        # Compute Time
        flops = 2 * self.N * (self.llm.W // self.pp_d)  # Linear layers
        flops += 2 * self.N * self.T * (self.llm.H_model // self.pp_d) * self.llm.H_model # Attention
        t_compute = flops / self.decode_gpu.flops
        
        # Memory Time
        t_memory = (weight_bytes + kv_bytes) / (self.decode_gpu.vram_bw_bps * 0.8)
        
        return max(t_compute, t_memory)

    def A_decode(self):
        """
        Return the size (in bytes) of the inter-rank activations for the decode cluster (1 token)
        """
        # Activation size for 1 token decode
        return self.N * 1 * self.llm.H_model * self.llm.B

    def start(self, simulator):
        # Initialize TP sync tracker
        self._init_tp_sync_gate()
        # Trigger the first prefill jobs for all TP shards in PP[0]
        for t in range(self.tp_degree):
            compute_time = self.T_prefill()
            simulator.add_compute(ComputeJob(f"P_Rank_PP0_TP{t}_Chunk0", compute_time))

    def on_compute_complete(self, simulator, job):
        """
        Callback triggered when a compute task finishes.
        2D Parallelism Logic:
        - Each pipeline stage (PP) is a group of TP GPUs working in lockstep.
        - Each compute job is for a specific PP stage, TP shard, and chunk: P_Rank_PP[p]_TP[t]_Chunk[c].
        - When a TP compute job finishes, we mark it as done in tp_sync_gate for (pp, chunk).
        - Only when all TP jobs for a (pp, chunk) are done, we trigger the All-Reduce (AR) for each TP GPU in that group.
        - Each AR is modeled as a DataBatch on the corresponding p_gpu_lanes[pp][tp].
        - After AR, on_data_transfer_complete will handle PP forwarding and handoff.
        - The sync gate is reset after all TP jobs for a (pp, chunk) are processed.
        """
        parts = job.name.split("_")
        if job.name.startswith("P_Rank_PP"):
            # Parse indices from job name: P_Rank_PP[p]_TP[t]_Chunk[c]
            p_idx = int(parts[2][2:])  # Pipeline stage index
            t_idx = int(parts[3][2:])  # Tensor parallel shard index
            c_idx = int(parts[4][5:])  # Chunk index
            key = (p_idx, c_idx)
            # Mark this TP job as done for this (pp, chunk)
            if not hasattr(self, 'tp_sync_gate'):
                self._init_tp_sync_gate()
            self.tp_sync_gate[key] += 1
            # Wait for all TP jobs in this group to finish before moving forward
            if self.tp_sync_gate[key] == self.tp_sync_total:
                # All TP jobs for this (pp, chunk) are done: trigger All-Reduce for each TP GPU
                for t in range(self.tp_degree):
                    ar_name = f"TP_AR_PP{p_idx}_Layer0_Chunk{c_idx}_TP{t}"
                    # Each AR is modeled as a DataBatch on the corresponding GPU lane
                    ar_path = [self.p_gpu_lanes[p_idx][t]]
                    ar_size = self.N * self.M * self.llm.H_model * self.llm.B // self.tp_degree
                    simulator.add_batch(DataBatch(ar_name, ar_size, ar_path))
                # Reset the sync gate for this (pp, chunk) after use
                self.tp_sync_gate[key] = 0
        elif job.name.startswith("D_Rank"):
            r_idx = int(parts[2])
            s_idx = int(parts[4])
            self._handle_decode_compute_complete(simulator, r_idx, s_idx)

    def on_data_transfer_complete(self, simulator, batch):
        """
        Callback triggered when a data transfer finishes.
        Handles 2D parallelism: after All-Reduce, trigger PP activation and Handoff.
        """
        self.completed_transfers.add(batch.name)

        if batch.name.startswith("TP_AR_PP"):
            # All-Reduce complete for a TP GPU in a PP stage
            parts = batch.name.split("_")
            p_idx = int(parts[2][2:])
            c_idx = int(parts[4][5:])
            t_idx = int(parts[5][2:])
            # After All-Reduce, send PP activation and Handoff for this TP GPU
            # PP Activation to next PP stage (if not last)
            if p_idx + 1 < self.pp_p:
                pp_act_name = f"PP_Act_FromP{p_idx}_ToP{p_idx+1}_TP{t_idx}_Chunk{c_idx}"
                pp_act_path = [self.p_gpu_lanes[p_idx][t_idx], self.p_gpu_lanes[p_idx+1][t_idx]]
                pp_act_size = self.N * self.M * self.llm.H_model * self.llm.B // self.tp_degree
                simulator.add_batch(DataBatch(pp_act_name, pp_act_size, pp_act_path))
            # Handoff to decode cluster
            handoff_name = f"Handoff_PP{p_idx}_TP{t_idx}_Chunk{c_idx}"
            ib_idx = (p_idx * self.tp_degree + t_idx) % self.num_prefill_ib_cards
            handoff_path = [self.p_gpu_lanes[p_idx][t_idx], self.p_ib_lanes[ib_idx], self.p_ib_cables[ib_idx], self.d_ib_cables[p_idx], self.d_gpu_lanes[p_idx]]
            handoff_size = (self.llm.KV(self.N, self.M) // self.pp_p) // self.tp_degree
            simulator.add_batch(DataBatch(handoff_name, handoff_size, handoff_path))
            # If this is TP0 and not last PP, trigger next PP compute for this chunk
            if t_idx == 0 and p_idx + 1 < self.pp_p:
                for t in range(self.tp_degree):
                    compute_time = self.T_prefill()
                    simulator.add_compute(ComputeJob(f"P_Rank_PP{p_idx+1}_TP{t}_Chunk{c_idx}", compute_time))
            # If this is TP0 and last PP, check if next chunk should start
            if t_idx == 0 and p_idx + 1 == self.pp_p:
                if (c_idx + 1) * self.M < self.T:
                    for t in range(self.tp_degree):
                        compute_time = self.T_prefill()
                        simulator.add_compute(ComputeJob(f"P_Rank_PP0_TP{t}_Chunk{c_idx+1}", compute_time))
                else:
                    # If last chunk, start decode phase (TP0 only)
                    simulator.add_compute(ComputeJob(f"D_Rank_0_Step_0", self.T_decode()))
        elif batch.name.startswith("PP_Act_FromP"):
            # No-op: just a transfer, no compute to trigger
            return None
        elif batch.name.startswith("Handoff_PP"):
            # No-op: just a transfer, no compute to trigger
            return None
        elif batch.name.startswith("Decode_Act"):
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
        # For legacy _handle_prefill_compute_complete, assume t_idx=0 (single TP) for backward compatibility
        t_idx = 0
        path2 = [
            self.p_gpu_lanes[r_idx][t_idx],    # GPU -> PLX
            self.p_ib_lanes[ib_idx],    # PLX -> IB Card
            self.p_ib_cables[ib_idx],   # Prefill IB Card -> IB Switch
            self.d_ib_cables[r_idx],    # IB Switch -> Decode IB Card
            self.d_gpu_lanes[r_idx]     # IB Card -> Decode GPU
        ]
        kv_handoff = self.KV_handoff()
        simulator.add_batch(DataBatch(f"Handoff_Rank_{r_idx}_Chunk_{c_idx}", 
                                      kv_handoff, path2))

        # 2. Start Inter-rank Activation (Path 1)
        if r_idx + 1 < self.pp_p:
            # For legacy _handle_prefill_compute_complete, assume t_idx=0 (single TP)
            path1 = [
                self.p_gpu_lanes[r_idx][t_idx],    # GPU -> PLX
                self.p_gpu_lanes[r_idx+1][t_idx]   # PLX -> Next GPU
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

    def calculate_ttft(self, simulator):
        decode_jobs = [job for job in simulator.completed_compute if job.name.startswith("D_Rank")]
        if not decode_jobs:
            return None
        decode_jobs_sorted = sorted(decode_jobs, key=lambda j: j.end_time)
        return decode_jobs_sorted[0].start_time

    def calculate_tpot(self, simulator):
        decode_jobs = [job for job in simulator.completed_compute if job.name.startswith("D_Rank")]
        if not decode_jobs:
            return None
        decode_jobs_sorted = sorted(decode_jobs, key=lambda j: j.end_time)
        first_decode_job = decode_jobs_sorted[0]
        last_decode_job = decode_jobs_sorted[-1]
        return last_decode_job.end_time - first_decode_job.start_time
