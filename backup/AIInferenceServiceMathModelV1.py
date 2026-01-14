# Notation for Inference Service Math Model
#
# This section explains the notation used for modeling various aspects of an inference service, especially for transformer-based models and hardware characteristics:
#

# Notation (Model and Hardware Parameters):
#
# - 𝑊: Total number of model weights (parameters)
# - 𝐿: Number of transformer layers
# - 𝐻_model: Model hidden size (dimension of hidden states)
# - 𝐵: Bytes per element (e.g., 2 for FP16, 1 for FP8)
# - 𝐻_kv: Number of key-value heads (for Grouped Query Attention, GQA)
# - 𝐻_q: Number of query heads (for GQA)
# - 𝐷_latent: Latent dimension (for Mixture-of-Latents Attention, MLA)
# - 𝑇: Context size (sequence length, e.g., 8K tokens)
# - 𝐶: Communication channel bandwidth (e.g., 12 GB/s for PCIe Gen4)
# - 𝜏: Communication channel latency (e.g., 20 μs for PCIe Gen4)
# - 𝑁: Batch size
# - 𝑅: GPU FLOPS/sec (FP16 Tensor)
# - 𝐵𝑊_mem: GPU VRAM bandwidth



class CommChannel:
    """
    Class representing a communication channel and its key attributes.
    Bandwidth is in bytes/sec, latency is in seconds.
    For attributes with a range, the main accessor returns the average, and there are methods for low and high values.
    """
    _comm_db: dict[str, dict[str, tuple[float, float]]] = {
        "NVLink": {"bw_gbps": (300, 400), "lat_us": (1, 2)},
        "PCIe Gen4": {"bw_gbps": (12, 15), "lat_us": (18, 25)},
        "PCI Express 5.0 x16": {"bw_gbps": (55, 60), "lat_us": (10, 15)},

        # RTX 6000 Ada (Gen4) on a Gen5 Switched Fabric
        "PCIe G4-on-G5-Switch": {
            "bw_gbps": (28, 30), # Maxing out the Gen4 physical limit of the GPU
            "lat_us": (12, 18)   # Lower latency due to reduced switch contention
        },        

        "Infiniband HDR": {"bw_gbps": (22, 24), "lat_us": (5, 8)},
        "Infiniband NDR": {"bw_gbps": (45, 48), "lat_us": (2, 5)},
        "Ethernet 100G": {"bw_gbps": (10, 11), "lat_us": (10, 20)},
        "Ethernet 25-40G": {"bw_gbps": (2.5, 4), "lat_us": (50, 100)},
        "Cross-DC (WAN)": {"bw_gbps": (0, 1), "lat_us": (10000, 100000)},
    }

    def __init__(self, name: str) -> None:
        if name not in self._comm_db:
            raise ValueError(f"Unknown Comm Link: {name}")
        self._name: str = name
        self._attrs: dict[str, tuple[float, float]] = self._comm_db[name]

    @property
    def name(self) -> str:
        return self._name

    # Bandwidth (bytes/sec)
    @property
    def bandwidth_bps(self) -> float:
        low, high = self._attrs["bw_gbps"]
        return ((low + high) / 2) * 1e9

    def bandwidth_bps_low(self) -> float:
        return self._attrs["bw_gbps"][0] * 1e9

    def bandwidth_bps_high(self) -> float:
        return self._attrs["bw_gbps"][1] * 1e9

    # Latency (seconds)
    @property
    def latency_s(self) -> float:
        low, high = self._attrs["lat_us"]
        return ((low + high) / 2) * 1e-6

    def latency_s_low(self) -> float:
        return self._attrs["lat_us"][0] * 1e-6

    def latency_s_high(self) -> float:
        return self._attrs["lat_us"][1] * 1e-6

    @classmethod
    def available_channels(cls) -> list[str]:
        """Return a list of available comm channel names."""
        return list(cls._comm_db.keys())

# Example usage:
# channel = CommChannel("PCIe Gen4")
# print(channel.bandwidth_bps, channel.bandwidth_bps_low(), channel.bandwidth_bps_high())
# print(channel.latency_s, channel.latency_s_low(), channel.latency_s_high())

class GPU:
    """
    Class representing a GPU and its key attributes for inference modeling, including price.
    All values are normalized to bytes (B) and FLOPS (not GB/s or TFLOPS).
    """
    def __init__(self, name: str, vram_b: float, vram_bw_bps: float, flops: float, comm_channel: CommChannel, price: float):
        self.name = name
        self.vram_b = vram_b
        self.vram_bw_bps = vram_bw_bps
        self.flops = flops
        self.comm_channel = comm_channel
        self.price = price

    @classmethod
    def from_name(cls, name: str) -> 'GPU':
        """Factory method to build GPU from a GPU name."""
        gpus = {
            "NVIDIA RTX PRO 6000": dict(vram_b=96 * 1024**3, vram_bw_bps=1694.5 * 1e9, flops=310 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=8346),
            "NVIDIA H100": dict(vram_b=80 * 1024**3, vram_bw_bps=3350 * 1e9, flops=989 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=29999),
            "NVIDIA A100-80GB": dict(vram_b=80 * 1024**3, vram_bw_bps=2000 * 1e9, flops=312 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=23300),
            "NVIDIA L40": dict(vram_b=48 * 1024**3, vram_bw_bps=864 * 1e9, flops=181 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=11225),
            "NVIDIA H200": dict(vram_b=141 * 1024**3, vram_bw_bps=4800 * 1e9, flops=989 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=30250),
            "AMD Instinct MI300A": dict(vram_b=128 * 1024**3, vram_bw_bps=5300 * 1e9, flops=981 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=29396),
            "AMD Instinct MI300X": dict(vram_b=192 * 1024**3, vram_bw_bps=5300 * 1e9, flops=1307 * 1e12, comm_channel=CommChannel("PCI Express 5.0 x16"), price=32000),
        }
        if name not in gpus:
            raise ValueError(f"Unknown GPU: {name}")
        return cls(name=name, **gpus[name])  # type: ignore

    @classmethod
    def available_gpus(cls) -> list[str]:
        """Return a list of available GPU names."""
        return [
            "NVIDIA RTX PRO 6000",
            "NVIDIA H100",
            "NVIDIA A100-80GB",
            "NVIDIA L40",
            "NVIDIA H200",
            "AMD Instinct MI300A",
            "AMD Instinct MI300X",
        ]

# Example usage:
# gpu = GPU("NVIDIA H100")
# print(gpu.vram_b, gpu.vram_bw_bps, gpu.flops, gpu.price)


class Model:
    """
    Provides attributes for several FP16 models (e.g., LLaMA-3-70B, GPT-OSS-20B, etc.).
    Returns model parameters and configuration.
    """
    def __init__(self,
                 name: str,
                 W: int,
                 L: int,
                 H_model: int,
                 B: int,
                 H_kv: int | None = None,
                 H_q: int | None = None,
                 D_latent: int | None = None,
                 attention_type: str | None = None,
                 moe: bool = False):
        self.name = name
        self.W = W
        self.L = L
        self.H_model = H_model
        self.B = B
        self.H_kv = H_kv
        self.H_q = H_q
        self.D_latent = D_latent
        self.attention_type = attention_type
        self.moe = moe

    @classmethod
    def from_name(cls, name: str) -> 'Model':
        """Factory method to build ModelAttributes from a model name."""
        models = {
            "LLaMA-3.1-70B": dict(W=70554470400, L=80, H_model=8192, B=2, H_kv=8, H_q=64, D_latent=None, attention_type="GQA", moe=False),
            "GPT-OSS-20B": dict(W=20910000000, L=24, H_model=2880, B=2, H_kv=8, H_q=64, D_latent=None, attention_type="GQA", moe=True),
            "DeepSeek-V3": dict(W=671000000000, L=61, H_model=7168, B=2, H_kv=1, H_q=128, D_latent=512, attention_type="MLA", moe=True),
            "Mistral-Small-24B": dict(W=24000000000, L=40, H_model=5120, B=2, H_kv=8, H_q=32, D_latent=None, attention_type="GQA", moe=False),
            "Qwen2.5-72B": dict(W=72700000000, L=80, H_model=8192, B=2, H_kv=8, H_q=64, D_latent=None, attention_type="GQA", moe=False),
            "LLaMA-3.1-405B": dict(W=405000000000, L=126, H_model=16384, B=2, H_kv=8, H_q=128, D_latent=None, attention_type="GQA", moe=False),
            "Mixtral-8x22B": dict(W=141000000000, L=56, H_model=6144, B=2, H_kv=8, H_q=48, D_latent=None, attention_type="GQA", moe=True),
        }
        if name not in models:
            raise ValueError(f"Unknown model: {name}")
        return cls(name=name, **models[name])

    @classmethod
    def available_models(cls) -> list[str]:
        """Return a list of available model names."""
        return [
            "LLaMA-3.1-70B",
            "GPT-OSS-20B",
            "DeepSeek-V3",
            "Mistral-Small-24B",
            "Qwen2.5-72B",
            "LLaMA-3.1-405B",
            "Mixtral-8x22B",
        ]

    def feasible_pp_prefill_configs(self, gpu) -> set[tuple[int, int, int]]:
        """
        Returns a set of tuples (num_ranks, batch_size, context_size) specifying feasible pipeline parallel (PP) prefill configurations
        for this model instance and the given GPU. Each tuple can be used to configure a pair of PP pipelines (prefill and decode) for inference.
        Only includes configs where weights and KV-cache fit in no more than 75% of the GPU's VRAM.
        The returned set is not exhaustive, but covers common practical configurations.
        """
        feasible = set()
        vram_limit = 0.70 * gpu.vram_b  # No more than 70% of VRAM may be used.
        pp_options = [r for r in [1, 2, 4, 8, 16] if self.L % r == 0]
        batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
        context_options = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        for num_ranks in pp_options:
            weights_per_rank = self.W // num_ranks
            for batch_size in batch_options:
                for context_size in context_options:
                    try:
                        kv = self.KV(batch_size, context_size)
                        total_mem = weights_per_rank * self.B + kv
                        if kv > 0 and weights_per_rank > 0 and total_mem <= vram_limit:
                            feasible.add((num_ranks, batch_size, context_size))
                    except Exception:
                        continue
        return feasible

    def KV(self, N: int, T: int) -> int:
        """
        Calculate the KV-cache size (in bytes) for a batch of size N and context size T.
        - For GQA: N * T * L * H_kv * (H_model / H_q) * 2 * B
        - For MHA: N * T * L * H_model * 2 * B
        - For MLA: N * T * L * H_kv * D_latent * 2 * B
        (2 for K and V, B is bytes per element)
        """
        attn_type = self.attention_type
        # Determine required variables and their types for each attention type
        if attn_type == "MLA":
            required = {
                "L": (self.L, int),
                "H_kv": (self.H_kv, int),
                "D_latent": (self.D_latent, int),
                "B": (self.B, int),
                "N": (N, int),
                "T": (T, int)
            }
        elif attn_type == "MHA":
            required = {
                "L": (self.L, int),
                "H_model": (self.H_model, int),
                "B": (self.B, int),
                "N": (N, int),
                "T": (T, int)
            }
        else:  # GQA (default)
            required = {
                "L": (self.L, int),
                "H_kv": (self.H_kv, int),
                "H_q": (self.H_q, float),
                "H_model": (self.H_model, float),
                "B": (self.B, int),
                "N": (N, int),
                "T": (T, int)
            }
        # Check for None and cast in one loop, assign to locals with correct type
        local_vars = {}
        for k, (v, typ) in required.items():
            if v is None:
                raise ValueError(f"Model attribute '{k}' is None; cannot compute KV-cache size.")
            try:
                local_vars[k] = typ(v)
            except Exception as e:
                raise TypeError(f"Model attribute '{k}' could not be cast to {typ.__name__}: {e}")
        # Unpack variables for calculation
        L = local_vars["L"]
        B = local_vars["B"]
        N = local_vars["N"]
        T = local_vars["T"]
        if attn_type == "MLA":
            H_kv = local_vars["H_kv"]
            D_latent = local_vars["D_latent"]
            return int(N * T * L * H_kv * D_latent * 2 * B)
        elif attn_type == "MHA":
            H_model = local_vars["H_model"]
            return int(N * T * L * H_model * 2 * B)
        else:  # GQA
            H_kv = local_vars["H_kv"]
            H_q = local_vars["H_q"]
            H_model = local_vars["H_model"]
            return int(N * T * L * H_kv * (H_model / H_q) * 2 * B)

class PrefillPPRank:
    """
    Represents a pipeline parallel rank in a PPInferenceNode.
    Each PrefillPPRank uses a single GPU and communicates with the next via a CommLink.
    """
    def __init__(self, 
                 gpu: 'GPU',
                 inter_rank_comm_channel: 'CommChannel',
                 model: Model,
                 L_rank: int
                 ) -> None:
        """
        Initialize a pipeline parallel rank (PrefillPPRank).

        Parameters:
            gpu (GPU):
                The GPU assigned to this pipeline rank. Used for compute and memory bandwidth modeling.
            inter_rank_comm_channel (CommChannel):
                The communication channel to the next rank. Used for modeling inter-rank data transfer.
            model (Model):
                The model configuration object. Provides model-wide parameters (weights, hidden size, etc).
            L_rank (int):
                Number of transformer layers assigned to this rank. Determines the fraction of the model handled by this rank.
        """
        self.gpu: GPU = gpu
        self.inter_rank_comm_channel: CommChannel = inter_rank_comm_channel
        self.L_rank: int = L_rank
        self.H_model: int = model.H_model
        self.B: int = model.B
        self.model: Model = model

    @property
    def W_rank(self) -> int:
        """
        Calculates the number of weights assigned to this PrefillPPRank.
        Proportional to the number of layers in this rank.
        """
        total_weights = self.model.W
        total_layers = self.model.L
        if total_weights is None or total_layers is None or total_layers == 0:
            return 0
        # Distribute weights proportionally to the number of layers in this rank
        return int(total_weights * self.L_rank / total_layers)

    def T_prefill(self, N: int, T: int) -> float:
        """
        Corrected Prefill Time with explicit H^2 scaling for projections
        and T^2 scaling for the attention matrix.
        """
        H = self.H_model
        L_rank = self.L_rank
        
        # 1. Compute Time (FLOPS)
        # Projections: Q, K, V, and O projections + MLP (up/down/gate)
        # Most models approximate total linear FLOPS as 2 * N * T * W_per_rank
        # But to be explicit about the H^2 in the attention block projections:
        # flops_attn_projections = L_rank * (8 * N * T * H**2) 
        
        # Linear FLOPS (including MLP and Attention Projections)
        flops_linear = 2 * N * T * self.W_rank
        
        # Quadratic Attention FLOPS (The QK^T and Score * V part)
        # Standard formula: 2 * N * L_rank * H * T^2
        flops_quadratic = 2 * N * L_rank * H * (T**2)
        
        total_flops = flops_linear + flops_quadratic
        t_compute = total_flops / self.gpu.flops

        # 2. Memory Time (VRAM BW)
        # Must read weights + write the resulting KV cache
        weight_bytes = self.W_rank * self.B
        kv_bytes = self.KV_handoff(N, T)
        
        vram_util = 0.80
        t_memory = (weight_bytes + kv_bytes) / (self.gpu.vram_bw_bps * vram_util)

        return max(t_compute, t_memory)


    def A_prefill(self, N: int = 1, T: int = 1) -> int:
        """
        Calculates the size of the hidden states (in bytes) that this PrefillPPRank must transfer to the next rank during prefill.

        Parameters:
            batch_size (int): Number of sequences in the batch.
            context_length (int): Number of tokens in each sequence (context size).

        Returns:
            int: Number of bytes to transfer to the next rank during prefill.
        """
        bytes_to_transfer: int = N * T * self.H_model * self.B
        return bytes_to_transfer


    def X_prefill(self, N: int = 1, T: int = 1, contending_data_volume : int = 0) -> float:
        """
        Calculates the transfer time (seconds) for bytes transferred to the next PrefillPPRank during prefill.
        Assumes transfer of hidden states for the batch and context.

        Parameters:
            N (int): Batch size (number of sequences).
            T (int): Context size (number of tokens per sequence).
            contending_data_volume (int): Additional volume of data (in bytes) contending for the same communication channel during transfer.

        Returns:
            float: Total transfer time in seconds, including channel latency and bandwidth effects.
        """
        bytes_to_transfer: int = self.A_prefill(N, T)

        bandwidth = self.inter_rank_comm_channel.bandwidth_bps
        if bandwidth is None or bandwidth == 0:
            raise ValueError("CommLink bandwidth is None or zero.")
        
        return self.inter_rank_comm_channel.latency_s + ((bytes_to_transfer + contending_data_volume) / float(bandwidth))

    def KV_handoff(self, N: int, T: int) -> int:
        """
        Calculates the KV-cache size (in bytes) for this PrefillPPRank, proportional to its number of layers.
        This is the amount of KV-cache that must be handed off to the decode rank.

        Parameters:
            N (int): Batch size (number of sequences).
            T (int): Context size (number of tokens per sequence).

        Returns:
            int: Number of bytes of KV-cache for this rank.
        """
        # Calculate the fraction of layers this rank is responsible for
        total_layers = self.model.L
        if total_layers == 0:
            return 0
        # Use the model's KV_req for the full model, then scale by this rank's layer fraction
        full_kv = self.model.KV(N, T)
        return int(full_kv * self.L_rank / total_layers)
    
    def X_handoff(self, N: int, T: int, handoff_comm_channel: CommChannel, handoff_comm_channel_contending_data_volume : int = 0) -> float:
        """
        Calculate the time to handoff the KV cache from this PrefillPPRank to the corresponding decode PrefillPPRank.

        N: batch size
        T: context length
        comm_channel: The CommChannel representing the connection between prefill and decode ranks.
        handoff_comm_channel_contending_data_volume: Volume of data that also must be transmitted via the handoff communication channel.
        """
        kv_bytes = self.KV_handoff(N, T)
        
        bandwidth = handoff_comm_channel.bandwidth_bps
        latency = handoff_comm_channel.latency_s
        
        return ((kv_bytes + handoff_comm_channel_contending_data_volume) / (bandwidth)) + latency

    def T_decode(self, N: int, T: int) -> float:
        """
        Corrected Decode Time:
        Time = max(Compute_Time, Memory_Access_Time)
        In decoding, we are almost always Memory Bound by (Weights + KV-Cache).
        """
        # 1. Compute time (FLOPS)
        # Using 2 * W as a heuristic for forward pass FLOPS per token
        flops_per_token = (2 * self.W_rank) * N
        t_compute = flops_per_token / self.gpu.flops

        # 2. Memory bandwidth time (The 'Memory Wall')
        # We must read ALL weights + the current KV cache from VRAM
        weights_per_rank = self.W_rank * self.model.B
        kv_cache_bytes = self.KV_handoff(N, T)

        vram_bw_utilization_factor = 0.75  # Assume 75% of VRAM bandwidth is usable

        t_memory = (weights_per_rank + kv_cache_bytes) / (self.gpu.vram_bw_bps * vram_bw_utilization_factor)
        
        # Decoding is usually memory-bound, so we take the max
        return max(t_compute, t_memory)

    def A_decode(self, N: int = 1) -> int:
        """
        Calculates the size of the hidden states (in bytes) that this PrefillPPRank must transfer to the next rank during decode (per token).

        Parameters:
            N (int): Batch size (number of sequences).

        Returns:
            int: Number of bytes to transfer to the next rank during decode.
        """
        return N * self.H_model * self.B

    def X_decode(self, N: int = 1) -> float:
        """
        Calculates the transfer time (sec) for bytes transferred to the next PrefillPPRank during decode (T=1).
        Assumes transfer of hidden states for the batch and a single token.
        """
        bytes_to_transfer = self.A_decode(N)
        # Add the base latency (tau) to the transfer time
        return self.inter_rank_comm_channel.latency_s + (bytes_to_transfer / self.inter_rank_comm_channel.bandwidth_bps)


class PPPipeline:
    """
    Represents an LLM node configured with pipeline parallelism.
    Consists of PPRanks with equal layer allocation.
    """
    def __init__(self, ranks: list['PrefillPPRank']) -> None:
        self.ranks: list[PrefillPPRank] = ranks

    @classmethod
    def build_ranks(cls, model: Model, gpu_list: list['GPU'], inter_rank_comm_channel_name: str) -> list['PrefillPPRank']:
        """
        Construct a list of PrefillPPRank objects for pipeline parallelism.

        Args:
            model (ModelAttributes): The model configuration object.
            gpu_list (list[GPU]): List of GPU objects, one per pipeline rank.
            inter_rank_comm_channel_name (str): Name of the communication channel (e.g., 'PCIe Gen4') between the ranks.

        Returns:
            list[PrefillPPRank]: List of PrefillPPRank objects, each representing a pipeline stage.  

        Logic:
            - Each PrefillPPRank is assigned a GPU and a CommChannel.
            - Each rank is assigned an equal number of layers (layers_per_rank).
            - Model-dependent parameters (hidden_size, bytes_per_elem, num_layers) are extracted from the model.
        """
        num_layers = model.L
        num_ranks = len(gpu_list)
        layers_per_rank: int = num_layers // num_ranks
        hidden_size = model.H_model
        bytes_per_elem = model.B

        return [
            PrefillPPRank(gpu_list[i], CommChannel(inter_rank_comm_channel_name), model, layers_per_rank)
            for i in range(num_ranks)
        ]

    def get_rank(self, idx: int) -> PrefillPPRank:
        return self.ranks[idx]

    def X_handoff(self, N: int, T: int, kv_handoff_comm_channel: CommChannel) -> float:
        """
        Calculate the time to handoff the KV cache from the Prefill PPInferenceNode to the Decode PPInferenceNode
        assuming each PP rank in the Prefill node transfers its local KV cache to its corresponding rank in the 
        Decode node.

        N: batch size
        T: context length
        kv_handoff_comm_channel: The CommChannel representing the connection between prefill and decode clusters.
        """
        # Find the rank with the most data (usually equal if layers are balanced)
        max_kv_bytes_per_rank = max(rank.KV_handoff(N, T) for rank in self.ranks)
        
        # Time is determined by the slowest single transfer
        bandwidth = kv_handoff_comm_channel.bandwidth_bps
        latency = kv_handoff_comm_channel.latency_s
        
        return (max_kv_bytes_per_rank / bandwidth) + latency


    def TTFT_chunking_streaming(self, N: int, T: int, M: int, kv_handoff_comm_channel: CommChannel) -> float:
        """
        Calculates TTFT including the time to handoff the KV cache from the prefill rank to the decode rank.  The assumption is that
        each Prefill rank has a separate kv_handoff_comm_channel to its corresponding decode rank, and the KV cache transfer occurs
        in parallel.

        N: batch size
        T: context length (sequence length)
        M: chunk size (microbatch size)
        kv_handoff_comm_channel: Communication channel between Prefill and Decode ranks
        """
        num_chunks = (T + M - 1) // M
        if num_chunks == 0: return 0.0

        # 1. Latency for a SINGLE chunk (M tokens)
        stage_latencies = []
        chunk_handoff_times = []
        
        for i, rank in enumerate(self.ranks):
            t_comp = rank.T_prefill(N=N, T=M)
            t_comm = rank.X_prefill(N=N, T=M) if i < len(self.ranks) - 1 else 0.0
            stage_latencies.append(t_comp + t_comm)
            
            # Calculate time to handoff just THIS chunk (M tokens)
            chunk_kv_bytes = rank.KV_handoff(N, M)
            t_hando_chunk = (chunk_kv_bytes / kv_handoff_comm_channel.bandwidth_bps) + kv_handoff_comm_channel.latency_s
            chunk_handoff_times.append(t_hando_chunk)

        # 2. Pipeline Fill (First chunk passes through all ranks)
        t_fill = sum(stage_latencies)

        # 3. Pipeline Steady State (Remaining chunks)
        # The bottleneck is now the MAX of (Stage Latency) or (KV Handoff Latency)
        # because they happen in parallel.
        t_handoff_bottleneck = max(chunk_handoff_times)
        t_compute_bottleneck = max(stage_latencies)
        
        # The system moves at the speed of the slowest part (Compute vs Network)
        effective_bottleneck = max(t_compute_bottleneck, t_handoff_bottleneck)
        t_steady_state = (num_chunks - 1) * effective_bottleneck

        # 4. Final Handoff Tail
        # The TTFT is complete when the LAST chunk's KV-cache arrives at the Decode node.
        # Since the last rank finishes its compute at t_fill + t_steady_state, 
        # we only add the transfer time of that final chunk.
        t_final_chunk_handoff = max(chunk_handoff_times) 

        return t_fill + t_steady_state + t_final_chunk_handoff

    def TPOT_throughput(self, N: int, T: int) -> float:
        """
        Calculate the Time Per Output Token (TPOT) for the pipeline node (throughput mode).
        N: batch size
        T: context length (sequence length)
        Returns TPOT in seconds.
        """
        # For each output token, all ranks must process the token and transfer hidden states.
        # The slowest stage (compute or comm) determines the pipeline throughput per token.
        stage_times = []
        for i, rank in enumerate(self.ranks):
            t_compute = rank.T_decode(N, T)
            t_comm = 0.0
            if i < len(self.ranks) - 1:
                t_comm = rank.X_decode(N=N)
            stage_times.append(t_compute + t_comm)
        # TPOT is determined by the slowest stage (pipeline bottleneck)
        return max(stage_times)

    def TPOT_latency(self, N: int, T: int) -> float:
        """
        Calculate the Time Per Output Token (TPOT) for the pipeline node (sequential latency mode).
        N: batch size
        T: context length (sequence length)
        Returns TPOT in seconds.
        """
        # For each output token, all ranks must process the token and transfer hidden states sequentially.
        total_time = 0.0
        for i, rank in enumerate(self.ranks):
            t_compute = rank.T_decode(N, T)
            t_comm = 0.0
            if i < len(self.ranks) - 1:
                t_comm = rank.X_decode(N=N)
            total_time += t_compute + t_comm
        return total_time



# Example: Instantiate PPInferenceNode for LLaMA-3.1-70B with pipeline parallelism (PP=4)
#
# from your_module import GPU, CommLink, ModelAttributes, PPInferenceNode
#
# model = ModelAttributes.from_name("LLaMA-3.1-70B")
#
# # For the Prefill pipeline, use four GPUs and Infiniband HDR for inter-rank communication
# gpu_list = [GPU("NVIDIA RTX PRO 6000") for _ in range(4)]
# prefill_ranks = PPInferenceNode.build_ranks(model, gpu_list, "Infiniband HDR")
# prefill_node = PPInferenceNode(prefill_ranks)
#
# # Access the first rank:
# rank0 = prefill_node.get_rank(0)
# print(rank0.num_layers, rank0.gpu.name, rank0.comm_link.name)
#
# # For the Decode pipeline, use four GPUs and Infiniband HDR for inter-rank communication
# decode_gpu_list = [GPU("NVIDIA RTX PRO 6000") for _ in range(4)]
# decode_ranks = PPInferenceNode.build_ranks(model, decode_gpu_list, "Infiniband HDR")
# decode_node = PPInferenceNode(decode_ranks)
#
# # Example: Compute KV transfer time from prefill to decode ranks via PCIe Gen4
# pcie_link = CommLink("PCIe Gen4")
# kv_transfer_time = prefill_node.KV_transfer_time_for_PD_disaggregation(N=4, T=8192, network_link=pcie_link, parallel=True)
# print(f"KV transfer time (PCIe Gen4, parallel): {kv_transfer_time:.4f} sec")
#
# # Compute TTFT for batch size 4, context 8192
# ttft = prefill_node.TTFT_no_chunking(N=4, T=8192)
# print(f"TTFT: {ttft:.4f} sec")
#
# # Compute TPOT (throughput and latency modes)

# Example: Print feasible PP prefill pipeline configurations for LLaMA-3.1-70B

# Print table of feasible num_ranks for each (batch_size, context_size)
if __name__ == "__main__":
    import pandas as pd

    model = Model.from_name("LLaMA-3.1-70B")
    gpu = GPU.from_name("NVIDIA H100")
    configs = model.feasible_pp_prefill_configs(gpu)

    batch_sizes = sorted(set(cfg[1] for cfg in configs))
    context_sizes = sorted(set(cfg[2] for cfg in configs))

    table = pd.DataFrame('', index=batch_sizes, columns=context_sizes)
    for num_ranks, batch_size, context_size in configs:
        current = table.at[batch_size, context_size]
        if pd.isna(current) or not str(current).strip():
            table.at[batch_size, context_size] = str(num_ranks)
        else:
            current_set = set(map(int, str(current).split(',')))
            current_set.add(num_ranks)
            table.at[batch_size, context_size] = ','.join(map(str, sorted(current_set)))

    table.index.name = 'Batch Size'
    table.columns.name = 'Context Size'

    print("\nFeasible num_ranks for each (batch_size, context_size) on", gpu.name)
    print(table.to_string())

