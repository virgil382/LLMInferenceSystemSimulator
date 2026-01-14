# --- LLM and GPU classes moved to the top ---
from typing import Optional

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

# - K: Number of chunks the prefill is broken into.
# - M: The size of one chunk (T / K).

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

class LLM:
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
                 H_kv: Optional[int] = None,
                 H_q: Optional[int] = None,
                 D_latent: Optional[int] = None,
                 attention_type: Optional[str] = None,
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
    def from_name(cls, name: str) -> 'LLM':
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
        return list({
            "LLaMA-3.1-70B",
            "GPT-OSS-20B",
            "DeepSeek-V3",
            "Mistral-Small-24B",
            "Qwen2.5-72B",
            "LLaMA-3.1-405B",
            "Mixtral-8x22B",
        })

    def feasible_pp_prefill_configs(self, gpu) -> set[tuple[int, int, int]]:
        feasible = set()
        vram_limit = 0.70 * gpu.vram_b
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
        attn_type = self.attention_type
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
        local_vars = {}
        for k, (v, typ) in required.items():
            if v is None:
                raise ValueError(f"Model attribute '{k}' is None; cannot compute KV-cache size.")
            try:
                local_vars[k] = typ(v)
            except Exception as e:
                raise TypeError(f"Model attribute '{k}' could not be cast to {typ.__name__}: {e}")
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


class GPU:
    """
    Class representing a GPU and its key attributes for inference modeling, including price.
    All values are normalized to bytes (B) and FLOPS (not GB/s or TFLOPS).
    """
    def __init__(self, name: str, vram_b: float, vram_bw_bps: float, flops: float, comm_channel, price: float):
        self.name = name
        self.vram_b = vram_b
        self.vram_bw_bps = vram_bw_bps
        self.flops = flops
        self.comm_channel = comm_channel
        self.price = price

    @classmethod
    def from_name(cls, name: str) -> 'GPU':
        gpus = {
            "NVIDIA RTX PRO 6000": dict(vram_b=96 * 1024**3, vram_bw_bps=1694.5 * 1e9, flops=440 * 1e12, comm_channel=None, price=8346),
            "NVIDIA H100":         dict(vram_b=80 * 1024**3, vram_bw_bps=3350 * 1e9, flops=989 * 1e12, comm_channel=None, price=29999),
            "NVIDIA A100-80GB":    dict(vram_b=80 * 1024**3, vram_bw_bps=2000 * 1e9, flops=312 * 1e12, comm_channel=None, price=23300),
            "NVIDIA L40":          dict(vram_b=48 * 1024**3, vram_bw_bps=864 * 1e9, flops=181 * 1e12, comm_channel=None, price=11225),
            "NVIDIA H200":         dict(vram_b=141 * 1024**3, vram_bw_bps=4800 * 1e9, flops=989 * 1e12, comm_channel=None, price=30250),
            "AMD Instinct MI300A": dict(vram_b=128 * 1024**3, vram_bw_bps=5300 * 1e9, flops=981 * 1e12, comm_channel=None, price=29396),
            "AMD Instinct MI300X": dict(vram_b=192 * 1024**3, vram_bw_bps=5300 * 1e9, flops=1307 * 1e12, comm_channel=None, price=32000),
        }
        if name not in gpus:
            raise ValueError(f"Unknown GPU: {name}")
        return cls(name=name, **gpus[name])  # type: ignore

    @classmethod
    def available_gpus(cls) -> list[str]:
        return list({
            "NVIDIA RTX PRO 6000",
            "NVIDIA H100",
            "NVIDIA A100-80GB",
            "NVIDIA L40",
            "NVIDIA H200",
            "AMD Instinct MI300A",
            "AMD Instinct MI300X",
        })


# --- End move ---
# === Pipeline Parallelism Classes ===

class PPRank:
    """
    Represents a PP pipeline rank. Base class for PrefillPPRank and DecodePPRank.
    Has a single GPU and is responsible for L_rank layers of the LLM's total number of layers.

    Parameters:
        llm (LLM): The LLM model instance (reference).
        gpu (GPU): The GPU assigned to this rank.
        L_rank (int): Number of layers assigned to this rank.
        N (int): Batch size.
        T (int): Context length.
    """
    def __init__(self, llm: LLM, gpu: GPU, L_rank: int, N: int, T: int):
        self.llm = llm
        self.gpu = gpu
        self.L_rank = L_rank
        self.N = N
        self.T = T

    def W_rank(self) -> int:
        """Returns the volume of model weights assigned to this rank."""
        # Each rank gets (total weights / total layers) * layers assigned to this rank
        return (self.llm.W // self.llm.L) * self.L_rank


class PrefillPPRank(PPRank):
    """
    Represents a prefill PP pipeline rank. Inherits from PPRank.
    Provides methods for data volumes: W_rank, A_prefill, KV_handoff.
    """
    def A_prefill(self) -> int:
        """Returns the volume of activations (in bytes) to send to the next PrefillPPRank after computing KV-cache for a batch."""
        return self.N * self.T * self.llm.H_model * self.llm.B

    def KV_handoff(self, spec_prefill_enabled: bool = False) -> int:
        """Returns the volume of KV-cache share (in bytes) to hand off to the corresponding decode rank."""
        full_kv = self.llm.KV(self.N, self.T)

        # The paper 'Speculative Prefill' often uses a keep_rate (alpha) 
        # around 0.1 to 0.3 for long context tasks.
        keep_rate = 0.1 if spec_prefill_enabled else 1.0

        return int((full_kv * self.L_rank / self.llm.L) * keep_rate)
    
    def T_prefill(self, spec_prefill_enabled: bool = False) -> float:
        """
        Corrected Prefill Time with explicit H^2 scaling for projections and T^2 scaling for the attention matrix.
        """
        N = self.N
        T = self.T
        H_model = self.llm.H_model
        L_rank = self.L_rank

        # 1. Compute Time (FLOPS)
        # Projections: Q, K, V, and O projections + MLP (up/down/gate)
        # Most models approximate total linear FLOPS as 2 * N * T * W_per_rank
        # But to be explicit about the H^2 in the attention block projections:
        # flops_attn_projections = L_rank * (8 * N * T * H_model**2)

        # Linear FLOPS (including MLP and Attention Projections)
        flops_linear = 2 * N * T * self.W_rank()

        # Quadratic Attention FLOPS (The QK^T and Score * V part)
        # Standard formula: 2 * N * L_rank * H_model * T^2
        flops_quadratic = 2 * N * L_rank * H_model * (T**2)

        # A constant factor representing the reduction in attention complexity
        spec_prefill_multiplier = 0.4 if spec_prefill_enabled else 1.0
        flops_quadratic *= spec_prefill_multiplier

        total_flops = flops_linear + flops_quadratic
        t_compute = total_flops / self.gpu.flops

        # 2. Memory Time (VRAM BW)
        # Must read weights + write the resulting KV cache
        weight_bytes = self.W_rank() * self.llm.B
        kv_bytes = self.KV_handoff(spec_prefill_enabled=spec_prefill_enabled)

        vram_util = 0.80
        t_memory = (weight_bytes + kv_bytes) / (self.gpu.vram_bw_bps * vram_util)

        return max(t_compute, t_memory)


class DecodePPRank(PPRank):
    """
    Represents a decode PP pipeline rank. Inherits from PPRank.
    Provides method for A_decode.
    """
    def A_decode(self) -> int:
        """Returns the volume of activations (in bytes) to send to the next DecodePPRank after computing new KV-cache entry for next token."""
        # Placeholder: actual formula depends on model details
        return self.N * self.llm.H_model * self.llm.B

    def T_decode(self) -> float:
        """
        Calculates the amount of time (in seconds) that the GPU needs to calculate the activations for 
        the next rank and the new entry in the KV cache. Includes both compute and memory time, 
        returns the maximum.
        """
        N = self.N
        H_model = self.llm.H_model
        L_rank = self.L_rank
        B = self.llm.B
        # For decode, T=1 (one token at a time)
        T = 1

        # 1. Compute Time (FLOPS)
        # Linear FLOPS (including MLP and Attention Projections)
        flops_linear = 2 * N * T * self.W_rank()

        # Quadratic Attention FLOPS (QK^T and Score * V part)
        flops_quadratic = 2 * N * L_rank * H_model * (T**2)

        total_flops = flops_linear + flops_quadratic
        t_compute = total_flops / self.gpu.flops

        # 2. Memory Time (VRAM BW)
        # Must read weights + write the resulting KV cache (for one token)
        weight_bytes = self.W_rank() * B
        # KV-cache for one token
        kv_bytes = self.llm.KV(N, T) // self.llm.L * L_rank

        vram_util = 0.80
        t_memory = (weight_bytes + kv_bytes) / (self.gpu.vram_bw_bps * vram_util)

        return max(t_compute, t_memory)


class PPPipeline:
    """
    Represents a PP pipeline. Consists of PPRanks.
    """
    def __init__(self, llm: LLM, PP: int, N: int, T: int, K: int = 1):
        self.llm = llm
        self.PP = PP
        self.N = N
        self.T = T
        self.K = K
        # Micro-batch size: The amount of data in one pipeline 'beat'
        self.M = max(1, T // K)
        # Placeholder for rank models, to be set by subclasses
        from typing import Optional
        self.prefill_rank_model: Optional[PrefillPPRank] = None
        self.decode_rank_model: Optional[DecodePPRank] = None

    
from abc import ABC, abstractmethod

class PrefillPPPipeline(PPPipeline, ABC):
    """
    Abstract prefill PP pipeline. Contains methods for pipeline statistics.
    """
    @abstractmethod
    def X_prefill_activations(self, r: int) -> float:
        """Latency (s) of transfer of activations from PrefillPPRank r to r+1."""
        pass

    @abstractmethod
    def X_handoff(self, r: int) -> float:
        """Latency (s) of transfer of activations from PrefillPPRank r to corresponding decode rank."""
        pass

    def T_rank(self, r: int) -> float:
        """
        Calculate the time that rank r takes to:
            - compute its share of the prefill (i.e. T_prefill())
            - transfer its activations to the next rank (i.e. X_prefill_activations()).  Note that the final rank does not have this step.
            - transfer the prefill to the corresponding rank in the decode cluster (i.e. X_handoff())
        The two transfers are sequential, but they both overlap with the compute.

        Returns the total time for this rank to complete its work.
        """
        # Compute time for this rank
        if self.prefill_rank_model is None:
                raise AttributeError("prefill_rank_model must be set before calling T_prefill().")
        t_compute = self.prefill_rank_model.T_prefill(self.spec_prefill_enabled)

        # Communication times
        t_x_activ = self.X_prefill_activations(r) if r < self.PP - 1 else 0.0
        t_x_handoff = self.X_handoff(r)

        # The two transfers are sequential, but both overlap with compute
        t_comm = t_x_activ + t_x_handoff

        # The total time is the maximum of compute and the sum of comms
        return max(t_compute, t_comm)

    def T_prefill_completion(self) -> float:
        """
        Calculate the total time (s) for the prefill pipeline to complete processing all K micro-batches.
        This implementation sums the per-rank, per-microbatch times, accounting for overlap, without using the bubble formula.
        """
        # For each micro-batch, each rank processes in sequence (no pipeline overlap)
        total_time = 0.0
        for k in range(self.K):
            # For each rank, accumulate the time for this micro-batch
            for r in range(self.PP):
                total_time += self.T_rank(r)
        return total_time


class DecodePPPipeline(PPPipeline, ABC):
    """
    Abstract decode PP pipeline. Contains methods for pipeline statistics.
    """
    @abstractmethod
    def X_decode_activations(self, r: int) -> float:
        """Latency (s) of transfer of activations from DecodePPRank r to r+1."""
        pass

    def TPOT(self) -> float:
        """
        Time per output token (s).
        In a pipeline, the throughput is determined by the bottleneck stage.
        Even though the token travels through all ranks, the next token 
        can begin as soon as the bottleneck stage is free.
        """
        # Calculate the total time each stage takes (Compute at rank R + Comm to R+1)
        # Note: The final rank has no 'X_decode' transfer, just T_decode compute.
        stage_latencies = []
        for r in range(self.PP - 1):
            stage_latencies.append(self.X_decode_activations(r))
        
        # Add the final rank's compute time (it has no outgoing X_decode)
        # We access the rank model through the deployment class in implementation
        if self.decode_rank_model is None:
            raise AttributeError("decode_rank_model must be set before calling T_decode().")
        stage_latencies.append(self.decode_rank_model.T_decode())
            
        return max(stage_latencies)


class AkamaiDeployment(PrefillPPPipeline, DecodePPPipeline):
    """
    Represents a physical deployment of a PD disaggregated inference system.
    Implements abstract methods of PrefillPPPipeline and DecodePPPipeline.

    Topology:
        - Prefill Pipeline Server: Each rank of the prefill pipeline is realized as a separate GPU with a PCI interface plugged into the PCI slots of the same server with a PLX switch. The server has a single Infiniband NDX card occupying one of the PCI slots.
        - Decode Pipeline Server: Each rank of the decode pipeline is realized as a separate GPU with a PCI interface plugged into the PCI slot of a separate server with a PLX switch. The same server hosts other ranks of other decode PP pipelines plugged into other PCI slots. The server also hosts an Infiniband NDX card plugged into another slot. The Infiniband card has one port configured as Ethernet.

    Data Transfer Paths:
        - The transfer path for the prefill inter-rank activations is GPU A -> PCI lanes A -> PLX -> PCI lanes B -> GPU B
        - The transfer path for the prefill KV-cache share handoff is GPU A -> PCI lanes A -> PLX -> PCI lanes C -> Infiniband card X -> Infiniband switch -> Infiniband card Y -> PCI lanes D -> PLX -> PCI lanes E -> GPU E
        - The transfer path for the decode inter-rank activations is GPU E -> PCI lanes E -> PLX -> Infiniband card Y Ethernet port -> Ethernet switch -> Infiniband card Z Ethernet port -> PCI lanes Z -> PLX -> PCI lanes F -> GPU F

    Contentions:
        - On the prefill pipeline server:
            - PCI lanes A are used to send the inter-rank activations and the KV-cache shares, causing contention.
            - PCI lanes C are used to send the KV-cache shares from multiple prefill GPUs, causing contention.
        - On the decode pipeline server:
            - PCI lanes Z are used to transfer the received KV-cache shares from multiple prefill ranks of other prefill pipelines, causing contention.  However, we can assume that these transfers occur infrequently only during the initialization of a decode pipeline in which GPU F participates.
            - PCI lanes Z are used to transfer the received decode inter-rank activations from decode ranks, causing additional contention.  We can assume that these transfers occur frequently (each time the decode pipeline in which GPU F participates decodes a token) 

    This class models the above topology and data transfer paths, allowing calculation of pipeline latencies and contentions without requiring the README for reference.
    """
    def __init__(self, llm, PP, N, T, K, prefill_gpu, decode_gpu, pci_link, ib_link, n_ib_links, eth_link, spec_prefill_enabled=False):
        """
        Initialize an AkamaiDeployment instance representing a physical deployment of a PD disaggregated inference system.

        Parameters:
            llm (LLM): The language model instance.
            PP (int): Number of pipeline parallel ranks (number of GPUs per pipeline).
            N (int): Batch size (number of sequences processed in parallel).
            T (int): Context size (sequence length in tokens).
            K (int): Number of chunks the prefill is broken into (micro-batches).
            prefill_gpu (GPU): GPU instance used for prefill pipeline ranks.
            decode_gpu (GPU): GPU instance used for decode pipeline ranks.
            pci_link (CommChannel): PCIe communication channel instance for intra-server GPU communication.
            ib_link (CommChannel): Infiniband communication channel instance for inter-server communication.
            n_ib_links (int): Number of Infiniband links available. Each has its own "PCI lanes C" to reduce contention.
            eth_link (CommChannel): Ethernet communication channel instance for decode inter-rank communication.
            spec_prefill_enabled (bool): Whether to enable speculative prefill optimizations affecting attention computation. (See https://github.com/Jingyu6/speculative_prefill).
        """
        super().__init__(llm, PP, N, T, K)

        self.pci = pci_link
        self.ib = ib_link
        self.n_ib_links = n_ib_links
        self.eth = eth_link
        self.spec_prefill_enabled = spec_prefill_enabled
        
        # We pass N (full batch) but M (the micro-segment of the sequence)
        # This ensures T_prefill() uses M for the T^2 calculation.
        self.prefill_rank_model = PrefillPPRank(llm, prefill_gpu, llm.L // PP, N, self.M)
        
        # Decode still works on T=1 (the next token)
        self.decode_rank_model = DecodePPRank(llm, decode_gpu, llm.L // PP, N, 1)

    def X_prefill_activations(self, r: int) -> float:
        """
        Latency of forwarding inter rank activations.

        Path: 
            GPU A -> PCI lanes A -> 
            PLX -> 
            PCI lanes B -> GPU B
        
        Contentions:
            - PCI lanes A are used to send the inter-rank activations and the KV-cache shares, causing contention.

        Modeling Notes:
            If two volumes V_1 and V_2 contend for a link with physical bandwidth BW, then the total time to 
            clear both is (V_1 + V_2) / BW. The effective bandwidth for V_1 is the portion of the total bandwidth
            it "sees" based on its share of the total payload so that its effective bandwidth is:
              BW_eff_1 = BW * ( V_1 / (V_1 + V_2))
        """
        if self.prefill_rank_model is None:
            raise AttributeError("prefill_rank_model must be set before calling T_prefill().")
        
        # 1. Define Volumes
        v_act = self.prefill_rank_model.A_prefill()     # V_1: Activations to next rank
        v_kv = self.prefill_rank_model.KV_handoff()     # V_2: KV-cache to handoff
        
        # 2. Model Contention on PCI Lanes A (GPU -> PLX)
        # Assume both activations and KV-cache share contend for the same link.
        # Apply the modeling note formula: BW_eff = BW * (V_1 / (V_1 + V_2))
        bw_eff_a = self.pci.bandwidth_bps * (v_act / (v_act + v_kv))
        
        # 3. Model PCI Lanes B (PLX -> next GPU)
        # Assuming no contention on the destination GPU's ingress for prefill
        bw_eff_b = self.pci.bandwidth_bps 
        
        # 4. Bottleneck and Communication Time
        # The transfer is limited by the slowest segment
        bottleneck_bw = min(bw_eff_a, bw_eff_b)
        
        # Transmission time + serialization latency
        t_comm = (v_act / bottleneck_bw) + self.pci.latency_s
        
        return t_comm

    def X_handoff(self, r: int) -> float:
        """
        Calculates the flight time of a KV-cache share from a Prefill rank to its Decode counterpart.
        
        Path: 
            GPU A -> PCI Lanes A -> PLX -> PCI Lanes C -> IB Card X -> 
            Infiniband Switch -> 
            IB Card Y -> PCI Lanes D -> PLX -> PCI Lanes E -> GPU E

        Contentions:
            - PCI lanes A are used to send the inter-rank activations and the KV-cache shares, causing contention.
            - PCI lanes C are used to send the KV-cache shares from multiple prefill GPUs, causing contention.

        Modeling Notes:
            If two volumes V_1 and V_2 contend for a link with physical bandwidth BW, then the total time to 
            clear both is (V_1 + V_2) / BW. The effective bandwidth for V_1 is the portion of the total bandwidth
            it "sees" based on its share of the total payload so that its effective bandwidth is:
              BW_eff_1 = BW * ( V_1 / (V_1 + V_2))
        """
        if self.prefill_rank_model is None:
            raise AttributeError("prefill_rank_model must be set before calling KV_handoff() or A_prefill().")
        if self.decode_rank_model is None:
            raise AttributeError("decode_rank_model must be set before calling A_decode().")
        v_kv = self.prefill_rank_model.KV_handoff()
        v_act_pre = self.prefill_rank_model.A_prefill()
        v_act_dec = self.decode_rank_model.A_decode()
        
        # 1. Effective BW: Lanes A (Local GPU Egress)
        # Ratio of KV to (KV + Prefill Activation)
        bw_eff_a = self.pci.bandwidth_bps * (v_kv / (v_kv + v_act_pre))
        
        # 2. Effective BW: Lanes C (Prefill Root Egress to IB)
        # Contended by all PP ranks sending their shares sequentially.  n_ib_links reduces contention.
        bw_eff_c = self.n_ib_links * self.pci.bandwidth_bps * (v_kv / (v_kv * self.PP))
        
        # 3. Effective BW: IB Link (Network)
        bw_eff_ib = self.ib.bandwidth_bps * (v_kv / (v_kv * self.PP))
        
        # 4. Effective BW: Lanes D (Decode Root Ingress)
        # The most congested segment: All KV handoffs + All Decode Activations
        total_vol_d = (v_kv * self.PP) + (v_act_dec * self.PP)
        bw_eff_d = self.pci.bandwidth_bps * (v_kv / total_vol_d)
        
        # 5. Effective BW: Lanes E (Local Decode GPU Ingress)
        bw_eff_e = self.pci.bandwidth_bps 

        # Bottleneck: The path is only as fast as its slowest effective segment
        bottleneck_bw = min(bw_eff_a, bw_eff_c, bw_eff_ib, bw_eff_d, bw_eff_e)
        
        return (v_kv / bottleneck_bw) + (2 * self.pci.latency_s + self.ib.latency_s)

    def X_decode_activations(self, r: int) -> float:
        """
        Calculates the latency for a single decode token activation transfer.
        
        Path:
            GPU E -> PCI lanes E -> PLX -> IB card Y (Ethernet mode) -> 
            Ethernet switch -> 
            IB card Z (Ethernet mode) -> PCI lanes Z -> PLX -> PCI lanes F -> GPU F

        Contentions:
            1. Intra-Server Decode Contention (Frequent): 
               There are 4 GPUs on the decode server. When rank 'r' sends its activation, 
               other decode pipelines on the same server are likely also using the 
               IB/Ethernet card. We model this by assuming the 4 GPUs share the IB bandwidth.
            2. KV-Handoff Contention (Infrequent): 
               Occasionally (modeled at 5% frequency), a new prefill-to-decode handoff 
               saturates the root PCI lanes (Lanes Z), delaying the decode activation.

        Modeling Notes:
            - We use a weighted average of 'Steady State' decode and 'Initialization 
              Contended' decode to find the mean expected latency.
        """
        if self.decode_rank_model is None:
            raise AttributeError("decode_rank_model must be set before calling A_decode().")
        if self.prefill_rank_model is None:
            raise AttributeError("prefill_rank_model must be set before calling KV_handoff().")
        v_act_dec = self.decode_rank_model.A_decode()
        v_kv = self.prefill_rank_model.KV_handoff()
        num_gpus_per_server = 4
        init_frequency = 0.05  # 5% of cycles face KV-handoff contention
        
        # --- Scenario A: Steady State (95% of time) ---
        # Contention: 4 GPUs sharing the IB Card bandwidth for decode activations.
        bw_steady_ib = self.eth.bandwidth_bps / num_gpus_per_server
        t_comm_steady = (v_act_dec / min(self.pci.bandwidth_bps, bw_steady_ib)) + \
                        (2 * self.pci.latency_s + self.eth.latency_s)
        
        # --- Scenario B: Initialization Contention (5% of time) ---
        # Contention: Root PCI lanes (Z) are saturated by both the 4-GPU decode 
        # shuffle AND the incoming KV-cache handoff stream from the prefill cluster.
        total_vol_root = (v_kv * self.PP) + (v_act_dec * num_gpus_per_server)
        bw_eff_root_contended = self.pci.bandwidth_bps * (v_act_dec / total_vol_root)
        
        t_comm_contended = (v_act_dec / min(bw_eff_root_contended, bw_steady_ib)) + \
                           (2 * self.pci.latency_s + self.eth.latency_s)

        # Weighted Average Communication Time
        t_comm_avg = ((1 - init_frequency) * t_comm_steady) + (init_frequency * t_comm_contended)
        
        return self.decode_rank_model.T_decode() + t_comm_avg

    def TTFT(self) -> float:
        """
        Calculates Time to First Token (TTFT).
        
        In a disaggregated PD (Prefill-Decode) architecture, TTFT is the duration 
        from the start of the prefill until the first token is generated by the 
        final rank of the decode pipeline.
        
        Formula:
            TTFT = (Time for Prefill to finish) + (Time for 1st token to clear Decode Pipeline)
        """
        t_prefill_end = self.T_prefill_completion()
        
        # The first token must visit every rank sequentially.
        # Sum of (Rank 0 compute + Comm) + (Rank 1 compute + Comm) ... + (Final Rank compute)
        if self.decode_rank_model is None:
            raise AttributeError("decode_rank_model must be set before calling T_decode().")
        t_first_token_path = sum(self.X_decode_activations(r) for r in range(self.PP - 1)) + self.decode_rank_model.T_decode()
        
        return t_prefill_end + t_first_token_path

    def total_deployment_cost(self) -> float:
        """
        Calculates the total hardware cost of the deployment.
        Assumes PP ranks for prefill and PP ranks for decode.
        """
        if self.prefill_rank_model is None:
            raise AttributeError("prefill_rank_model must be set before accessing its GPU price.")
        if self.decode_rank_model is None:
            raise AttributeError("decode_rank_model must be set before accessing its GPU price.")
        prefill_cost = self.PP * self.prefill_rank_model.gpu.price
        decode_cost = self.PP * self.decode_rank_model.gpu.price
        return prefill_cost + decode_cost

    def tokens_per_second(self) -> float:
        """
        Calculates the aggregate throughput (TPS) of the decode cluster.
        Formula: Batch Size (N) / TPOT
        """
        tpot = self.TPOT()
        if tpot == 0:
            return 0.0
        return self.N / tpot

    def calculate_tps_per_dollar(self) -> float:
        """
        Returns the economic efficiency of the deployment.
        Metric: Tokens/Sec per $100k USD of hardware investment.
        """
        tps = self.tokens_per_second()
        total_cost = self.total_deployment_cost()
        
        if total_cost == 0:
            return 0.0
        
        # Scaling by 100k makes the output easier to compare (e.g., 4.5 vs 1.2)
        return (tps / total_cost) * 100_000

    def generate_scaling_report(self, batch_sizes: list[int]):
        """Generates a scannable table of performance vs. batch size."""
        print(f"Prefill cluster IB links: {self.n_ib_links}")
        print(f"SpecPrefill enabled: {self.spec_prefill_enabled}")
        print(f"{'Batch (N)':<10} | {'TTFT (s)':<10} | {'TPOT (ms)':<10} | {'TPS':<10} | {'TPS/$100k':<10}")
        print("-" * 65)
        for n in batch_sizes:
            self.N = n
            # Update internal rank models for the new batch size
            if self.prefill_rank_model is not None:
                self.prefill_rank_model.N = n
            if self.decode_rank_model is not None:
                self.decode_rank_model.N = n
            ttft = self.TTFT()
            tpot_ms = self.TPOT() * 1000
            tps = self.tokens_per_second()
            eff = self.calculate_tps_per_dollar()
            print(f"{n:<10} | {ttft:<10.3f} | {tpot_ms:<10.1f} | {tps:<10.1f} | {eff:<10.1f}")

    def find_optimal_batch(self, max_ttft_s: float = 0.500):
        """
        Finds the largest batch size N that maximizes throughput 
        without violating the TTFT latency constraint.
        """
        best_n = 1
        max_eff = 0
        # Test powers of 2 (standard for GPU kernels)
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            self.N = n
            if self.prefill_rank_model is not None:
                self.prefill_rank_model.N = n
            if self.decode_rank_model is not None:
                self.decode_rank_model.N = n
            current_ttft = self.TTFT()
            if current_ttft > max_ttft_s:
                break # Latency constraint violated
            current_eff = self.calculate_tps_per_dollar()
            if current_eff > max_eff:
                max_eff = current_eff
                best_n = n
        return best_n, max_eff




# 1. Initialize Infrastructure
pci_g5 = CommChannel("PCI Express 5.0 x16")
ib_ndr = CommChannel("Infiniband NDR")
eth_100 = CommChannel("Ethernet 100G")

# 2. Select Model (LLaMA-3-70B)
llama3 = LLM.from_name("LLaMA-3.1-70B")

# 3. Define the Physical Deployments
# We'll compare a "Value" cluster vs a "Performance" cluster
clusters = {
    "NVIDIA RTX PRO 6000 Cluster 1": AkamaiDeployment(
        llama3, PP=4, N=1, T=4096, K=4, 
        prefill_gpu=GPU.from_name("NVIDIA RTX PRO 6000"),
        decode_gpu=GPU.from_name("NVIDIA RTX PRO 6000"),
        pci_link=pci_g5, ib_link=ib_ndr, n_ib_links=1, eth_link=eth_100, spec_prefill_enabled=False
    ),
    "NVIDIA RTX PRO 6000 Cluster 2": AkamaiDeployment(
        llama3, PP=4, N=1, T=4096, K=4, 
        prefill_gpu=GPU.from_name("NVIDIA RTX PRO 6000"),
        decode_gpu=GPU.from_name("NVIDIA RTX PRO 6000"),
        pci_link=pci_g5, ib_link=ib_ndr, n_ib_links=2, eth_link=eth_100, spec_prefill_enabled=False
    ),
    "NVIDIA RTX PRO 6000 Cluster 3": AkamaiDeployment(
        llama3, PP=4, N=1, T=4096, K=4, 
        prefill_gpu=GPU.from_name("NVIDIA RTX PRO 6000"),
        decode_gpu=GPU.from_name("NVIDIA RTX PRO 6000"),
        pci_link=pci_g5, ib_link=ib_ndr, n_ib_links=4, eth_link=eth_100, spec_prefill_enabled=False
    ),
    "H100 Performance Cluster 1": AkamaiDeployment(
        llama3, PP=4, N=1, T=4096, K=4, 
        prefill_gpu=GPU.from_name("NVIDIA H100"),
        decode_gpu=GPU.from_name("NVIDIA H100"),
        pci_link=pci_g5, ib_link=ib_ndr, n_ib_links=1, eth_link=eth_100, spec_prefill_enabled=False
    ),
    "H100 Performance Cluster 2": AkamaiDeployment(
        llama3, PP=4, N=1, T=4096, K=4, 
        prefill_gpu=GPU.from_name("NVIDIA H100"),
        decode_gpu=GPU.from_name("NVIDIA H100"),
        pci_link=pci_g5, ib_link=ib_ndr, n_ib_links=2, eth_link=eth_100, spec_prefill_enabled=False
    ),
    "H100 Performance Cluster 3": AkamaiDeployment(
        llama3, PP=4, N=1, T=4096, K=4, 
        prefill_gpu=GPU.from_name("NVIDIA H100"),
        decode_gpu=GPU.from_name("NVIDIA H100"),
        pci_link=pci_g5, ib_link=ib_ndr, n_ib_links=4, eth_link=eth_100, spec_prefill_enabled=False
    ),
}

# --- EXECUTION ---

for name, deployment in clusters.items():
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Hardware Cost: ${deployment.total_deployment_cost():,.2f}")
    
    # A. Generate Scaling Report (Comparing N=1 to N=128)
    print("\n[Scaling Report]")
    deployment.generate_scaling_report(batch_sizes=[1, 4, 8, 16, 32, 64, 128])
    
    # B. Run the Optimizer (SLA: TTFT must be under 400ms)
    sla_limit = 0.400 
    opt_n, opt_eff = deployment.find_optimal_batch(max_ttft_s=sla_limit)
    
    print(f"\n[Optimization Result]")
    print(f"Optimal Batch (N) for {sla_limit*1000:.0f}ms SLA: {opt_n}")
    print(f"Max Efficiency at SLA: {opt_eff:.2f} TPS/$100k")
