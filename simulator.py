import abc
from typing import Optional

class CommChannel:
    """
    Class representing a communication channel and its key attributes.
    Bandwidth is in bytes/sec, latency is in seconds.
    For attributes with a range, the main accessor returns the average, and there are methods for low and high values.
    """
    _comm_db: dict[str, dict[str, tuple[float, float]]] = {
        "NVLink": {"bw_gbps": (300, 400), "lat_us": (1, 2)},
        "PCIe Gen4": {"bw_gbps": (12, 15), "lat_us": (18, 25)},
        "PCIe Gen5 x16": {"bw_gbps": (55, 60), "lat_us": (10, 15)},

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

class DataBatch:
    def __init__(self, name, size_bytes, path_channels):
        self.name = name
        self.total_size = size_bytes
        self.rem_size = size_bytes
        self.path = path_channels  # List of CommChannel objects
        
        # Total latency is the sum of latencies in the path
        self.total_latency = sum(c.latency_s for c in self.path)
        self.latency_remaining = self.total_latency
        
        self.start_time: float | None = None
        self.end_time: float | None = None

class ComputeJob:
    def __init__(self, name, duration_s):
        self.name = name
        self.total_duration = duration_s
        self.rem_duration = duration_s
        self.start_time = None
        self.end_time = None

class SimulatedSystem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def on_data_transfer_complete(self, simulator, batch): pass

    @abc.abstractmethod
    def on_compute_complete(self, simulator, compute_job): pass

class CommNetworkSimulator:
    def __init__(self):
        self.active_batches = []
        self.active_compute = []
        self.completed_batches = []
        self.completed_compute = []
        self.current_time = 0.0

    def add_batch(self, batch):
        batch.start_time = self.current_time
        self.active_batches.append(batch)

    def add_compute(self, compute_job):
        compute_job.start_time = self.current_time
        self.active_compute.append(compute_job)

    def run(self, system, t_delta=0.00001): # Smaller delta for microsecond precision
        while self.active_batches or self.active_compute:
            # 1. Update Network Transfers
            if self.active_batches:
                # Only batches that have finished their 'latency' phase consume bandwidth
                transferring = [b for b in self.active_batches if b.latency_remaining <= 0]
                waiting = [b for b in self.active_batches if b.latency_remaining > 0]

                # Resolve bandwidth for those actually moving data
                if transferring:
                    rates = self._calculate_max_min_rates(transferring)
                    for b in transferring:
                        b.rem_size -= rates[b] * t_delta
                
                # Update latency for those still in flight
                for b in waiting:
                    b.latency_remaining -= t_delta

                # Check for completion
                for b in self.active_batches[:]:
                    if b.rem_size <= 0 and b.latency_remaining <= 0:
                        b.end_time = self.current_time
                        self.active_batches.remove(b)
                        self.completed_batches.append(b)
                        job = system.on_data_transfer_complete(self, b)
                        if isinstance(job, ComputeJob): self.add_compute(job)

            # 2. Update Compute
            for c in self.active_compute[:]:
                c.rem_duration -= t_delta
                if c.rem_duration <= 0:
                    c.end_time = self.current_time
                    self.active_compute.remove(c)
                    self.completed_compute.append(c)
                    system.on_compute_complete(self, c)

            self.current_time += t_delta

    def _calculate_max_min_rates(self, batches):
        # We need to map which batches are using which physical CommChannel objects
        rates = {b: 0.0 for b in batches}
        fixed = set()
        
        # Get all unique channels currently in use
        active_channels = set()
        for b in batches:
            for channel in b.path:
                active_channels.add(channel)
        
        rem_cap = {c: c.bandwidth_bps for c in active_channels}

        while len(fixed) < len(batches):
            channel_users = {c: 0 for c in active_channels}
            for b in batches:
                if b not in fixed:
                    for c in b.path: channel_users[c] += 1

            bottleneck_rate = float('inf')
            for b in batches:
                if b not in fixed:
                    for c in b.path:
                        share = rem_cap[c] / channel_users[c]
                        bottleneck_rate = min(bottleneck_rate, share)

            newly_fixed = []
            for b in batches:
                if b not in fixed:
                    if any((rem_cap[c]/channel_users[c]) <= bottleneck_rate + 1e-9 for c in b.path):
                        rates[b] = bottleneck_rate
                        newly_fixed.append(b)

            for b in newly_fixed:
                fixed.add(b)
                for c in b.path: rem_cap[c] -= rates[b]
        return rates
