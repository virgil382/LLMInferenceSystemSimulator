## Plan: Adapt DisaggregatedPDSystem2D for Tensor Parallelism (TP) at Every Rank

This plan will modify the `DisaggregatedPDSystem2D` class so that each prefill pipeline rank also performs tensor parallelism (TP), rather than only pipeline parallelism (PP). 

Integrating Tensor Parallelism (TP) alongside Pipeline Parallelism (PP) transforms the model into a 2D parallelism scheme. In this configuration, each "rank" in your pipeline is no longer a single GPU, but a TP Group of GPUs working in lockstep.

The primary change is that a single ComputeJob for a layer is now split into multiple parallel compute tasks, and new All-Reduce or All-Gather communication overheads must be modeled within each pipeline stage.

# 1. Updated Architectural Assumptions
- TP Group: A set of $TP$ GPUs. Each GPU holds a shard of the weights for the layers assigned to that PP stage.
- Intra-Stage Communication: Within a single PP stage, GPUs must perform an All-Reduce operation twice per Transformer layer (once after Attention, once after MLP).
- Inter-Stage Communication: When moving from PP stage $i$ to $i+1$, each GPU in TP Group $i$ typically sends its activation shard to the corresponding GPU in TP Group $i+1$.

# 2. New Compute Job Naming Convention
To track 2D parallelism, compute jobs must include the TP index ($t$) in addition to the PP index ($p$) and chunk index ($c$).

Format: P_Rank_PP[p]_TP[t]_Chunk[c]
- P: Prefill phase.
- PP[p]: The pipeline stage (0 to $PP-1$).
- TP[t]: The tensor parallel shard index (0 to $TP-1$).
- Chunk[c]: The specific prefill chunk being processed.

# 3. New Data Exchange (Batch) Naming Convention
## A. Intra-Stage: TP All-Reduce
In the prefill phase, All-Reduce is typically implemented as a Reduce-Scatter followed by an All-Gather. For the simulator, we model the total data moved to synchronize a layer.
- Name: TP_AR_PP[p]_Layer[l]_Chunk[c]
- Path: Typically uses NVLink or NVSwitch (local to the server).
- Trigger: Starts after the partial matrix multiplication compute job on each TP GPU.

## B. Inter-Stage: PP Activation Shards
Instead of one large activation batch, we send $TP$ parallel shards.
- Name: PP_Act_FromP[p]_ToP[p+1]_TP[t]_Chunk[c]
- Path: [GPU_P[p][t]_PCI, GPU_P[p+1][t]_PCI] (Assuming they share the same PLX).

## C. Handoff: Disaggregated KV-Cache Shards
Each TP GPU in the prefill cluster is responsible for a shard of the KV cache.
- Name: Handoff_PP[p]_TP[t]_Chunk[c]
- Path: Same as your original IB path, but mapped to NICs based on the global GPU index $(p \times TP + t)$.

# 4. Implementation Plan
## Step 1: Update Constructor & VRAM Logic
- Add tp_degree parameter.
- Update params_per_rank: (W // L) * (L // pp_degree) // tp_degree.
- Update kv_bytes: Divide the per-rank KV cache by tp_degree (since heads are sharded).

## Step 2: Update T_prefill for TP Overhead
The compute time remains largely the same (total FLOPS divided by total GPUs), but you must add a "TP Sync" penalty.

$$T_{sync} \approx 2 \times \text{LayersPerRank} \times \text{AllReduce}(Size_{act} / TP)$$

## Step 3: Orchestration State Machine (on_compute_complete)
1. Start: P_Rank_PP[0]_TP[0...t]_Chunk[0] compute jobs are added.
2. TP Sync: When a TP compute job finishes, check if all $t$ peers in that group are done. If yes, inject the TP_AR data batches.
3. PP Forward: When TP_AR is complete, inject the PP_Act shards to the next PP stage.
4. Handoff: Simultaneously inject the Handoff shards to the decode cluster.

## Step 4: Logic for "All Peers Ready"
Add a tracker (e.g., self.tp_sync_gate = defaultdict(int)) to ensure the pipeline only moves forward when all $TP$ GPUs have finished their local shard of the computation.
