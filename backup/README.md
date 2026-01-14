# AI Inference Service Math Model

This project is a tool that helps to plan the topology of an LLM inference service.

It provides a class named `GPU` whose instances contain the attributes of various GPUs. Some of the attributes are the GPU name, the amount of VRAM, the VRAM bandwidth, etc.


It provides a class named `LLM` whose properties represent various LLM attributes such as the model name, the number of weights, the bytes per weight, etc. The `LLM` class also contains a method named `KV` that returns the KV-cache size for that model.

`LLM` also implements a method named `feasible_pp_prefill_configs` that generates tuples, each of which represents a prefill PP pipeline capable of generating a complete KV-cache for a batch of size N and context size T. The values of each tuple can be used to construct feasible `PPPipeline`s for additional modeling.

## Simulating a PP Pipeline

### PPPipeline
The class `PPPipeline` represents a PP pipeline. It consists of PP `PPRank`s. Its constructor takes the following parameters:
- the `LLM`, to which it keeps a reference
- the length `PP` of the pipeline
- the batch size `N`
- the context length `T`

### PPRank
The class `PPRank` represents a PP pipeline rank. It is the base class of `PrefillPPRank` and `PrefillPPRank`. It has a single `GPU` and it is responsible for `L_rank` layers of the `LLM`'s total number of layers. Its constructor takes the following parameters:
- the `LLM`, to which it keeps a reference
- the `GPU`
- `L_rank`
- the batch size `N`
- the context length `T`
It impleents the following methods:
- `W_rank` calculates and returns the volume of the model weights assigned to the PP rank.

### PrefillPPRank
The class `PrefillPPRank` is a `PPRank` that represents a prefill PP pipeline rank and contains properties and methods that return various quantities that generally describe data volumes. For example:
- `A_prefill` calculates and returns the volume of activations (in bytes) that the `PrefillPPRank` must send to the next `PrefillPPRank` after computing its share of the KV-cache for a batch.
- `KV_handoff` calculates and returns the volume of the KV-cache share (in bytes) that the `PrefillPPRank` computes. This share of the KV-cache must be handed off to a corresponding rank in the decode pipeline.
- `T_prefill` calculates the amount of time (in seconds) that the GPU needs to calculate the activations for the next rank and the rank's share of the KV-cache

### DecodePPRank
The class `DecodePPRank` is a `PPRank` that represents a decode PP pipeline rank and contains properties and methods that return various quantities that generally describe data volumes. For example:
- `A_decode` calculates and returns the volume of activations (in bytes) that the `DecodePPRank` must send to the next `DecodePPRank` after computing its share of the new KV-cache entry for the next token generated from a batch.
- `T_decode` calculates the amount of time (in seconds) that the GPU needs to calculate the activations for the next rank and the new entry in the KV cache.


### PrefillPPPipeline
The class `PrefillPPPipeline` is an abstract `PPPipeline`. It contains methods that return various prefill PP pipeline statistics. For example:
- `X_prefill_rank(r)` represents the latency in seconds of the transfer of activations from `PrefillPPRank` r to `PrefillPPRank` r+1. Note that this is an abstract method that must be implemented by a concrete `PrefillPPPipeline` such as `AkamaiDeployment`.
- `X_handoff(r)` represents the latency in seconds of the transfer of activations from `PrefillPPRank` r to the corresponding decode rank. Note that this is an abstract method that must be implemented by a concrete `PrefillPPPipeline` such as `AkamaiDeployment`.
- `TTFT` represents the time to first token in seconds. This is a template method that relies on the methods `X_prefill_rank(r)` and `X_handoff(r)` to calculate the total time in seconds from the moment a batch enters `PrefillPPRank` 0, until the entire KV-cache is received by every `DecodePPRank` in the decode PP pipeline.


### DecodePPPipeline
The class `DecodePPPipeline` represents an abstract `PPPipeline`. It contains methods that return various pipeline statistics. For example:
- `X_decode(r)` represents the latency in seconds of the transfer of activations from `DecodePPRank` r to `DecodePPRank` r+1. Note that this is an abstract method that must be implemented by a concrete `DecodePPPipeline` such as `AkamaiDeployment`.
- `TPOT` represents the time per output token in seconds. This is a template method that relies on the method `X_decode(r)` to calculate the total time in seconds from the moment a batch enters `DecodePPRank` 0, until the last `DecodePPRank` transfers out its activations.


### AkamaiDeployment
The class `AkamaiDeployment` represents a physical deployment of a PD disaggregated inference system. It implements the abstract methods of `PrefillPPPipeline` and `DecodePPPipeline` by modeling a physical disaggregated PD deployment with the following topology:
- Prefill Pipeline Server: Each rank of the prefill pipeline is realized as a separate GPU with a PCI interface plugged into the PCI slots of the same server with a PLX switch. The server has a single Infiniband NDX card occupying one of the PCI slots.
- Decode Pipeline Server: Each rank of the decode pipeline is realized as a separate GPU with a PCI interface plugged into the PCI slot of a separate server with a PLX switch. The same server hosts other ranks of other decode PP pipelines plugged into other PCI slots. The server also hosts an Infiniband NDX card plugged into another slot. The Infiniband card has one port configured as Ethernet.


The topology implies the following data transfer paths:
  - The transfer path for the prefill inter-rank activations is GPU A -> PCI lanes A -> PLX -> PCI lanes B -> GPU B
  - The transfer path for the prefill KV-cache share handoff is GPU A -> PCI lanes A -> PLX -> PCI lanes C -> Infiniband card X -> Infiniband switch -> Infiniband card Y -> PCI lanes D -> PLX -> PCI lanes E -> GPU E
  - The transfer path for the decode inter-rank activations is GPU E -> PCI lanes E -> PLX -> Infiniband card Y Ethernet port -> Ethernet switch -> Infiniband card Z Ethernet port -> PCI lanes Z -> PLX -> PCI lanes F -> GPU F

The data transfer paths and topology imply the following contentions:
  - On the prefill pipeline server:
    - PCI lanes A are used to send the inter-rank activations and the KV-cache shares, causing contention.
    - PCI lanes C are used to send the KV-cache shares from multiple prefill GPUs, causing contention.
  - On the decode pipeline server:
    - PCI lanes Z are used to transfer the received KV-cache shares from multiple prefill ranks of other prefill pipelines, causing contention.  However, we can assume that these transfers occur infrequently only during the initialization of a decode pipeline in which GPU F participates.
    - PCI lanes Z are used to transfer the received decode inter-rank activations from decode ranks, causing additional contention.  We can assume that these transfers occur frequently (each time the decode pipeline in which GPU F participates decodes a token) 

