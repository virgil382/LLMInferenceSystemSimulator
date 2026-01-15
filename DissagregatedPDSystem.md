# Disaggregated PD Inference System

## 1. System Parameters
- **N**: Batch Size (Number of concurrent requests).
- **T**: Context Length (Tokens per request).
- **M**: Prefill Chunk Size (Tokens processed per pipeline step).
- **PP_prefill**: Pipeline Parallelism degree for the Prefill Cluster.
- **PP_decode**: Pipeline Parallelism degree for the Decode Cluster.
- **Num_IB_Cards**: Number of InfiniBand NICs in the Prefill Server.

---

## 2. Hardware Topology

### A. Prefill Server (Single Node)
- **GPUs**: $PP_{prefill}$ GPUs (NVIDIA/AMD) connected to a central PLX Switch.
- **Interconnect**: Local PCIe fabric through the PLX Switch.
- **External I/O**: $Num\_IB\_Cards$ InfiniBand NICs connected to the same PLX Switch.
- **Network**: All IB NICs connect to a central **InfiniBand Switch**.

### B. Decode Cluster (Distributed)
- **Servers**: $PP_{decode}$ individual servers.
- **GPUs per Server**: 1 Active Decode GPU.
- **Internal I/O**: 
    - 1 InfiniBand NIC (for receiving KV-cache/activations from Prefill).
    - 1 Ethernet NIC (for inter-rank decode activations).
- **Networks**: 
    - All IB NICs connect to the **InfiniBand Switch**.
    - All Ethernet NICs connect to the **Ethernet Switch**.

---

## 3. Segment Definitions
| Segment Group | Notation | Description |
| :--- | :--- | :--- |
| **Prefill GPU PCIe** | `GPU_P[i]_PCI` | PCIe lanes between Prefill GPU $i$ and PLX. |
| **Prefill IB PCIe** | `IB_NIC_P[k]_PCI` | PCIe lanes between Prefill IB NIC $k$ and PLX. |
| **Prefill IB Cable**| `IB_Cable_P[k]` | Physical cable: Prefill IB NIC $k \to$ IB Switch. |
| **Decode IB Cable** | `IB_Cable_D[j]` | Physical cable: IB Switch $\to$ Decode Server $j$. |
| **Decode GPU PCIe** | `GPU_D[j]_PCI` | PCIe lanes between Decode GPU $j$ and local PLX. |
| **Decode Eth PCIe** | `Eth_NIC_D[j]_PCI` | PCIe lanes between Decode Eth NIC $j$ and local PLX. |
| **Decode Eth Cable**| `Eth_Cable_D[j]` | Physical cable: Decode Eth NIC $j \to$ Eth Switch. |

---

## 4. Communication Paths

### Path 1: Prefill Inter-GPU (Local Server)
*Used for moving activations between GPUs $i$ and $i+1$ within the prefill node.*
> **Path**: `[GPU_P[i]_PCI, GPU_P[i+1]_PCI]`

### Path 2: Prefill-to-Decode Handoff (IB Fabric)
*Used to transport prefill shares from Prefill GPU $i$ to the corresponding Decode Server $i$.*
*Assumes IB NIC mapping: $k = i \pmod{Num\_IB\_Cards}$.*
> **Path**: `[GPU_P[i]_PCI, IB_NIC_P[k]_PCI, IB_Cable_P[k], IB_Cable_D[i], GPU_D[i]_PCI]`

### Path 3: Decode Inter-GPU (Ethernet Fabric)
*Used for moving activations between decode GPUs $j$ and $j+1$.*
> **Path**: `[GPU_D[j]_PCI, Eth_NIC_D[j]_PCI, Eth_Cable_D[j], Eth_Cable_D[j+1], Eth_NIC_D[j+1]_PCI, GPU_D[j+1]_PCI]`# Disaggregated PD Inference System Specification

## 1. System Parameters
- **N**: Batch Size (Number of concurrent requests).
- **T**: Context Length (Tokens per request).
- **M**: Prefill Chunk Size (Tokens processed per pipeline step).
- **PP_prefill**: Pipeline Parallelism degree for the Prefill Cluster.
- **PP_decode**: Pipeline Parallelism degree for the Decode Cluster.
- **Num_IB_Cards**: Number of InfiniBand NICs in the Prefill Server.

---

## 2. Hardware Topology

### A. Prefill Server (Single Node)
- **GPUs**: $PP_{prefill}$ GPUs (NVIDIA/AMD) connected to a central PLX Switch.
- **Interconnect**: Local PCIe fabric through the PLX Switch.
- **External I/O**: $Num\_IB\_Cards$ InfiniBand NICs connected to the same PLX Switch.
- **Network**: All IB NICs connect to a central **InfiniBand Switch**.

### B. Decode Cluster (Distributed)
- **Servers**: $PP_{decode}$ individual servers.
- **GPUs per Server**: 1 Active Decode GPU.
- **Internal I/O**: 
    - 1 InfiniBand NIC (for receiving KV-cache/activations from Prefill).
    - 1 Ethernet NIC (for inter-rank decode activations).
- **Networks**: 
    - All IB NICs connect to the **InfiniBand Switch**.
    - All Ethernet NICs connect to the **Ethernet Switch**.

---

## 3. Segment Definitions
| Segment Group | Notation | Description |
| :--- | :--- | :--- |
| **Prefill GPU PCIe** | `GPU_P[i]_PCI` | PCIe lanes between Prefill GPU $i$ and PLX. |
| **Prefill IB PCIe** | `IB_NIC_P[k]_PCI` | PCIe lanes between Prefill IB NIC $k$ and PLX. |
| **Prefill IB Cable**| `IB_Cable_P[k]` | Physical cable: Prefill IB NIC $k \to$ IB Switch. |
| **Decode IB Cable** | `IB_Cable_D[j]` | Physical cable: IB Switch $\to$ Decode Server $j$. |
| **Decode GPU PCIe** | `GPU_D[j]_PCI` | PCIe lanes between Decode GPU $j$ and local PLX. |
| **Decode Eth PCIe** | `Eth_NIC_D[j]_PCI` | PCIe lanes between Decode Eth NIC $j$ and local PLX. |
| **Decode Eth Cable**| `Eth_Cable_D[j]` | Physical cable: Decode Eth NIC $j \to$ Eth Switch. |

---

## 4. Communication Paths

### Path 1: Prefill Inter-Rank (Local Server)
*Used for moving activations between pipeline stages $i$ and $i+1$ within the prefill node.*
> **Path**: `[GPU_P[i]_PCI, GPU_P[i+1]_PCI]`

### Path 2: Prefill-to-Decode Handoff (IB Fabric)
*Used to transport prefill shares from Prefill GPU $i$ to the corresponding Decode Server $i$.*
*Assumes IB NIC mapping: $k = i \pmod{Num\_IB\_Cards}$.*
> **Path**: `[GPU_P[i]_PCI, IB_NIC_P[k]_PCI, IB_Cable_P[k], IB_Cable_D[i], GPU_D[i]_PCI]`

### Path 3: Decode Inter-Rank (Ethernet Fabric)
*Used for moving activations between disaggregated decode stages $j$ and $j+1$.*
> **Path**: `[GPU_D[j]_PCI, Eth_NIC_D[j]_PCI, Eth_Cable_D[j], Eth_Cable_D[j+1], Eth_NIC_D[j+1]_PCI, GPU_D[j+1]_PCI]`