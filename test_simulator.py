import unittest
from simulator import CommChannel, CommNetworkSimulator, DataBatch, ComputeJob

class TestCommNetworkSimulator(unittest.TestCase):
    def test_three_custom_commlinks_bandwidth(self):
        """
        Test scenario: Data transfer over three custom CommLinks with controlled bandwidths.

        Setup:
        - Three custom CommChannel objects are created with bandwidths of 10, 20, and 30 Gbps, and latencies of 10 us each.
        - A DataBatch of 1 GB is sent through all three links in sequence.
        - The CommNetworkSimulator is used to simulate the transfer.

        Verification:
        - The bottleneck bandwidth is the lowest (10 Gbps), so the transfer rate is limited by link1.
        - The expected total time is the sum of all link latencies plus the transmission time at the bottleneck bandwidth.
        - The test asserts that the batch's end_time matches the expected time within two decimal places.

        Expected result:
        - The simulation should complete with the batch's end_time matching the calculated expected time.
        """
        # Create three custom CommChannel objects with controlled bandwidths
        class CustomCommChannel(CommChannel):
            def __init__(self, name, bandwidth_gbps, latency_us):
                self._name = name
                self._attrs = {
                    "bw_gbps": (bandwidth_gbps, bandwidth_gbps),
                    "lat_us": (latency_us, latency_us)
                }
        
        # Bandwidths in Gbps
        link1 = CustomCommChannel("Link1", 10, 10)  # 10 Gbps
        link2 = CustomCommChannel("Link2", 20, 10)  # 20 Gbps
        link3 = CustomCommChannel("Link3", 30, 10)  # 30 Gbps
        
        # Create a DataBatch that uses all three links in its path
        batch_size = 1e9  # 1 GB
        batch = DataBatch(name="test_batch", size_bytes=batch_size, path_channels=[link1, link2, link3])
        
        # Create and run the simulator
        sim = CommNetworkSimulator()
        sim.add_batch(batch)
        sim.run(system=DummySystem(), t_delta=0.001)
        
        # The bottleneck bandwidth is the lowest (10 Gbps)
        expected_time = batch_size / link1.bandwidth_bps + link1.latency_s + link2.latency_s + link3.latency_s
        
        # Ensure batch.end_time is not None before assertion
        self.assertIsNotNone(batch.end_time)
        end_time: float = batch.end_time  # type: ignore[assignment]
        self.assertAlmostEqual(end_time, float(expected_time), places=2)

    def test_pingpong_two_commlinks_with_compute(self):
        """
        Test scenario: Ping-pong a DataBatch between two nodes using two custom CommLinks, 
        with compute on both nodes before each transmission.

        Setup:
        - Two custom CommChannel objects are created: linkA (10 Gbps, 10 us) and linkB (20 Gbps, 20 us).
        - Node A performs compute (0.5s), then sends a 2 GB DataBatch to Node B via linkA.
        - Node B performs compute (0.3s), then sends a 2 GB DataBatch back to Node A via linkB.
        - The PingPongSystem class orchestrates the sequence of compute and transfer events.

        Verification:
        - The test checks the start and end times of each completed compute and batch in the simulator.
        - It asserts that each event starts immediately after the previous one ends, and that durations 
          match expected values based on bandwidth and latency.
        - Specifically, it checks:
            - computeA: starts at 0, ends at 0.5s
            - batch1: starts at 0.5s, ends at 0.5s + linkA latency + transmission time
            - computeB: starts after batch1, ends after its duration
            - batch2: starts after computeB, ends after linkB latency + transmission time

        Expected result:
        - All event times should match the calculated expected values within two decimal places.
        """
        # Custom CommChannel for controlled bandwidths
        class CustomCommChannel(CommChannel):
            def __init__(self, name, bandwidth_gbps, latency_us):
                self._name = name
                self._attrs = {
                    "bw_gbps": (bandwidth_gbps, bandwidth_gbps),
                    "lat_us": (latency_us, latency_us)
                }

        # Create two links
        linkA = CustomCommChannel("LinkA", 10, 10)  # 10 Gbps
        linkB = CustomCommChannel("LinkB", 20, 20)  # 20 Gbps

        # Simulate two nodes, each with compute before sending/receiving
        compute_duration_A = 0.5  # seconds
        compute_duration_B = 0.3  # seconds
        batch_size = 2e9  # 2 GB

        # Create simulator
        sim = CommNetworkSimulator()

        # Node A compute
        computeA = ComputeJob(name="computeA", duration_s=compute_duration_A)
        sim.add_compute(computeA)

        # Node A sends DataBatch to Node B via linkA
        batch1 = DataBatch(name="batch1", size_bytes=batch_size, path_channels=[linkA])
        # Node B compute
        computeB = ComputeJob(name="computeB", duration_s=compute_duration_B)

        # Node B sends DataBatch back to Node A via linkB
        batch2 = DataBatch(name="batch2", size_bytes=batch_size, path_channels=[linkB])

        # Custom system to trigger next steps
        class PingPongSystem:
            def __init__(self):
                self.sim = sim
                self.computeB_added = False
                self.batch2_added = False
            def on_data_transfer_complete(self, sim, batch):
                if batch.name == "batch1" and not self.computeB_added:
                    sim.add_compute(computeB)
                    self.computeB_added = True
                elif batch.name == "batch2":
                    pass
                return None
            def on_compute_complete(self, sim, compute):
                if compute.name == "computeA" and not self.batch1_added:
                    sim.add_batch(batch1)
                    self.batch1_added = True
                elif compute.name == "computeB" and not self.batch2_added:
                    sim.add_batch(batch2)
                    self.batch2_added = True

        system = PingPongSystem()
        system.batch1_added = False
        sim.run(system=system, t_delta=0.001)

        # Verification using completed batches and computes
        # Get completed computes
        completed_computes = {c.name: c for c in sim.completed_compute}
        completed_batches = {b.name: b for b in sim.completed_batches}

        # computeA: starts at 0, ends at 0.5
        self.assertAlmostEqual(completed_computes["computeA"].start_time, 0.0, places=2)
        self.assertAlmostEqual(completed_computes["computeA"].end_time, compute_duration_A, places=2)
        # batch1: starts after computeA, ends after latency+transmission
        expected_batch1_start = compute_duration_A
        expected_batch1_end = expected_batch1_start + linkA.latency_s + batch_size / linkA.bandwidth_bps
        self.assertAlmostEqual(completed_batches["batch1"].start_time, expected_batch1_start, places=2)
        self.assertAlmostEqual(completed_batches["batch1"].end_time, expected_batch1_end, places=2)
        # computeB: starts after batch1
        expected_computeB_start = expected_batch1_end
        expected_computeB_end = expected_computeB_start + compute_duration_B
        self.assertAlmostEqual(completed_computes["computeB"].start_time, expected_computeB_start, places=2)
        self.assertAlmostEqual(completed_computes["computeB"].end_time, expected_computeB_end, places=2)
        # batch2: starts after computeB
        expected_batch2_start = expected_computeB_end
        expected_batch2_end = expected_batch2_start + linkB.latency_s + batch_size / linkB.bandwidth_bps
        self.assertAlmostEqual(completed_batches["batch2"].start_time, expected_batch2_start, places=2)
        self.assertAlmostEqual(completed_batches["batch2"].end_time, expected_batch2_end, places=2)

class DummySystem:
    def on_data_transfer_complete(self, sim, batch):
        return None
    def on_compute_complete(self, sim, compute):
        pass

if __name__ == "__main__":
    unittest.main()
