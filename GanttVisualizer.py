import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class GanttVisualizer:
    def __init__(self, system):
        """
        Initialize the visualizer with the target system configuration.
        """
        self.system = system

    def _parse_resource_from_name(self, name):
        # Heuristic to assign a 'row' for the Gantt chart
        if "P_Rank" in name:
            parts = name.split("_")
            rank = int(parts[2])
            return f"Prefill Rank {rank} (Compute)", "Compute"
        elif "D_Rank" in name:
            parts = name.split("_")
            rank = int(parts[2])
            return f"Decode Rank {rank} (Compute)", "Compute"
        elif "Handoff" in name:
            parts = name.split("_")
            rank = int(parts[2])
            return f"Prefill Rank {rank} (Tx Handoff)", "Transfer"
        elif "Prefill_Act" in name:
            parts = name.split("_")
            rank = int(parts[3])
            return f"Prefill Rank {rank} (Tx Activation)", "Transfer"
        elif "Decode_Act" in name:
            parts = name.split("_")
            rank = int(parts[3])
            return f"Decode Rank {rank} (Tx Activation)", "Transfer"
        return "Other", "Other"

    def generate(self, sim, output_file="gantt_chart.png"):
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Collect all events
        events = []
        
        for job in sim.completed_compute:
            events.append({
                "name": job.name,
                "start": job.start_time,
                "end": job.end_time,
                "type": "Compute"
            })
            
        for batch in sim.completed_batches:
            events.append({
                "name": batch.name,
                "start": batch.start_time,
                "end": batch.end_time,
                "type": "Transfer"
            })
            
        # Sort events by time
        events.sort(key=lambda x: x["start"])
        
        # Map resources to Y-axis
        resources = set()
        resource_start_times = {}

        for e in events:
            res, _ = self._parse_resource_from_name(e["name"])
            resources.add(res)
            e["resource"] = res
            
            # Track start time for sorting
            if res not in resource_start_times:
                resource_start_times[res] = e["start"]
            else:
                resource_start_times[res] = min(resource_start_times[res], e["start"])

        # Sort resources by their earliest start time (Reverse=True puts earliest at top)
        sorted_resources = sorted(list(resources), key=lambda r: resource_start_times[r], reverse=True)
        y_map = {res: i for i, res in enumerate(sorted_resources)}
        
        colors = {
            "Compute": "skyblue",
            "Transfer": "orange",
            "Other": "gray"
        }
        
        for e in events:
            y = y_map[e["resource"]]
            width = e["end"] - e["start"]
            color = colors.get(e["type"], "gray")
            
            # Draw bar
            ax.barh(y, width, left=e["start"], height=0.6, align='center', color=color, edgecolor='black', alpha=0.8)
            
            # Add labels for larger blocks?
            # if width > (sim.current_time * 0.05):
            #    ax.text(e["start"] + width/2, y, e["name"], ha='center', va='center', fontsize=8, color='black')

        # Formatting
        ax.set_yticks(range(len(sorted_resources)))
        ax.set_yticklabels(sorted_resources)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Inference System Gantt Chart (M={self.system.M}, T={self.system.T})")
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Legend
        handles = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
        ax.legend(handles=handles)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Chart saved to {output_file}")

