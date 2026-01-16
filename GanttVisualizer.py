import plotly.graph_objects as go
from utils_env import show_or_save_plotly_figure

class GanttVisualizer:
    def __init__(self, system, chart_duration=None):
        """
        Initialize the visualizer with the target system configuration.
        Optionally specify the chart's duration (in seconds).
        """
        self.system = system
        self.chart_duration = chart_duration

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

    def generate(self, sim, output_file="gantt_chart.html"):
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
        events.sort(key=lambda x: x["start"])

        # Map resources to Y-axis
        resources = set()
        resource_start_times = {}
        for e in events:
            res, _ = self._parse_resource_from_name(e["name"])
            resources.add(res)
            e["resource"] = res
            if res not in resource_start_times:
                resource_start_times[res] = e["start"]
            else:
                resource_start_times[res] = min(resource_start_times[res], e["start"])
        sorted_resources = sorted(list(resources), key=lambda r: resource_start_times[r], reverse=True)
        y_map = {res: i for i, res in enumerate(sorted_resources)}

        colors = {
            "Compute": "skyblue",
            "Transfer": "orange",
            "Other": "gray"
        }

        fig = go.Figure()
        bar_height = 0.3  # Reduce row height for Gantt chart
        for e in events:
            y = y_map[e["resource"]]
            color = colors.get(e["type"], "gray")
            fig.add_trace(go.Bar(
                x=[e["end"] - e["start"]],
                y=[e["resource"]],
                base=[e["start"]],
                orientation='h',
                marker_color=color,
                name=e["type"],
                hovertext=f"{e['name']}<br>Start: {e['start']:.3f}s<br>End: {e['end']:.3f}s",
                hoverinfo="text",
                width=bar_height
            ))

        fig.update_layout(
            barmode='stack',
            xaxis_title="Time (s)",
            yaxis=dict(
                categoryorder='array',
                categoryarray=sorted_resources,
                tickfont=dict(size=12),
                automargin=True,
                dtick=1,
                # Removed invalid categorygap property
            ),
            title=f"Inference System Gantt Chart (M={self.system.M}, T={self.system.T})",
            showlegend=False,
            height=30 * max(6, len(sorted_resources)),
            margin=dict(l=120, r=40, t=60, b=40)
        )
        if self.chart_duration is not None:
            fig.update_xaxes(range=[0, self.chart_duration])
        show_or_save_plotly_figure(fig, output_file)

