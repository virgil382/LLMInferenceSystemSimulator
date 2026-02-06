import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from simulator import CommNetworkSimulator, GPU, LLM
from DisaggregatedPDSystemPP import DisaggregatedPDSystemPP
import pandas as pd


class Gantt6DVisualizer:
    def __init__(self, base_config, slider_ranges=None, height=500):
        self.base_config = base_config
        # slider_ranges is a dict with keys: 'pp', 'm', 't', each value is a dict with min, max, step, marks_step
        default_ranges = {
            'pp': {'min': 1, 'max': 32, 'step': 1, 'marks_step': 1},
            'm': {'min': 32, 'max': 2048, 'step': 32, 'marks_step': 256},
            't': {'min': 1024, 'max': 16384, 'step': 1024, 'marks_step': 2048},
            'n': {'min': 1, 'max': 128, 'step': 1, 'marks_step': 8},
            'time_range': {'min': 0.0, 'max': 10.0, 'step': 0.01, 'marks_step': 1, 'default': [0.0, 2.0]},
        }
        self.slider_ranges = default_ranges if slider_ranges is None else {**default_ranges, **slider_ranges}
        self.height = height
        self.app = dash.Dash(__name__)
        self.last_ttds = 0.0
        self.last_tpot = 0.0
        # List of available GPU names from GPU class, sorted by name
        self.available_gpus = sorted(GPU.available_gpus())
        self._setup_layout()
        self._setup_callbacks()

    def _slider_label(self, label, value_id):
        return html.Label([
            label + ": ",
            html.Span(id=value_id, style={"fontWeight": "bold", "marginLeft": "4px"})
        ])

    def _setup_layout(self):
        def _gpu_name(val):
            from simulator import GPU
            if isinstance(val, GPU):
                return val.name
            return val if isinstance(val, str) else self.available_gpus[0]

        self.app.layout = html.Div([
            html.H2("Dynamic 6D Gantt Visualizer"),

            html.Div([
                html.Div([
                    html.Label("Prefill GPU Type"),
                    dcc.Dropdown(
                        id='prefill-gpu-dropdown',
                        options=[{"label": name, "value": name} for name in self.available_gpus],
                        value=_gpu_name(self.base_config.get('prefill_gpu', self.available_gpus[0])),
                        clearable=False
                    )
                ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    html.Label("Decode GPU Type"),
                    dcc.Dropdown(
                        id='decode-gpu-dropdown',
                        options=[{"label": name, "value": name} for name in self.available_gpus],
                        value=_gpu_name(self.base_config.get('decode_gpu', self.available_gpus[0])),
                        clearable=False
                    )
                ], style={'width': '45%', 'display': 'inline-block'}),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Div([
                    self._slider_label("PP Degree (Pipeline Parallelism)", "pp-value"),
                    dcc.Slider(
                        self.slider_ranges['pp']['min'],
                        self.slider_ranges['pp']['max'],
                        step=self.slider_ranges['pp']['step'],
                        value=self.base_config['pp_degree'],
                        id='pp-slider',
                        marks={i: str(i) for i in range(self.slider_ranges['pp']['min'], self.slider_ranges['pp']['max']+1, self.slider_ranges['pp']['marks_step'])}
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    self._slider_label("N (Batch Size)", "n-value"),
                    dcc.Slider(
                        self.slider_ranges['n']['min'],
                        self.slider_ranges['n']['max'],
                        step=self.slider_ranges['n']['step'],
                        value=self.base_config.get('N', 1),
                        id='n-slider',
                        marks={i: str(i) for i in range(self.slider_ranges['n']['min'], self.slider_ranges['n']['max']+1, self.slider_ranges['n']['marks_step'])}
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    self._slider_label("T (Sequence Length)", "t-value"),
                    dcc.Slider(
                        self.slider_ranges['t']['min'],
                        self.slider_ranges['t']['max'],
                        step=self.slider_ranges['t']['step'],
                        value=self.base_config.get('T', 8192),
                        id='t-slider',
                        marks={i: str(i) for i in range(self.slider_ranges['t']['min'], self.slider_ranges['t']['max']+1, self.slider_ranges['t']['marks_step'])}
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    self._slider_label("M (Chunk Size)", "m-value"),
                    dcc.Slider(
                        self.slider_ranges['m']['min'],
                        self.slider_ranges['m']['max'],
                        step=self.slider_ranges['m']['step'],
                        value=self.base_config.get('M', 128),
                        id='m-slider',
                        marks={i: str(i) for i in range(self.slider_ranges['m']['min'], self.slider_ranges['m']['max']+1, self.slider_ranges['m']['marks_step'])}
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),

            ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'marginBottom': '20px'}),

            dcc.Graph(id='gantt-plot', style={'height': f'{self.height}px'}),
            html.Div([
                html.Label([
                    "Time Range: ",
                    html.Span(id="time-range-value", style={"fontWeight": "bold", "marginLeft": "4px"})
                ]),
                dcc.RangeSlider(
                    id='time-range-slider',
                    min=self.slider_ranges['time_range']['min'],
                    max=self.slider_ranges['time_range']['max'],
                    step=self.slider_ranges['time_range']['step'],
                    value=self.slider_ranges['time_range']['default'],
                    allowCross=False,
                    marks={i: str(i) for i in range(int(self.slider_ranges['time_range']['min']), int(self.slider_ranges['time_range']['max'])+1, self.slider_ranges['time_range']['marks_step'])},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginTop': '0px'})
        ])

    def _parse_resource_from_name(self, name):
        # Reusing your existing heuristic
        if "P_Rank" in name:
            rank = name.split("_")[2]
            return f"Prefill Rank {rank} (Compute)", "Compute"
        elif "D_Rank" in name:
            rank = name.split("_")[2]
            return f"Decode Rank {rank} (Compute)", "Compute"
        elif "Handoff" in name:
            rank = name.split("_")[2]
            return f"Prefill Rank {rank} (Tx Handoff)", "Transfer"
        elif "Prefill_Act" in name:
            rank = name.split("_")[3]
            return f"Prefill Rank {rank} (Tx Activation)", "Transfer"
        elif "Decode_Act" in name:
            rank = name.split("_")[3]
            return f"Decode Rank {rank} (Tx Activation)", "Transfer"
        return "Other", "Other"

    def _setup_callbacks(self):
        @self.app.callback(
            Output('gantt-plot', 'figure'),
            Output('pp-value', 'children'),
            Output('n-value', 'children'),
            Output('t-value', 'children'),
            Output('m-value', 'children'),
            Output('time-range-value', 'children'),
            Input('pp-slider', 'value'),
            Input('n-slider', 'value'),
            Input('t-slider', 'value'),
            Input('m-slider', 'value'),
            Input('time-range-slider', 'value'),
            Input('prefill-gpu-dropdown', 'value'),
            Input('decode-gpu-dropdown', 'value')
        )
        def update_chart(pp, n, t, m, time_range, prefill_gpu_name, decode_gpu_name):
            # 1. Update config and re-run simulation
            config = self.base_config.copy()
            config.update({"pp_degree": pp, "M": m, "T": t, "N": n})
            # Set GPU objects from dropdowns
            config['prefill_gpu'] = GPU.from_name(prefill_gpu_name)
            config['decode_gpu'] = GPU.from_name(decode_gpu_name)

            try:
                pd_system = DisaggregatedPDSystemPP(**config)
                sim = CommNetworkSimulator()
                pd_system.start(sim)
                sim.run(pd_system)

                # Query pd_system for TTDS and TPOT and save them in members for later use
                self.last_ttds = pd_system.calculate_ttds(sim)
                self.last_tpot = pd_system.calculate_tpot(sim)

            except Exception as e:
                # Handle VRAM overflows or invalid configs gracefully
                return go.Figure().update_layout(title=f"Invalid Configuration: {str(e)}"), pp, n, t, m, f"[{0.0}, {2.0}]"

            # 2. Extract events
            events = []
            for job in sim.completed_compute:
                events.append({"name": job.name, "start": job.start_time, "end": job.end_time, "type": "Compute"})
            for batch in sim.completed_batches:
                events.append({"name": batch.name, "start": batch.start_time, "end": batch.end_time, "type": "Transfer"})

            if not events:
                return go.Figure()

            # 3. Process Y-Axis mapping
            resources = {}
            for e in events:
                res, _ = self._parse_resource_from_name(e["name"])
                e["resource"] = res
                resources[res] = min(resources.get(res, float('inf')), e["start"])

            sorted_res = sorted(resources.keys(), key=lambda r: resources[r], reverse=True)
            
            # 4. Build Figure
            fig = go.Figure()
            colors = {"Compute": "skyblue", "Transfer": "orange", "Other": "gray"}

            for e in events:
                fig.add_trace(go.Bar(
                    x=[e["end"] - e["start"]],
                    y=[e["resource"]],
                    base=[e["start"]],
                    orientation='h',
                    marker_color=colors.get(e["type"], "gray"),
                    hovertext=f"{e['name']}<br>Duration: {e['end']-e['start']:.4f}s",
                    hoverinfo="text",
                    width=0.4,
                    showlegend=False
                ))

            # Ensure ttds and tpot are always defined for the title, even if not drawing lines
            ttds = self.last_ttds
            tpot = self.last_tpot
            ttft = ttds + tpot

            # Add vertical lines for TTDS and TTFT if valid
            if ttds > 0:
                fig.add_vline(
                    x=ttds,
                    line_dash="dot",
                    line_color="green",
                    annotation_text="TTDS",
                    annotation_position="top left"
                )
            if tpot > 0:
                fig.add_vline(
                    x=ttft,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="TTFT (TTDS + TPOT)",
                    annotation_position="top right"
                )
                
            # Draw a double-headed arrow between TTDS and TTFT, labeled TPOT
            y_arrow = sorted_res[0] if sorted_res else 0
            # Draw arrow at the top row (first resource)
            top_resource = sorted_res[-1] if sorted_res else 0
            fig.add_annotation(
                x=ttft,
                y=top_resource,
                text="",
                showarrow=True,
                arrowhead=2,
                arrowside="end+start",
                axref="x",
                ayref="y",
                ax=ttds,
                ay=top_resource,
                xref="x",
                yref="y",
                arrowwidth=1,
                arrowcolor="purple",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="purple"
            )

            # Place label above the middle of the arrow
            fig.add_annotation(
                x=(ttds + ttft) / 2,
                y=top_resource,
                text="TPOT",
                showarrow=False,
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="bottom",
                font=dict(color="black", size=12),
                align="center"
            )


            fig.update_layout(
                title=f"Live Gantt: PP={pp}, N={n}, T={t}, M={m} | TTDS={ttds:.4f}s | TPOT={tpot:.6f}s | TTFT={ttft:.6f}s",
                xaxis_title="Time (s)",
                xaxis=dict(range=time_range),
                yaxis=dict(categoryorder='array', categoryarray=sorted_res, automargin=True),
                height=self.height,
                barmode='stack',
                margin=dict(l=150)
            )
            return fig, pp, n, t, m, f"[{time_range[0]}, {time_range[1]}]"

    def run(self, debug=False, port=8050, host="127.0.0.1"):
        # Using threading or a separate process is possible, 
        # but for main.py we usually run it at the end.
        self.app.run(debug=debug, port=port, host=host)
        