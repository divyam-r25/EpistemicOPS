import gradio as gr
import pandas as pd
import json

# This is a stub for the Gradio UI demonstrating the panels

def create_demo():
    with gr.Blocks(title="EpistemicOps Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 EpistemicOps: SRE Agent Training Environment")
        gr.Markdown("Training an agent on temporal uncertainty, memory compression, and Socratic oversight.")
        
        with gr.Row():
            # Panel 1: World State
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🌍 World State")
                era_display = gr.Markdown("**Era:** 3 / 5  |  **Phase:** SOCRATIC_RECOVERY")
                
                gr.Markdown("#### Services")
                service_status = gr.HTML(
                    "<ul>"
                    "<li>✅ metrics-api (STABLE)</li>"
                    "<li>⚠️ incident-api (DRIFTED: status is now string)</li>"
                    "<li>✅ deploy-api (STABLE)</li>"
                    "</ul>"
                )
                
                gr.Markdown("#### Open Incidents")
                incidents_display = gr.DataFrame(
                    pd.DataFrame([{"ID": "INC-2089", "Severity": "P2", "Status": "OPEN"}]),
                    interactive=False
                )

            # Panel 2: Agent Actions
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("### 🤖 Agent Actions")
                
                with gr.Group():
                    gr.Markdown("#### Primary Agent (Student)")
                    agent_action = gr.Textbox(
                        label="Step 17",
                        value="call_tool(get_incident_status, INC-2089) -> 200 OK {status: 'RESOLVED'}",
                        interactive=False
                    )
                    agent_thought = gr.Textbox(
                        label="Reasoning Trace",
                        value="Wait, status 'RESOLVED' evaluated to False when checking `status == 2`. Did the schema change?",
                        interactive=False,
                        lines=2
                    )

                with gr.Group():
                    gr.Markdown("#### Oversight Agent (Teacher)")
                    oversight_msg = gr.Textbox(
                        label="Socratic Intervention",
                        value="What assumption did you make about the type of the 'status' field?",
                        interactive=False
                    )

            # Panel 3: Reward Dashboard
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 📈 Reward Dashboard")
                
                rewards_data = [
                    ["R_era_task", "0.75"],
                    ["R_calibration", "1.3x"],
                    ["R_teacher_Δ", "0.60"],
                    ["R_legacy_util", "0.45"],
                    ["R_leakage", "-0.30"]
                ]
                rewards_df = pd.DataFrame(rewards_data, columns=["Component", "Value"])
                gr.DataFrame(rewards_df, interactive=False)
                
                gr.Markdown("### TOTAL: 1.50 / 3.50 (43%)")
                
                gr.Markdown("#### Metrics")
                gr.Checkbox(value=True, label="Drift Detected", interactive=False)
                gr.Slider(minimum=0, maximum=1, value=0.82, label="Legacy Quality", interactive=False)
                
        with gr.Row():
            play_btn = gr.Button("▶️ Run Next Step", variant="primary")
            reset_btn = gr.Button("🔄 Reset Era")

    return demo

if __name__ == "__main__":
    app = create_demo()
    app.launch()
