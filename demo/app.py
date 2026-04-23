import gradio as gr
import pandas as pd
import httpx
import asyncio
import json
import os

# Configuration for OpenEnv server
ENV_URL = os.getenv("EPISTEMICOPS_ENV_URL", "http://localhost:8000")

async def fetch_state():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ENV_URL}/state", timeout=5.0)
            return resp.json()
    except Exception as e:
        return {"error": str(e)}

async def reset_env():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{ENV_URL}/reset", json={"scenario_id": "cascading_incident"}, timeout=10.0)
            return resp.json(), "Environment reset to Era 1."
    except Exception as e:
        return {}, f"Failed to reset: {str(e)}"

async def step_env(agent_role: str, action: dict):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{ENV_URL}/step", json={"agent_role": agent_role, "action": action}, timeout=10.0)
            return resp.json(), f"Step executed for {agent_role}."
    except Exception as e:
        return {}, f"Step failed: {str(e)}"

def update_ui(state_data):
    if "error" in state_data:
        return f"**Error connecting to Environment:** {state_data['error']}", "N/A", "N/A", pd.DataFrame(), pd.DataFrame(), "", ""
        
    era_text = f"**Era:** {state_data.get('era_id', 1)} / 5  |  **Phase:** {state_data.get('phase', 'UNKNOWN')}"
    
    # Format Services
    services_html = "<ul>"
    for name, s in state_data.get("services", {}).items():
        status = s.get("status", "STABLE")
        icon = "✅" if status == "STABLE" else "⚠️"
        services_html += f"<li>{icon} {name} ({status})</li>"
    services_html += "</ul>"
    
    # Format Incidents
    incidents = state_data.get("incident_history", [])
    inc_df = pd.DataFrame(incidents) if incidents else pd.DataFrame([{"ID": "None", "Status": "N/A"}])
    
    reward_state = state_data.get("reward_state", {})
    rewards_data = [
        ["R_era_task",    f"{reward_state.get('R_era_task', 0.0):.3f}"],
        ["R_calibration", f"{reward_state.get('R_calibration', 1.0):.2f}×"],
        ["R_teacher_Δ",   f"{reward_state.get('R_teacher_delta', 0.0):.3f}"],
        ["R_legacy_util", f"{reward_state.get('R_legacy_utility', 0.0):.3f}"],
        ["R_leakage",     f"{reward_state.get('R_answer_leakage', 0.0):.3f}"],
        ["TOTAL",         f"{reward_state.get('R_normalized', 0.0):.3f}"],
    ]
    rewards_df = pd.DataFrame(rewards_data, columns=["Component", "Value"])
    
    # For agents, we mock the last actions in the UI for the demo unless we build a full orchestrator
    # Here, we'll fetch the action history if available (which we don't expose in state yet, but could)
    
    return era_text, services_html, inc_df, rewards_df, "Waiting for agent action...", "Waiting for oversight intervention..."

async def on_run_step():
    # In a full demo, this would ping the actual Primary Agent API, then send to env.
    # For now, we simulate a mock tool call that fails to trigger socratic mode.
    action = {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}}
    step_resp, msg = await step_env("primary", action)
    state = await fetch_state()
    
    era_text, services_html, inc_df, rewards_df, _, _ = update_ui(state)
    
    agent_action_str = "call_tool(get_incident_status, INC-2041) -> Failed. Drift suspected."
    oversight_str = "If your tool failed, what assumption did you make about the return type?"
    
    return era_text, services_html, inc_df, rewards_df, agent_action_str, oversight_str

async def on_reset():
    _, msg = await reset_env()
    state = await fetch_state()
    era_text, services_html, inc_df, rewards_df, _, _ = update_ui(state)
    return era_text, services_html, inc_df, rewards_df, "System Initialized.", ""


def create_demo():
    with gr.Blocks(title="EpistemicOps Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 EpistemicOps: SRE Agent Training Environment")
        gr.Markdown("Training an agent on temporal uncertainty, memory compression, and Socratic oversight. **(Live via FastAPI Backend)**")
        
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🌍 World State")
                era_display = gr.Markdown("Connecting...")
                
                gr.Markdown("#### Services")
                service_status = gr.HTML("<ul><li>Connecting...</li></ul>")
                
                gr.Markdown("#### Open Incidents")
                incidents_display = gr.DataFrame(interactive=False)

            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("### 🤖 Agent Actions")
                with gr.Group():
                    gr.Markdown("#### Primary Agent (Student)")
                    agent_action = gr.Textbox(label="Last Action & Reasoning", interactive=False, lines=3)

                with gr.Group():
                    gr.Markdown("#### Oversight Agent (Teacher)")
                    oversight_msg = gr.Textbox(label="Socratic Intervention", interactive=False, lines=2)

            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 📈 Reward Dashboard")
                rewards_display = gr.DataFrame(interactive=False)
                
        with gr.Row():
            play_btn = gr.Button("▶️ Run Next Step", variant="primary")
            reset_btn = gr.Button("🔄 Reset Era")

        play_btn.click(
            on_run_step, 
            outputs=[era_display, service_status, incidents_display, rewards_display, agent_action, oversight_msg]
        )
        reset_btn.click(
            on_reset, 
            outputs=[era_display, service_status, incidents_display, rewards_display, agent_action, oversight_msg]
        )
        
        # Load initial state
        demo.load(
            on_reset,
            outputs=[era_display, service_status, incidents_display, rewards_display, agent_action, oversight_msg]
        )

    return demo

if __name__ == "__main__":
    app = create_demo()
    app.launch()
