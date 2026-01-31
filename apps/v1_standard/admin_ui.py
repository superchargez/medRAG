# admin_frontend.py
import gradio as gr
import requests
import json

API_URL = "http://localhost:8002/ingest_record"

def submit_record(p_id, p_name, cnic, phone, date, condition, note):
    
    if not p_name or not note:
        return "❌ Error: Name and Note are required."

    payload = {
        "patient_id": p_id.strip(),
        "patient_name": p_name.strip(),
        "gov_id": cnic.strip(),
        "contact_number": phone.strip(),
        "date": date,
        "condition": condition.strip(),
        "note_content": note.strip()
    }

    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()
        
        if data.get("status") == "success":
            fid = data.get("patient_id")
            # Generate the "Card" text
            card_text = (
                f"✅ **RECORD SAVED**\n"
                f"--------------------------\n"
                f"👤 Name: {p_name}\n"
                f"🆔 System ID: `{fid}`  <-- (QR Code Data)\n"
                f"💳 CNIC: {cnic}\n"
                f"📞 Phone: {phone}\n"
                f"--------------------------\n"
                f"Use `{fid}` or `{cnic}` or `{phone}` to find this patient later."
            )
            return card_text
        else:
            return f"⛔ ERROR: {data.get('message')}"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

# --- UI ---
with gr.Blocks(title="MediFlow Admin", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinic Reception & Data Entry")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Patient Identity")
            p_id = gr.Textbox(label="System ID (PT-XXX)", placeholder="Leave blank for New Patient")
            p_name = gr.Textbox(label="Full Name")
            cnic = gr.Textbox(label="CNIC / Gov ID", placeholder="e.g. 42101-1234567-1")
            phone = gr.Textbox(label="Phone Number", placeholder="e.g. 0300-1234567")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Visit Details")
            date = gr.Textbox(label="Date", value="2024-01-01")
            condition = gr.Textbox(label="Condition", placeholder="e.g. Flu")
            note = gr.TextArea(label="Clinical Note", lines=6)
    
    btn = gr.Button("💾 Register / Save Visit", variant="primary")
    status = gr.Markdown()
    
    btn.click(
        submit_record,
        inputs=[p_id, p_name, cnic, phone, date, condition, note],
        outputs=[status]
    )

if __name__ == "__main__":
    demo.launch(server_port=7861)