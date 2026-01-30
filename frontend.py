# frontend.py
import gradio as gr
import requests
import json

API_URL = "http://localhost:8002/ask"

def query_backend(message, selected_patient_id=None):
    params = {"query": message}
    if selected_patient_id:
        params["query"] = f"{message} (Patient ID: {selected_patient_id})"

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def chat_logic(user_message, history, current_candidates_state):
    """
    Main logic driver.
    """
    if history is None:
        history = []

    # 1. Update History immediately
    history.append({"role": "user", "content": user_message})

    # 2. Call API
    data = query_backend(user_message)

    # 3. Handle Errors
    if "error" in data:
        history.append({"role": "assistant", "content": f"❌ Error: {data['error']}"})
        # Return user_message to state so we don't lose it, though error flow ends here
        return history, "", gr.update(visible=False), [], user_message

    # 4. Handle Clarification (Ambiguity)
    if data.get("action") == "CLARIFICATION_NEEDED":
        candidates = data.get("candidates", [])
        options = [f"{c['name']} (ID: {c['id']})" for c in candidates]
        
        history.append({"role": "assistant", "content": data['message']})
        
        return (
            history, 
            "", # Clear the visual input box
            gr.update(visible=True, choices=options, value=None, label="Select Patient"),
            candidates,
            user_message # <--- SAVE THE QUERY TO STATE
        )

    # 5. Standard Answer
    answer = data.get("answer", "No answer received.")
    footer = ""
    if data.get("patient_id"):
        footer = f"\n\n---\n*Focus: {data['patient_id']} | Route: {data.get('route')}*"

    history.append({"role": "assistant", "content": answer + footer})
    
    return history, "", gr.update(visible=False), [], "" # Clear state if done

def on_selection(selection, candidates_state, saved_query, history):
    """
    Uses the SAVED query state, not the empty text box.
    """
    if not selection or not candidates_state:
        return history, gr.update(visible=True)

    selected_id = None
    for c in candidates_state:
        if f"{c['name']} (ID: {c['id']})" == selection:
            selected_id = c['id']
            break
            
    if selected_id:
        # Use saved_query (which contains "is aspirin good...")
        data = query_backend(saved_query, selected_patient_id=selected_id)
        
        answer = data.get("answer", "No answer.")
        footer = f"\n\n---\n*Selected ID: {selected_id}*"
        
        history.append({"role": "assistant", "content": answer + footer})
        
        return history, gr.update(visible=False)
    
    return history, gr.update(visible=True)

# --- UI SETUP ---
with gr.Blocks(title="MediFlow AI") as demo:
    gr.Markdown("# 🏥 MediFlow Clinical Assistant")
    
    # STATES
    candidates_state = gr.State([])
    query_state = gr.State("") # <--- NEW: Remembers the question
    
    chatbot = gr.Chatbot(height=500)
    msg_input = gr.Textbox(placeholder="E.g., 'Is Aspirin safe for Ali Khan?'", label="Query")
    patient_selector = gr.Radio(choices=[], visible=False, label="Multiple Patients Found")
    
    msg_input.submit(
        chat_logic,
        inputs=[msg_input, chatbot, candidates_state],
        outputs=[chatbot, msg_input, patient_selector, candidates_state, query_state]
    )
    
    patient_selector.input(
        on_selection,
        inputs=[patient_selector, candidates_state, query_state, chatbot],
        outputs=[chatbot, patient_selector]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft())