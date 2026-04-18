import gradio as gr
from inference import run

history = []

def chat(user_msg, chat_history):
    global history
    if not user_msg.strip():
        return "", chat_history
    response = run(user_msg, history)
    display_response = response
    if "<tool_call>" in response:
        display_response = f"🔧 Tool Call:\n```json\n{response.replace('<tool_call>','').replace('</tool_call>','').strip()}\n```"
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": display_response})
    return "", chat_history

def clear():
    global history
    history = []
    return []

with gr.Blocks(title="Pocket-Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Pocket-Agent\nOn-device tool-calling assistant.")
    chatbot = gr.Chatbot(height=450, type="messages", label="Conversation")
    msg = gr.Textbox(placeholder="e.g. What's the weather in Karachi?", label="Message")
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")
    send.click(chat, [msg, chatbot], [msg, chatbot])
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear, [], [chatbot])

demo.launch(share=True)
