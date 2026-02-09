import sys
from pathlib import Path

# 상위 디렉토리(example)를 Python 경로에 추가하여 형제 모듈 import 가능하게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from dotenv import load_dotenv

from answer import answer_question

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result

def get_text_content(content):
    """Gradio 6.x에서 content가 다양한 형식일 수 있으므로 문자열로 변환"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 리스트 내 문자열 요소들을 결합
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return " ".join(texts) if texts else str(content)
    if isinstance(content, dict) and "text" in content:
        return content["text"]
    return str(content) if content else ""

def chat(history):
    print(f"DEBUG history: {history}")  # 디버깅용
    last_message = get_text_content(history[-1]["content"])
    print(f"DEBUG last_message: {last_message}")  # 디버깅용
    prior = [{"role": msg["role"], "content": get_text_content(msg["content"])} for msg in history[:-1]]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)

def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
