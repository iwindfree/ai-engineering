"""
TechMall 고객 상담 챗봇

RAG 기반의 고객 상담 챗봇 웹 애플리케이션입니다.
Gradio를 사용하여 대화형 UI를 제공합니다.
"""

import gradio as gr
from dotenv import load_dotenv

from src.rag_chain import generate_answer

# 환경 변수 로드
load_dotenv(override=True)


def render_source_panel(documents: list) -> str:
    """
    검색된 문서들을 HTML 형식으로 렌더링합니다.

    Args:
        documents: 검색된 문서 리스트

    Returns:
        HTML 형식의 문자열
    """
    if not documents:
        return "<p style='color: #888;'>검색된 참고 자료가 없습니다.</p>"

    html_parts = ["<h3 style='color: #2563eb;'>📚 참고 자료</h3>"]

    for idx, doc in enumerate(documents, 1):
        category = doc.metadata.get("category", "기타")
        filename = doc.metadata.get("filename", "unknown")
        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content

        html_parts.append(f"""
        <div style='margin-bottom: 15px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #2563eb;'>
            <div style='font-size: 0.85em; color: #64748b; margin-bottom: 6px;'>
                <span style='background: #e0e7ff; padding: 2px 8px; border-radius: 4px;'>{category}</span>
                <span style='margin-left: 8px;'>{filename}</span>
            </div>
            <div style='font-size: 0.95em; color: #334155; line-height: 1.5;'>{content_preview}</div>
        </div>
        """)

    return "\n".join(html_parts)


def handle_chat(message: str, chat_history: list) -> tuple[str, list, str]:
    """
    채팅 메시지를 처리하고 응답을 생성합니다.
    """
    if not message.strip():
        return "", chat_history, ""

    # RAG 파이프라인 실행 (chat_history는 이미 dict 형식)
    answer_text, source_docs = generate_answer(message, chat_history)

    # 대화 이력에 추가 (dict 형식)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer_text})

    # 참고 자료 패널 렌더링
    source_html = render_source_panel(source_docs)

    return "", chat_history, source_html


def create_ui() -> gr.Blocks:
    """Gradio UI를 생성합니다."""
    with gr.Blocks(title="TechMall 고객 상담") as ui:
        # 헤더
        gr.Markdown("""
        # 🛒 TechMall 고객 상담 챗봇
        제품, 배송, 교환/환불, 보증 등 궁금한 점을 물어보세요!
        """)

        with gr.Row():
            # 왼쪽: 채팅 영역
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 상담 내용",
                    height=550,
                )

                user_input = gr.Textbox(
                    placeholder="질문을 입력하세요... (예: 배송은 얼마나 걸리나요?)",
                    show_label=False,
                    container=False
                )

                with gr.Row():
                    submit_btn = gr.Button("전송", variant="primary", scale=1)
                    clear_btn = gr.Button("대화 초기화", scale=1)

            # 오른쪽: 참고 자료 영역
            with gr.Column(scale=1):
                source_panel = gr.HTML(
                    value="<p style='color: #888; padding: 20px;'>질문을 하시면 관련 참고 자료가 여기에 표시됩니다.</p>",
                    label="📚 참고 자료"
                )

        # 예시 질문
        gr.Examples(
            examples=[
                "SmartWatch Pro X1 가격이 얼마인가요?",
                "환불은 어떻게 신청하나요?",
                "배송은 얼마나 걸리나요?",
                "보증 기간을 연장할 수 있나요?",
                "이어버드 한쪽만 구매 가능한가요?"
            ],
            inputs=user_input,
            label="💡 이런 질문을 해보세요"
        )

        # 이벤트 핸들러 연결
        # Enter 키로 전송
        user_input.submit(
            fn=handle_chat,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, source_panel]
        )

        # 전송 버튼 클릭
        submit_btn.click(
            fn=handle_chat,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, source_panel]
        )

        # 대화 초기화
        clear_btn.click(
            fn=lambda: ([], "<p style='color: #888; padding: 20px;'>질문을 하시면 관련 참고 자료가 여기에 표시됩니다.</p>"),
            outputs=[chatbot, source_panel]
        )

    return ui


def main():
    """메인 실행 함수"""
    theme = gr.themes.Soft(primary_hue="blue")
    ui = create_ui()
    ui.launch(
        inbrowser=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme
    )


if __name__ == "__main__":
    main()
