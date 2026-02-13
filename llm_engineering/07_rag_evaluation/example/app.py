import gradio as gr
from dotenv import load_dotenv

from answer import (
    answer_question_basic,
    answer_question_with_rewrite,
    answer_question_with_expansion,
)

load_dotenv(override=True)

# UI Radio 선택지 → 답변 함수 매핑
STRATEGY_MAP = {
    "Basic (원본 쿼리만)": answer_question_basic,           # 전략 1: 원본 쿼리만 사용
    "Rewrite (원본 + 재작성)": answer_question_with_rewrite,  # 전략 2: 원본 + 재작성 쿼리
    "Expansion (원본 + 재작성 + 확장)": answer_question_with_expansion,  # 전략 3: 원본 + 재작성 + 확장 쿼리 2개
}


def format_context(context):
    """검색된 청크들을 HTML 형식으로 포맷하여 우측 패널에 표시"""
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

def chat(history, strategy):
    """Chatbot 이벤트 핸들러: 선택된 전략으로 RAG 답변 생성 후 히스토리·컨텍스트 업데이트"""
    last_message = get_text_content(history[-1]["content"])
    prior = [{"role": msg["role"], "content": get_text_content(msg["content"])} for msg in history[:-1]]
    answer_fn = STRATEGY_MAP[strategy]
    answer, context = answer_fn(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)

def do_ingest():
    """Ingest 버튼 핸들러 (generator): 문서 로드 → 청킹 → 임베딩 → DB 재로드

    lazy import로 모듈 로드 시점의 불필요한 의존성 방지.
    Gradio generator 패턴으로 '진행 중' → '완료' 상태 전환 표시.
    """
    from ingest import run_ingest
    from answer import reload_collection
    yield "Ingest 진행 중... (문서 로드 → 청킹 → 임베딩)"
    docs, chunks = run_ingest()
    reload_collection()  # 새로 생성된 벡터 DB를 answer 모듈에 반영
    yield f"완료! 문서 {docs}개 → 청크 {chunks}개 → 벡터 DB 생성"


def main():
    def put_message_in_chatbot(message, history):
        """사용자 입력을 챗봇 히스토리에 추가하고 입력창 초기화"""
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Expert Assistant", theme=theme) as ui:
        gr.Markdown("# RAG Expert Assistant\nAsk me anything!")

        with gr.Row():
            ingest_btn = gr.Button("Ingest (벡터 DB 생성)", variant="secondary")
            ingest_status = gr.Textbox(
                label="Ingest 상태",
                interactive=False,
                scale=3,
            )

        ingest_btn.click(fn=do_ingest, outputs=[ingest_status])

        with gr.Row():
            with gr.Column(scale=1):
                strategy = gr.Radio(
                    choices=list(STRATEGY_MAP.keys()),
                    value="Basic (원본 쿼리만)",
                    label="Query Strategy",
                )
                chatbot = gr.Chatbot(
                    label="Conversation", height=550
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # 메시지 제출 → 히스토리에 추가 → RAG 답변 생성 (체이닝)
        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=[chatbot, strategy], outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
