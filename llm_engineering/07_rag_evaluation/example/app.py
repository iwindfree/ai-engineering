import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from answer import (
    run_pipeline,
    PipelineConfig,
    STEP_PRESETS,
)

load_dotenv(override=True)

STEP_NAMES = list(STEP_PRESETS.keys())  # ["Step 1", ..., "Step 5"]
PRESET_CHOICES = STEP_NAMES + ["Custom"]


def _config_from_preset(step_name: str) -> PipelineConfig:
    """Step 프리셋 이름으로 PipelineConfig 반환"""
    return STEP_PRESETS[step_name]


def _config_from_options(chunking, search, reranking, query_strategy) -> PipelineConfig:
    """개별 옵션으로 PipelineConfig 생성"""
    return PipelineConfig(
        name="Custom",
        chunking=chunking.lower(),
        search=search.lower(),
        reranking=reranking,
        query_strategy=query_strategy.lower(),
    )


def _get_current_config(preset, chunking, search, reranking, query_strategy) -> PipelineConfig:
    """현재 UI 상태에서 PipelineConfig 추출"""
    if preset != "Custom":
        return _config_from_preset(preset)
    return _config_from_options(chunking, search, reranking, query_strategy)


# ─── Step 프리셋 ↔ 개별 옵션 연동 ──────────────────────────────

_PRESET_TO_OPTIONS = {
    "Step 1": ("LLM", "Vector", False, "Basic"),
    "Step 2": ("LLM", "Hybrid", False, "Basic"),
    "Step 3": ("Markdown", "Hybrid", False, "Basic"),
    "Step 4": ("Markdown", "Hybrid", True, "Basic"),
    "Step 5": ("Markdown", "Hybrid", True, "Expansion"),
}


def on_preset_change(preset):
    """Step 프리셋 변경 시 개별 옵션 자동 세팅"""
    if preset == "Custom":
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    chunking, search, reranking, query_strategy = _PRESET_TO_OPTIONS[preset]
    return (
        gr.update(value=chunking, interactive=False),
        gr.update(value=search, interactive=False),
        gr.update(value=reranking, interactive=False),
        gr.update(value=query_strategy, interactive=False),
    )


# ─── 챗봇 탭 ──────────────────────────────────────────────────

def format_context(context):
    """검색된 청크들을 HTML 형식으로 포맷"""
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


def chat(history, preset, chunking, search, reranking, query_strategy):
    """Chatbot 이벤트 핸들러: 선택된 설정으로 RAG 답변 생성"""
    config = _get_current_config(preset, chunking, search, reranking, query_strategy)
    last_message = get_text_content(history[-1]["content"])
    prior = [{"role": msg["role"], "content": get_text_content(msg["content"])} for msg in history[:-1]]
    answer, context = run_pipeline(last_message, config, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def do_ingest():
    """Ingest 버튼 핸들러 (generator)"""
    from ingest import run_ingest
    from answer import reload_collection, clear_bm25_cache
    yield "Ingest 진행 중... (문서 로드 → LLM 청킹 + 마크다운 청킹 → 임베딩)"
    docs, llm_chunks, md_chunks = run_ingest()
    reload_collection()
    clear_bm25_cache()
    yield f"완료! 문서 {docs}개 → LLM 청크 {llm_chunks}개, MD 청크 {md_chunks}개"


# ─── 평가 탭 ──────────────────────────────────────────────────

def run_evaluation(selected_steps, use_all_tests, selected_test_indices):
    """평가 실행: 선택된 Step × 테스트로 비교 평가 (generator)"""
    from eval import run_comparison
    from test import load_tests

    if not selected_steps:
        yield (
            gr.update(),
            gr.update(),
            "Step을 1개 이상 선택하세요.",
        )
        return

    configs = [STEP_PRESETS[s] for s in selected_steps]
    tests = load_tests()

    if not use_all_tests and selected_test_indices:
        indices = [int(i) for i in selected_test_indices]
        tests = [tests[i] for i in indices if i < len(tests)]

    if not tests:
        yield (
            gr.update(),
            gr.update(),
            "테스트 질문이 없습니다.",
        )
        return

    rows = []
    for config_name, test_idx, r_eval, a_eval, answer_text, progress in run_comparison(configs, tests):
        rows.append({
            "Step": config_name,
            "질문": tests[test_idx].question[:50] + "...",
            "Accuracy": a_eval.accuracy,
            "Completeness": a_eval.completeness,
            "Relevance": a_eval.relevance,
            "MRR": round(r_eval.mrr, 3),
            "nDCG": round(r_eval.ndcg, 3),
            "P@K": round(r_eval.precision_at_k, 3),
            "R@K": round(r_eval.recall_at_k, 3),
        })

        df = pd.DataFrame(rows)

        summary = df.groupby("Step")[
            ["Accuracy", "Completeness", "Relevance", "MRR", "nDCG", "P@K", "R@K"]
        ].mean().round(3).reset_index()

        progress_text = f"진행 중... {int(progress * 100)}%"
        yield (
            gr.update(value=df),
            gr.update(value=summary),
            progress_text,
        )

    yield (
        gr.update(value=df),
        gr.update(value=summary),
        f"완료! {len(configs)}개 Step x {len(tests)}개 질문 = {len(rows)}건 평가",
    )


# ─── Main UI ──────────────────────────────────────────────────

def main():
    def put_message_in_chatbot(message, history):
        """사용자 입력을 챗봇 히스토리에 추가하고 입력창 초기화"""
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Expert Assistant") as ui:
        gr.Markdown("# RAG Expert Assistant\nAsk me anything!")

        with gr.Tabs():
            # ─── Tab 1: 챗봇 ────────────────────────────
            with gr.Tab("챗봇"):
                with gr.Row():
                    ingest_btn = gr.Button("Ingest (벡터 DB 생성)", variant="secondary")
                    ingest_status = gr.Textbox(label="Ingest 상태", interactive=False, scale=3)

                ingest_btn.click(fn=do_ingest, outputs=[ingest_status])

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 파이프라인 설정")

                        preset = gr.Radio(
                            choices=PRESET_CHOICES,
                            value="Step 1",
                            label="Step 프리셋",
                        )

                        with gr.Group():
                            gr.Markdown("**개별 옵션** (Custom 선택 시 활성화)")
                            chunking = gr.Radio(
                                choices=["LLM", "Markdown"],
                                value="LLM",
                                label="청킹 방식",
                                interactive=False,
                            )
                            search = gr.Radio(
                                choices=["Vector", "Hybrid"],
                                value="Vector",
                                label="검색 방식",
                                interactive=False,
                            )
                            reranking = gr.Checkbox(
                                value=False,
                                label="Reranking",
                                interactive=False,
                            )
                            query_strategy = gr.Radio(
                                choices=["Basic", "Rewrite", "Expansion"],
                                value="Basic",
                                label="쿼리 전략",
                                interactive=False,
                            )

                        preset.change(
                            fn=on_preset_change,
                            inputs=[preset],
                            outputs=[chunking, search, reranking, query_strategy],
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(label="Conversation", height=550)
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

                message.submit(
                    put_message_in_chatbot,
                    inputs=[message, chatbot],
                    outputs=[message, chatbot],
                ).then(
                    chat,
                    inputs=[chatbot, preset, chunking, search, reranking, query_strategy],
                    outputs=[chatbot, context_markdown],
                )

            # ─── Tab 2: 평가 대시보드 ────────────────────
            with gr.Tab("평가 대시보드"):
                gr.Markdown("### Step별 RAG 파이프라인 비교 평가")

                with gr.Row():
                    with gr.Column(scale=1):
                        eval_steps = gr.CheckboxGroup(
                            choices=STEP_NAMES,
                            value=["Step 1", "Step 2"],
                            label="비교할 Step 선택",
                        )

                    with gr.Column(scale=1):
                        use_all_tests = gr.Checkbox(value=True, label="전체 테스트 사용")
                        test_indices = gr.Textbox(
                            label="테스트 인덱스 (쉼표 구분, 예: 0,1,3)",
                            placeholder="0,1,2",
                        )

                eval_btn = gr.Button("평가 실행", variant="primary")
                eval_status = gr.Textbox(label="평가 상태", interactive=False)

                gr.Markdown("#### 상세 결과")
                eval_detail_df = gr.Dataframe(
                    label="비교 결과 테이블",
                    interactive=False,
                )

                gr.Markdown("#### Step별 평균 요약")
                eval_summary_df = gr.Dataframe(
                    label="Step별 평균 점수",
                    interactive=False,
                )

                def parse_test_indices(indices_text):
                    """쉼표 구분 텍스트를 인덱스 리스트로 파싱"""
                    if not indices_text or not indices_text.strip():
                        return []
                    return [s.strip() for s in indices_text.split(",") if s.strip().isdigit()]

                eval_btn.click(
                    fn=lambda steps, use_all, indices_text: run_evaluation(
                        steps, use_all, parse_test_indices(indices_text)
                    ),
                    inputs=[eval_steps, use_all_tests, test_indices],
                    outputs=[eval_detail_df, eval_summary_df, eval_status],
                )

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
