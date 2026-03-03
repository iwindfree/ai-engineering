import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from answer import (
    run_pipeline,
    PipelineConfig,
    STEP_PRESETS,
)
from ingest import KNOWLEDGE_BASE_PATH

load_dotenv(override=True)

STEP_NAMES = list(STEP_PRESETS.keys())  # ["Step 1", ..., "Step 5"]
STEP_CHOICES = [
    ("Step 1 — LLM + Vector", "Step 1"),
    ("Step 2 — LLM + Hybrid", "Step 2"),
    ("Step 3 — MD + Hybrid", "Step 3"),
    ("Step 4 — MD + Hybrid + Rerank", "Step 4"),
    ("Step 5 — MD + Hybrid + Rerank + Expansion", "Step 5"),
]

# ─── Custom CSS ────────────────────────────────────────────────

CUSTOM_CSS = """
/* ══ 전체 배경 ══ */
body, .gradio-container {
    background: #0F1117 !important;
    color: #E2E8F0 !important;
}

/* ══ 헤더 배너 ══ */
#app-header {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #A855F7 100%);
    border-radius: 12px;
    padding: 22px 32px;
    margin-bottom: 20px;
    border: 1px solid rgba(168, 85, 247, 0.4);
    box-shadow: 0 4px 24px rgba(79, 70, 229, 0.35);
}
#app-header h1 {
    color: #FFFFFF !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 4px 0 !important;
}
#app-header p {
    color: rgba(255,255,255,0.75) !important;
    margin: 0 !important;
    font-size: 0.9rem;
}

/* ══ 탭 ══ */
.tab-nav {
    border-bottom: 1px solid #2D3748 !important;
    background: transparent !important;
}
.tab-nav button {
    color: #94A3B8 !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: color 0.15s, background 0.15s;
}
.tab-nav button:hover {
    color: #C4B5FD !important;
    background: rgba(124, 58, 237, 0.08) !important;
}
.tab-nav button.selected {
    border-bottom: 2px solid #A855F7 !important;
    color: #C4B5FD !important;
    font-weight: 700 !important;
    background: rgba(124, 58, 237, 0.12) !important;
}

/* ══ 카드형 설정 그룹 ══ */
.settings-card {
    background: #1A1F2E !important;
    border: 1px solid #2D3748 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.settings-card:hover {
    border-color: #7C3AED !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.2) !important;
}
.settings-card-title {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #A855F7 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 10px !important;
}

/* ══ 입력 컴포넌트 공통 ══ */
input[type=text], textarea, .gr-textbox textarea {
    background: #1A1F2E !important;
    border: 1px solid #2D3748 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
input[type=text]:focus, textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.25) !important;
}

/* ══ 라벨 ══ */
label span, .gr-form label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

/* ══ Radio / Checkbox ══ */
input[type=radio], input[type=checkbox] {
    accent-color: #A855F7;
}

/* ══ Slider ══ */
input[type=range] {
    accent-color: #A855F7;
}

/* ══ Primary 버튼 ══ */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 12px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.15s;
}
button.primary:hover, .gr-button-primary:hover {
    background: linear-gradient(135deg, #4338CA, #6D28D9) !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.5) !important;
    transform: translateY(-1px);
}

/* ══ Secondary 버튼 ══ */
button.secondary {
    background: #1A1F2E !important;
    border: 1px solid #4A5568 !important;
    color: #CBD5E1 !important;
    border-radius: 8px !important;
    transition: all 0.15s;
}
button.secondary:hover {
    border-color: #7C3AED !important;
    color: #C4B5FD !important;
}

/* ══ 상태 박스 ══ */
.ingest-status textarea {
    border-left: 3px solid #A855F7 !important;
    background: #141820 !important;
    color: #A5F3FC !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}

/* ══ 테이블 ══ */
table {
    background: #1A1F2E !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
table thead tr {
    background: rgba(124, 58, 237, 0.25) !important;
}
table thead th {
    color: #C4B5FD !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 10px 12px !important;
}
table tbody tr {
    border-bottom: 1px solid #2D3748 !important;
}
table tbody tr:hover {
    background: rgba(124, 58, 237, 0.08) !important;
}
table tbody td {
    color: #E2E8F0 !important;
    padding: 8px 12px !important;
}

/* ══ Chatbot ══ */
.message.user > div {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: #FFFFFF !important;
    border-radius: 14px 14px 4px 14px !important;
}
.message.bot > div {
    background: #1E2535 !important;
    border: 1px solid #2D3748 !important;
    color: #E2E8F0 !important;
    border-radius: 14px 14px 14px 4px !important;
}

/* ══ 섹션 헤더 ══ */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 12px;
}
.section-header::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 22px;
    background: linear-gradient(180deg, #4F46E5, #A855F7);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ══ Group / Panel 배경 ══ */
.gr-group, .gr-panel, .gr-box {
    background: #1A1F2E !important;
    border: 1px solid #2D3748 !important;
    border-radius: 10px !important;
}

/* ══ 스크롤바 ══ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0F1117; }
::-webkit-scrollbar-thumb { background: #4A5568; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7C3AED; }
"""


def browse_folder():
    """네이티브 OS 폴더 선택 다이얼로그 열기 (macOS: osascript, Windows: PowerShell)"""
    import sys
    import subprocess

    if sys.platform == "darwin":
        result = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "문서 폴더 선택")'],
            capture_output=True,
            text=True,
        )
        folder = result.stdout.strip().rstrip("/")
    elif sys.platform == "win32":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.Description = '문서 폴더 선택';"
            "[void]$d.ShowDialog();"
            "$d.SelectedPath"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
        )
        folder = result.stdout.strip()
    else:
        return gr.update()

    return folder if folder else gr.update()


def _config_from_options(chunking, search, reranking, query_strategy) -> PipelineConfig:
    """개별 옵션으로 PipelineConfig 생성"""
    return PipelineConfig(
        name="Custom",
        chunking=chunking.lower(),
        search=search.lower(),
        reranking=reranking,
        query_strategy=query_strategy.lower(),
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


def chat(history, chunking, search, reranking, query_strategy):
    """Chatbot 이벤트 핸들러: 선택된 설정으로 RAG 답변 생성"""
    config = _config_from_options(chunking, search, reranking, query_strategy)
    last_message = get_text_content(history[-1]["content"])
    prior = [{"role": msg["role"], "content": get_text_content(msg["content"])} for msg in history[:-1]]
    answer, context = run_pipeline(last_message, config, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def do_ingest(knowledge_base_path, mode_label, avg_chunk_size, workers, md_chunk_size, md_chunk_overlap):
    """Ingest 버튼 핸들러 (generator) — run_ingest_stream을 순회해 문서마다 UI 갱신"""
    from pathlib import Path
    from ingest import run_ingest_stream
    from answer import reload_collection, clear_bm25_cache

    kb_path = Path(knowledge_base_path)
    if not kb_path.exists() or not kb_path.is_dir():
        yield f"오류: 경로를 찾을 수 없습니다 — {knowledge_base_path}"
        return

    mode_map = {"LLM 청킹만": "llm", "마크다운 청킹만": "markdown", "둘 다 실행": "both"}
    mode = mode_map[mode_label]

    yield "▶ Ingest 시작 중..."

    result = None
    for done, total, stage, extra in run_ingest_stream(
        mode=mode,
        avg_chunk_size=int(avg_chunk_size),
        workers=int(workers),
        md_chunk_size=int(md_chunk_size),
        md_chunk_overlap=int(md_chunk_overlap),
        knowledge_base_path=kb_path,
    ):
        if stage == "llm":
            bar = "█" * done + "░" * (total - done)
            yield f"LLM 청킹 중... [{bar}] {done}/{total}"
        elif stage == "embedding":
            yield "⚙ LLM 임베딩 저장 중..."
        elif stage == "md_embedding":
            yield "⚙ 마크다운 임베딩 저장 중..."
        elif stage == "done":
            result = extra

    yield "⚙ 컬렉션 재로드 중..."
    reload_collection()
    clear_bm25_cache()

    parts = [f"문서 {result['doc_count']}개"]
    if result["llm_count"] is not None:
        parts.append(f"LLM 청크 {result['llm_count']}개")
    if result["md_count"] is not None:
        parts.append(f"MD 청크 {result['md_count']}개")
    yield "✔ 완료! " + " → ".join(parts)


# ─── 평가 탭 ──────────────────────────────────────────────────

def make_eval_charts(df: pd.DataFrame):
    """평가 결과 DataFrame → (retrieval_fig, answer_fig) plotly 차트"""
    if df is None or df.empty:
        return None, None

    summary = df.groupby("Step")[
        ["Accuracy", "Completeness", "Relevance", "MRR", "nDCG", "P@K", "R@K"]
    ].mean().round(3)
    steps = summary.index.tolist()

    retrieval_metrics = ["MRR", "nDCG", "P@K", "R@K"]
    fig1 = go.Figure()
    for step in steps:
        fig1.add_trace(go.Bar(
            name=step,
            x=retrieval_metrics,
            y=[summary.loc[step, m] for m in retrieval_metrics],
        ))
    fig1.update_layout(
        title="Step별 검색 지표",
        barmode="group",
        yaxis=dict(range=[0, 1.1]),
        height=300,
        margin=dict(t=40, b=30, l=30, r=10),
        legend=dict(orientation="h", y=-0.2),
    )

    answer_metrics = ["Accuracy", "Completeness", "Relevance"]
    fig2 = go.Figure()
    for step in steps:
        fig2.add_trace(go.Bar(
            name=step,
            x=answer_metrics,
            y=[summary.loc[step, m] for m in answer_metrics],
        ))
    fig2.update_layout(
        title="Step별 답변 지표",
        barmode="group",
        yaxis=dict(range=[0, 5.5]),
        height=300,
        margin=dict(t=40, b=30, l=30, r=10),
        legend=dict(orientation="h", y=-0.2),
    )

    return fig1, fig2


def _parse_test_indices(text: str, total: int) -> list[int]:
    """인덱스/범위 텍스트 파싱 — '0,2,5-10' → [0, 2, 5, 6, 7, 8, 9, 10]"""
    if not text or not text.strip():
        return []
    indices = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) == 2 and bounds[0].strip().isdigit() and bounds[1].strip().isdigit():
                start, end = int(bounds[0].strip()), int(bounds[1].strip())
                indices.update(range(start, end + 1))
        elif part.isdigit():
            indices.add(int(part))
    return sorted(i for i in indices if i < total)


def run_evaluation(selected_steps, test_mode, test_count, test_indices_text):
    """평가 실행: 선택된 Step × 테스트로 비교 평가 (generator)"""
    from eval import run_comparison
    from test import load_tests

    if not selected_steps:
        yield (gr.update(), gr.update(), "Step을 1개 이상 선택하세요.", None, None)
        return

    configs = [STEP_PRESETS[s] for s in selected_steps]
    all_tests = load_tests()
    total = len(all_tests)

    if test_mode == "개수 지정":
        n = int(test_count) if test_count and int(test_count) > 0 else total
        tests = all_tests[:min(n, total)]
    elif test_mode == "인덱스/범위 지정":
        indices = _parse_test_indices(test_indices_text, total)
        tests = [all_tests[i] for i in indices]
    else:
        tests = all_tests

    if not tests:
        yield (gr.update(), gr.update(), "테스트 문항이 없습니다.", None, None)
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

        retrieval_chart, answer_chart = make_eval_charts(df)
        progress_text = f"진행 중... {int(progress * 100)}%"
        yield (
            gr.update(value=df),
            gr.update(value=summary),
            progress_text,
            retrieval_chart,
            answer_chart,
        )

    retrieval_chart, answer_chart = make_eval_charts(df)
    yield (
        gr.update(value=df),
        gr.update(value=summary),
        f"완료! {len(configs)}개 Step x {len(tests)}개 질문 = {len(rows)}건 평가",
        retrieval_chart,
        answer_chart,
    )


# ─── Main UI ──────────────────────────────────────────────────

def main():
    def put_message_in_chatbot(message, history):
        """사용자 입력을 챗봇 히스토리에 추가하고 입력창 초기화"""
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0F1117",
        body_background_fill_dark="#0F1117",
        body_text_color="#E2E8F0",
        body_text_color_dark="#E2E8F0",
        block_background_fill="#1A1F2E",
        block_background_fill_dark="#1A1F2E",
        block_border_color="#2D3748",
        block_border_color_dark="#2D3748",
        block_label_text_color="#A855F7",
        block_label_text_color_dark="#A855F7",
        block_title_text_color="#C4B5FD",
        block_title_text_color_dark="#C4B5FD",
        input_background_fill="#141820",
        input_background_fill_dark="#141820",
        input_border_color="#2D3748",
        input_border_color_dark="#2D3748",
        input_placeholder_color="#4A5568",
        input_placeholder_color_dark="#4A5568",
        checkbox_background_color="#1A1F2E",
        checkbox_background_color_dark="#1A1F2E",
        checkbox_border_color="#4A5568",
        checkbox_border_color_dark="#4A5568",
        button_primary_background_fill="linear-gradient(135deg, #4F46E5, #7C3AED)",
        button_primary_background_fill_dark="linear-gradient(135deg, #4F46E5, #7C3AED)",
        button_primary_background_fill_hover="linear-gradient(135deg, #4338CA, #6D28D9)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #4338CA, #6D28D9)",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#1A1F2E",
        button_secondary_background_fill_dark="#1A1F2E",
        button_secondary_border_color="#4A5568",
        button_secondary_border_color_dark="#4A5568",
        button_secondary_text_color="#CBD5E1",
        button_secondary_text_color_dark="#CBD5E1",
        border_color_accent="#7C3AED",
        border_color_accent_dark="#7C3AED",
        color_accent="#A855F7",
        color_accent_soft="rgba(168, 85, 247, 0.15)",
    )

    with gr.Blocks(title="RAG Expert Assistant") as ui:
        gr.Markdown(
            """# 🚀 RAG Expert Assistant
문서 검색 및 RAG 파이프라인 평가 시스템""",
            elem_id="app-header",
        )

        with gr.Tabs():
            # ─── Tab 1: 데이터 준비 ──────────────────────
            with gr.Tab("데이터 준비"):
                gr.Markdown("## 데이터 준비 (벡터 DB 생성)")

                with gr.Row():
                    knowledge_base_path = gr.Textbox(
                        label="문서 폴더 경로",
                        value=str(KNOWLEDGE_BASE_PATH),
                        info="하위 폴더에 .md 파일이 있어야 합니다 (예: path/employees/*.md)",
                        scale=4,
                    )
                    browse_btn = gr.Button("📁 선택", scale=1, min_width=80)

                browse_btn.click(fn=browse_folder, inputs=[], outputs=[knowledge_base_path])

                with gr.Row():
                    with gr.Column(scale=1):
                        ingest_mode = gr.Radio(
                            choices=["LLM 청킹만", "마크다운 청킹만", "둘 다 실행"],
                            value="둘 다 실행",
                            label="청킹 모드",
                        )
                        ingest_btn = gr.Button("Ingest 실행", variant="primary")
                        ingest_status = gr.Textbox(
                            label="Ingest 상태",
                            interactive=False,
                            lines=3,
                            elem_classes="ingest-status",
                        )

                    with gr.Column(scale=2):
                        with gr.Group(visible=True, elem_classes="settings-card") as llm_group:
                            gr.Markdown("**LLM 청킹 설정**", elem_classes="settings-card-title")
                            avg_chunk_size = gr.Slider(
                                minimum=50, maximum=500, value=500, step=50,
                                label="평균 청크 크기 (글자 수)",
                            )
                            workers = gr.Slider(
                                minimum=1, maximum=8, value=3, step=1,
                                label="병렬 워커 수",
                            )

                        with gr.Group(visible=True, elem_classes="settings-card") as md_group:
                            gr.Markdown("**마크다운 청킹 설정**", elem_classes="settings-card-title")
                            md_chunk_size = gr.Slider(
                                minimum=200, maximum=2000, value=500, step=100,
                                label="청크 크기 (글자 수)",
                            )
                            md_chunk_overlap = gr.Slider(
                                minimum=0, maximum=400, value=100, step=50,
                                label="청크 오버랩 (글자 수)",
                            )

                def on_mode_change(mode):
                    show_llm = mode in ("LLM 청킹만", "둘 다 실행")
                    show_md = mode in ("마크다운 청킹만", "둘 다 실행")
                    return gr.update(visible=show_llm), gr.update(visible=show_md)

                ingest_mode.change(
                    fn=on_mode_change,
                    inputs=[ingest_mode],
                    outputs=[llm_group, md_group],
                )
                ingest_btn.click(
                    fn=do_ingest,
                    inputs=[knowledge_base_path, ingest_mode, avg_chunk_size, workers, md_chunk_size, md_chunk_overlap],
                    outputs=[ingest_status],
                )

            # ─── Tab 2: 챗봇 ────────────────────────────
            with gr.Tab("챗봇"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 파이프라인 설정")

                        with gr.Group(elem_classes="settings-card"):
                            gr.Markdown("**검색 옵션**", elem_classes="settings-card-title")
                            chunking = gr.Radio(
                                choices=["LLM", "Markdown"],
                                value="LLM",
                                label="컬렉션 선택 (사용할 DB)",
                            )
                            search = gr.Radio(
                                choices=["Vector", "Hybrid"],
                                value="Vector",
                                label="검색 방식",
                            )
                            reranking = gr.Checkbox(
                                value=False,
                                label="Reranking",
                            )
                            query_strategy = gr.Radio(
                                choices=["Basic", "Rewrite", "Expansion"],
                                value="Basic",
                                label="쿼리 전략",
                            )

                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Conversation", height=500)
                        message = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything...",
                            show_label=False,
                        )

                    with gr.Column(scale=2):
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
                    inputs=[chatbot, chunking, search, reranking, query_strategy],
                    outputs=[chatbot, context_markdown],
                )

            # ─── Tab 3: 평가 대시보드 ────────────────────
            with gr.Tab("평가 대시보드"):
                gr.Markdown("### RAG 파이프라인 비교 평가")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 평가 파이프라인 선택")
                        eval_steps = gr.CheckboxGroup(
                            choices=STEP_CHOICES,
                            value=STEP_NAMES,
                            label="비교할 Step",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 테스트 문항 선택")
                        test_mode = gr.Radio(
                            choices=["전체", "개수 지정", "인덱스/범위 지정"],
                            value="전체",
                            label="선택 방식",
                        )
                        test_count = gr.Number(
                            value=10,
                            label="테스트 개수 (앞에서부터)",
                            precision=0,
                            minimum=1,
                            visible=False,
                        )
                        test_indices_text = gr.Textbox(
                            label="인덱스/범위 (예: 0,2,5-10)",
                            placeholder="0,1,3 또는 0-9 또는 0,2,5-10",
                            visible=False,
                        )

                eval_btn = gr.Button("평가 실행", variant="primary")
                eval_status = gr.Textbox(label="평가 상태", interactive=False)

                gr.Markdown("#### 상세 결과")
                eval_detail_df = gr.Dataframe(
                    label="비교 결과 테이블",
                    interactive=False,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Step별 평균 요약")
                        eval_summary_df = gr.Dataframe(
                            label="Step별 평균 점수",
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        retrieval_chart = gr.Plot(label="검색 지표")

                    with gr.Column(scale=1):
                        answer_chart = gr.Plot(label="답변 지표")

                def on_test_mode_change(mode):
                    return (
                        gr.update(visible=mode == "개수 지정"),
                        gr.update(visible=mode == "인덱스/범위 지정"),
                    )

                test_mode.change(
                    fn=on_test_mode_change,
                    inputs=[test_mode],
                    outputs=[test_count, test_indices_text],
                )

                def _run_eval(steps, mode, count, indices_text):
                    yield from run_evaluation(steps, mode, count, indices_text)

                eval_btn.click(
                    fn=_run_eval,
                    inputs=[eval_steps, test_mode, test_count, test_indices_text],
                    outputs=[eval_detail_df, eval_summary_df, eval_status, retrieval_chart, answer_chart],
                )

    ui.launch(inbrowser=True, theme=theme, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
