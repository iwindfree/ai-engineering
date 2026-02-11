"""
RAG 체인 모듈 (Gradio 6.x 전용)
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

# 설정
BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """당신은 하늘여행사의 친절한 고객 상담원입니다.

## 답변 지침
- 아래 참고 자료를 기반으로 답변하세요.
- 참고 자료에 없는 내용은 "확인 후 안내드리겠습니다"라고 말씀해주세요.

## 참고 자료
{context}
"""


def generate_answer(query: str, history: list = None):
    """RAG 파이프라인 실행"""
    if history is None:
        history = []

    # 1. 벡터 검색
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embedding)
    docs = vectorstore.similarity_search(query, k=5)

    # 2. 컨텍스트 구성
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 3. 메시지 구성
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    # history는 Gradio 6.x messages 형식: [{"role": "user", "content": "..."}, ...]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    # 4. LLM 호출
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    response = llm.invoke(messages)

    return response.content, docs


if __name__ == "__main__":
    answer, docs = generate_answer("비자대행 서비스도 가능한가요?")
    print(f"답변: {answer}")
    print(f"참조 문서: {len(docs)}개")
