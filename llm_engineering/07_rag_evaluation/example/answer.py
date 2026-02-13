from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential

load_dotenv(override=True)
MODEL = "openai/gpt-4.1-nano"  # 답변 생성·쿼리 재작성·리랭킹에 사용하는 LLM
DB_NAME = str(Path(__file__).parent / "vector_db")  # ChromaDB 저장 경로

collection_name = "docs"
embedding_model = "text-embedding-3-large"  # 쿼리 임베딩 모델 (ingest.py와 동일해야 함)
wait = wait_exponential(multiplier=1, min=10, max=240)  # API 재시도 대기 (지수 백오프)

openai = OpenAI()

# 모듈 로드 시점에 벡터 DB 연결 (ingest 후에는 reload_collection()으로 재로드)
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)


def reload_collection():
    """벡터 DB가 새로 생성된 후 collection을 재로드"""
    global chroma, collection
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(collection_name)


RETRIEVAL_K = 10  # 벡터 검색 시 가져올 최대 청크 수

# RAG 시스템 프롬프트 — {context}에 검색된 청크들이 삽입됨
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the travel company 하늘여행사 (Sky Travel).
You are chatting with a user about 하늘여행사.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

class Result(BaseModel):
    page_content: str # 청크의 텍스트 내용
    metadata: dict # 메타데이터 (출처, 페이지 번호 등)

class RankOrder(BaseModel):
      # 관련성 순으로 정렬된 청크 ID 리스트
      order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

class ExpandedQueries(BaseModel):
    """확장된 검색 쿼리 목록"""
    queries: list[str] = Field(
        description="원본 질문과 같은 정보를 찾을 수 있는 다양한 검색 쿼리 2개"
    )


@retry(wait=wait)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """LLM을 사용하여 검색된 청크들을 질문과의 관련성 순으로 재정렬합니다.

    벡터 검색은 의미적 유사성 기반이라 정확한 관련성 순서가 아닐 수 있습니다.
    LLM이 질문과 각 청크를 함께 보고 관련성을 판단하여 (Cross-encoder 효과)
    가장 관련성 높은 청크를 먼저 배치합니다.

    Args:
        question: 사용자가 입력한 질문 문자열
        chunks: 벡터 검색으로 가져온 Result 객체들의 리스트

    Returns:
        관련성 순으로 재정렬된 Result 객체들의 리스트
    """
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    # model_validate_json : Pydantic v2의 메서드로, JSON 문자열을 직접 파싱하여 모델 인스턴스로 변환합니다.
    order = RankOrder.model_validate_json(reply).order #RankOrder 객체에서 order 리스트 추출
    return [chunks[i - 1] for i in order] # LLM이 반환한 순서대로 청크 재배열 (1-based → 0-based 인덱스 변환)


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    """RAG 응답 생성을 위한 메시지 리스트를 구성합니다.

    검색된 청크들을 컨텍스트로 포함한 시스템 프롬프트와
    대화 히스토리, 현재 질문을 결합하여 LLM에 전달할 메시지를 생성합니다.

    Args:
        question: 사용자가 입력한 현재 질문
        history: 이전 대화 히스토리 (role, content 키를 가진 딕셔너리 리스트)
        chunks: 검색된 Result 객체들의 리스트

    Returns:
        LLM에 전달할 메시지 딕셔너리들의 리스트
    """
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] = []) -> str:
    """사용자의 질문을 Knowledge Base 검색에 최적화된 형태로 재작성합니다.

    벡터 검색은 짧고 명확한 쿼리에서 더 좋은 결과를 냅니다.
    대화 맥락을 고려하여 모호한 질문을 구체적인 검색 쿼리로 변환합니다.

    Args:
        question: 사용자가 입력한 원본 질문
        history: 이전 대화 히스토리 (role, content 키를 가진 딕셔너리 리스트)

    Returns:
        Knowledge Base 검색에 최적화된 재작성된 질문 문자열
    """
    message = f"""
You are in a conversation with a user, answering questions about the travel company 하늘여행사 (Sky Travel).
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


@retry(wait=wait)
def expand_queries(question: str, history: list[dict] = []) -> list[str]:
    """원본 질문에서 2개의 확장 검색 쿼리를 생성합니다.

    Args:
        question: 사용자가 입력한 원본 질문
        history: 이전 대화 히스토리

    Returns:
        2개의 확장 검색 쿼리 리스트
    """
    message = f"""
You are helping search a Knowledge Base about the travel company 하늘여행사 (Sky Travel).

This is the conversation history:
{history}

The user's question is:
{question}

Generate exactly 2 different search queries that could find information relevant to this question.
Each query should approach the question from a different angle or use different keywords.
Keep each query short and specific.
IMPORTANT: Write queries in the same language as the user's question.
"""
    response = completion(
        model=MODEL,
        messages=[{"role": "system", "content": message}],
        response_format=ExpandedQueries,
    )
    return ExpandedQueries.model_validate_json(response.choices[0].message.content).queries


def fetch_context_unranked(question: str) -> list[Result]:
    """주어진 질문에 대해 벡터 검색으로 관련 청크를 가져옵니다.

    질문을 임베딩으로 변환한 후 ChromaDB에서 유사한 문서 청크를 검색합니다.
    이 단계에서는 재정렬(reranking)을 수행하지 않습니다.

    Args:
        question: 검색할 질문 문자열

    Returns:
        검색된 Result 객체들의 리스트 (최대 RETRIEVAL_K개)
    """
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context_multi(question: str, queries: list[str]) -> list[Result]:
    """여러 쿼리로 검색 → 중복 제거 → Reranking 파이프라인

    Args:
        question: 원본 질문 (reranking 기준)
        queries: 검색에 사용할 쿼리 리스트

    Returns:
        중복 제거 + reranking된 Result 리스트 (최대 RETRIEVAL_K개)
    """
    print(f"검색 쿼리 ({len(queries)}개):")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")

    # 각 쿼리로 검색 → 청크 수집
    all_chunks = []
    for q in queries:
        all_chunks.extend(fetch_context_unranked(q))

    # page_content 기준 중복 제거
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)

    print(f"전체 수집: {len(all_chunks)}개 → 중복 제거 후: {len(unique_chunks)}개")

    # Reranking → 상위 RETRIEVAL_K개
    reranked = rerank(question, unique_chunks)
    return reranked[:RETRIEVAL_K]


@retry(wait=wait)
def answer_question_basic(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """원본 쿼리만으로 검색하여 답변합니다.

    전략 1: 쿼리 변환 없이 원본 질문만으로 검색 후 답변 생성

    Args:
        question: 사용자가 입력한 현재 질문
        history: 이전 대화 히스토리

    Returns:
        tuple: (LLM이 생성한 답변 문자열, 검색된 Result 객체들의 리스트)
    """
    queries = [question]
    chunks = fetch_context_multi(question, queries)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks


@retry(wait=wait)
def answer_question_with_rewrite(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """원본 + Rewrite 쿼리로 검색하여 답변합니다.

    전략 2: 원본 질문 + 재작성된 질문으로 검색 후 답변 생성

    Args:
        question: 사용자가 입력한 현재 질문
        history: 이전 대화 히스토리

    Returns:
        tuple: (LLM이 생성한 답변 문자열, 검색된 Result 객체들의 리스트)
    """
    rewritten = rewrite_query(question, history)
    queries = [question, rewritten]
    chunks = fetch_context_multi(question, queries)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks


@retry(wait=wait)
def answer_question_with_expansion(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """원본 + Rewrite + Expand(2개) 쿼리로 검색하여 답변합니다.

    전략 3: 원본 질문 + 재작성된 질문 + 확장 쿼리 2개로 검색 후 답변 생성

    Args:
        question: 사용자가 입력한 현재 질문
        history: 이전 대화 히스토리

    Returns:
        tuple: (LLM이 생성한 답변 문자열, 검색된 Result 객체들의 리스트)
    """
    rewritten = rewrite_query(question, history)
    expanded = expand_queries(question, history)  # 2개
    queries = [question, rewritten] + expanded
    chunks = fetch_context_multi(question, queries)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
