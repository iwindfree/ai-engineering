import math
from collections import Counter
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
from kiwipiepy import Kiwi

load_dotenv(override=True)
MODEL = "openai/gpt-4.1-nano"  # 답변 생성·쿼리 재작성·리랭킹에 사용하는 LLM
DB_NAME = str(Path(__file__).parent / "vector_db")  # ChromaDB 저장 경로 (LLM 청킹)
DB_NAME_MD = str(Path(__file__).parent / "vector_db_md")  # ChromaDB 저장 경로 (마크다운 청킹)

collection_name = "docs"
md_collection_name = "docs_markdown"
embedding_model = "text-embedding-3-large"  # 쿼리 임베딩 모델 (ingest.py와 동일해야 함)
wait = wait_exponential(multiplier=1, min=10, max=240)  # API 재시도 대기 (지수 백오프)

openai = OpenAI()

# 모듈 로드 시점에 벡터 DB 연결 (ingest 후에는 reload_collection()으로 재로드)
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

chroma_md = PersistentClient(path=DB_NAME_MD)
md_collection = chroma_md.get_or_create_collection(md_collection_name)


def reload_collection():
    """벡터 DB가 새로 생성된 후 두 컬렉션 모두 재로드"""
    global chroma, collection, chroma_md, md_collection
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(collection_name)
    chroma_md = PersistentClient(path=DB_NAME_MD)
    md_collection = chroma_md.get_or_create_collection(md_collection_name)


RETRIEVAL_K = 10  # 벡터 검색 시 가져올 최대 청크 수
K_RRF = 60  # RRF 상수 (표준값)

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


# ─── Pydantic Models ───────────────────────────────────────────

class Result(BaseModel):
    page_content: str  # 청크의 텍스트 내용
    metadata: dict  # 메타데이터 (출처, 페이지 번호 등)

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

class ExpandedQueries(BaseModel):
    """확장된 검색 쿼리 목록"""
    queries: list[str] = Field(
        description="원본 질문과 같은 정보를 찾을 수 있는 다양한 검색 쿼리 2개"
    )


# ─── PipelineConfig & Step Presets ─────────────────────────────

@dataclass
class PipelineConfig:
    name: str
    chunking: str       # "llm" | "markdown"
    search: str         # "vector" | "hybrid"
    reranking: bool     # True | False
    query_strategy: str  # "basic" | "rewrite" | "expansion"


STEP_PRESETS = {
    "Step 1": PipelineConfig(name="Step 1", chunking="llm", search="vector", reranking=False, query_strategy="basic"),
    "Step 2": PipelineConfig(name="Step 2", chunking="llm", search="hybrid", reranking=False, query_strategy="basic"),
    "Step 3": PipelineConfig(name="Step 3", chunking="markdown", search="hybrid", reranking=False, query_strategy="basic"),
    "Step 4": PipelineConfig(name="Step 4", chunking="markdown", search="hybrid", reranking=True, query_strategy="basic"),
    "Step 5": PipelineConfig(name="Step 5", chunking="markdown", search="hybrid", reranking=True, query_strategy="expansion"),
}


# ─── BM25 Keyword Search ──────────────────────────────────────

_kiwi = Kiwi()
_bm25_stats_cache = {}
_all_docs_cache = {}


def _tokenize_korean(text):
    """kiwipiepy 형태소 분석 기반 토큰화"""
    tokens = []
    for token in _kiwi.tokenize(text):
        form = token.form.lower()
        if len(form) >= 1:
            tokens.append(form)
    return tokens


def _get_all_docs(coll):
    """컬렉션의 전체 문서를 캐시하여 반환"""
    key = coll.name
    if key in _all_docs_cache:
        return _all_docs_cache[key]

    results = coll.get(include=["documents", "metadatas"])
    docs = list(zip(results["documents"], results["metadatas"]))
    _all_docs_cache[key] = docs
    print(f"전체 문서 로드: {key} → {len(docs)}개")
    return docs


def _get_bm25_stats(coll):
    """전체 문서에서 BM25에 필요한 통계를 사전 계산 (캐시)"""
    key = coll.name
    if key in _bm25_stats_cache:
        return _bm25_stats_cache[key]

    all_docs = _get_all_docs(coll)
    N = len(all_docs)

    df_word = {}
    total_tokens = 0

    for doc_text, _ in all_docs:
        tokens = _tokenize_korean(doc_text)
        total_tokens += len(tokens)
        for t in set(tokens):
            df_word[t] = df_word.get(t, 0) + 1

    avg_dl = total_tokens / N if N > 0 else 1

    _bm25_stats_cache[key] = {
        'N': N,
        'avg_dl': avg_dl,
        'df_word': df_word,
    }
    print(f"BM25 통계 계산 완료: 문서 {N}개, 평균 토큰 수 {avg_dl:.1f}, 고유 단어 {len(df_word)}개")
    return _bm25_stats_cache[key]


def _idf(term, df_dict, N):
    """BM25 IDF 계산"""
    df = df_dict.get(term, 0)
    return math.log((N - df + 0.5) / (df + 0.5) + 1)


def keyword_search(question, coll, top_k=RETRIEVAL_K):
    """BM25 + 형태소 분석 키워드 검색"""
    k1 = 1.2
    b = 0.75

    stats = _get_bm25_stats(coll)
    N, avg_dl = stats['N'], stats['avg_dl']
    df_word = stats['df_word']

    all_docs = _get_all_docs(coll)

    query_tokens = set(_tokenize_korean(question))

    scored = []
    for doc_text, metadata in all_docs:
        doc_tokens = _tokenize_korean(doc_text)
        dl = len(doc_tokens)

        word_score = 0.0
        tf_counter = Counter(doc_tokens)

        for t in query_tokens:
            if t not in tf_counter:
                continue
            tf = tf_counter[t]
            idf = _idf(t, df_word, N)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            word_score += idf * tf_norm

        if word_score > 0:
            scored.append((word_score, doc_text, metadata))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [Result(page_content=text, metadata=meta) for _, text, meta in scored[:top_k]]


def clear_bm25_cache():
    """BM25 캐시 초기화 (ingest 후 호출)"""
    global _bm25_stats_cache, _all_docs_cache
    _bm25_stats_cache = {}
    _all_docs_cache = {}


# ─── Reranking ─────────────────────────────────────────────────

@retry(wait=wait)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """LLM을 사용하여 검색된 청크들을 질문과의 관련성 순으로 재정렬"""
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
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


# ─── RAG Messages ──────────────────────────────────────────────

def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    """RAG 응답 생성을 위한 메시지 리스트를 구성"""
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


# ─── Query Strategies ─────────────────────────────────────────

@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] = []) -> str:
    """사용자의 질문을 Knowledge Base 검색에 최적화된 형태로 재작성"""
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
    """원본 질문에서 2개의 확장 검색 쿼리를 생성"""
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


# ─── Search Functions ──────────────────────────────────────────

def fetch_context_vector_only(question: str, coll, top_k=RETRIEVAL_K) -> list[Result]:
    """순수 벡터 검색 (BM25 없음)"""
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = coll.query(query_embeddings=[query], n_results=top_k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks


def fetch_context_hybrid(question: str, coll, top_k=RETRIEVAL_K) -> list[Result]:
    """Hybrid Search: 벡터 검색 + BM25 키워드 검색을 RRF로 병합"""
    # 벡터 검색
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = coll.query(query_embeddings=[query], n_results=top_k)
    vector_chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        vector_chunks.append(Result(page_content=doc, metadata=meta))

    # BM25 키워드 검색
    keyword_chunks = keyword_search(question, coll, top_k=top_k)

    # RRF 병합
    seen = set()
    merged = []
    rrf_scores = {}

    for rank, chunk in enumerate(vector_chunks):
        key = chunk.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (K_RRF + rank + 1)
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    for rank, chunk in enumerate(keyword_chunks):
        key = chunk.page_content
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (K_RRF + rank + 1)
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    merged.sort(key=lambda c: rrf_scores.get(c.page_content, 0), reverse=True)
    return merged[:top_k]


# ─── Legacy Functions (backward compatibility) ─────────────────

def fetch_context_unranked(question: str) -> list[Result]:
    """기존 호환용: LLM 컬렉션에서 벡터 검색"""
    return fetch_context_vector_only(question, collection)


def fetch_context_multi(question: str, queries: list[str]) -> list[Result]:
    """기존 호환용: 여러 쿼리로 검색 → 중복 제거 → Reranking"""
    print(f"검색 쿼리 ({len(queries)}개):")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")

    all_chunks = []
    for q in queries:
        all_chunks.extend(fetch_context_unranked(q))

    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)

    print(f"전체 수집: {len(all_chunks)}개 → 중복 제거 후: {len(unique_chunks)}개")

    reranked = rerank(question, unique_chunks)
    return reranked[:RETRIEVAL_K]


# ─── Unified Pipeline ─────────────────────────────────────────

@retry(wait=wait)
def run_pipeline(question: str, config: PipelineConfig, history: list[dict] = []) -> tuple[str, list[Result]]:
    """설정 기반 통합 RAG 파이프라인

    1. config.query_strategy에 따라 쿼리 생성
    2. config.chunking에 따라 collection 선택
    3. 각 쿼리에 대해 config.search에 따라 검색
    4. 중복 제거
    5. config.reranking이면 rerank 적용
    6. 답변 생성
    """
    # 1. 쿼리 생성
    if config.query_strategy == "expansion":
        rewritten = rewrite_query(question, history)
        expanded = expand_queries(question, history)
        queries = [question, rewritten] + expanded
    elif config.query_strategy == "rewrite":
        rewritten = rewrite_query(question, history)
        queries = [question, rewritten]
    else:  # basic
        queries = [question]

    print(f"[{config.name}] 검색 쿼리 ({len(queries)}개):")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")

    # 2. 컬렉션 선택
    coll = md_collection if config.chunking == "markdown" else collection

    # 3. 검색
    search_fn = fetch_context_hybrid if config.search == "hybrid" else fetch_context_vector_only
    all_chunks = []
    for q in queries:
        all_chunks.extend(search_fn(q, coll))

    # 4. 중복 제거
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)

    print(f"[{config.name}] 수집: {len(all_chunks)}개 → 중복 제거: {len(unique_chunks)}개")

    # 5. Reranking
    if config.reranking:
        chunks = rerank(question, unique_chunks)[:RETRIEVAL_K]
    else:
        chunks = unique_chunks[:RETRIEVAL_K]

    # 6. 답변 생성
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks


# ─── Legacy Answer Functions (backward compatibility) ──────────

@retry(wait=wait)
def answer_question_basic(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """원본 쿼리만으로 검색하여 답변 (기존 호환용)"""
    queries = [question]
    chunks = fetch_context_multi(question, queries)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks


@retry(wait=wait)
def answer_question_with_rewrite(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """원본 + Rewrite 쿼리로 검색하여 답변 (기존 호환용)"""
    rewritten = rewrite_query(question, history)
    queries = [question, rewritten]
    chunks = fetch_context_multi(question, queries)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks


@retry(wait=wait)
def answer_question_with_expansion(question: str, history: list[dict] = []) -> tuple[str, list[Result]]:
    """원본 + Rewrite + Expand 쿼리로 검색하여 답변 (기존 호환용)"""
    rewritten = rewrite_query(question, history)
    expanded = expand_queries(question, history)
    queries = [question, rewritten] + expanded
    chunks = fetch_context_multi(question, queries)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
