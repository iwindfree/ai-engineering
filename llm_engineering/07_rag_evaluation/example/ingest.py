from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter



load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"  # 청킹용 LLM 모델

DB_NAME = str(Path(__file__).parent / "vector_db")  # ChromaDB 저장 경로 (LLM 청킹)
DB_NAME_MD = str(Path(__file__).parent / "vector_db_md")  # ChromaDB 저장 경로 (마크다운 청킹)
collection_name = "docs"
md_collection_name = "docs_markdown"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent.parent / "00_test_data" / "knowledge_base"  # 원본 문서 경로
AVERAGE_CHUNK_SIZE = 100  # 청크 수 추정용 평균 글자 수 기준
wait = wait_exponential(multiplier=1, min=10, max=240)  # API 재시도 대기 (지수 백오프)

# 병렬 처리 워커 수 (Rate limit 발생 시 1로 낮출 것)
WORKERS = 3

openai = OpenAI()


class Result(BaseModel):
    """벡터 DB에 저장될 청크 단위 결과물"""
    page_content: str  # headline + summary + original_text 결합 텍스트
    metadata: dict  # 출처(source)와 문서 유형(type) 정보


class Chunk(BaseModel):
    """LLM이 문서를 분할하여 반환하는 개별 청크 (Structured Output)"""
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        """Chunk → Result 변환: headline/summary/original_text를 하나의 텍스트로 결합"""
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )

class Chunks(BaseModel):
    """LLM의 Structured Output 응답 형식 — 청크 리스트를 감싸는 래퍼"""
    chunks: list[Chunk]


def fetch_documents(knowledge_base_path: Path = KNOWLEDGE_BASE_PATH) -> list[dict]:
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in knowledge_base_path.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents




def make_prompt(document, avg_chunk_size=AVERAGE_CHUNK_SIZE):
    """문서를 청킹하기 위한 LLM 프롬프트 생성 (25% 오버랩 포함 지시)"""
    how_many = (len(document["text"]) // avg_chunk_size) + 1  # 문서 길이 기반 권장 청크 수
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a travel company called 하늘여행사 (Sky Travel).
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document, avg_chunk_size=AVERAGE_CHUNK_SIZE):
    """LLM API 호출용 메시지 리스트 생성"""
    return [
        {"role": "user", "content": make_prompt(document, avg_chunk_size)},
    ]


@retry(wait=wait)
def process_document(document, avg_chunk_size=AVERAGE_CHUNK_SIZE):
    """단일 문서를 LLM으로 청킹하여 Result 리스트로 반환 (재시도 포함)"""
    messages = make_messages(document, avg_chunk_size)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents, avg_chunk_size=AVERAGE_CHUNK_SIZE, workers=WORKERS):
    """ThreadPoolExecutor로 병렬 LLM 청킹 (I/O 바운드 → 스레드가 적합)"""
    chunks = []
    total = len(documents)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_document, doc, avg_chunk_size) for doc in documents]
        for future in tqdm(as_completed(futures), total=total, desc="LLM chunking"):
            chunks.extend(future.result())
    return chunks


def run_ingest_stream(
    mode: str = "both",
    avg_chunk_size: int = AVERAGE_CHUNK_SIZE,
    workers: int = WORKERS,
    md_chunk_size: int = 800,
    md_chunk_overlap: int = 100,
    knowledge_base_path: Path = KNOWLEDGE_BASE_PATH,
):
    """Generator: 진행 상황을 실시간으로 yield하는 ingest 파이프라인.

    yields (done, total, stage, extra)
      stage "llm"          : LLM 청킹 진행 중 (done/total 업데이트)
      stage "embedding"    : LLM 임베딩 저장 완료
      stage "md_embedding" : 마크다운 임베딩 저장 완료
      stage "done"         : 전체 완료, extra = {"doc_count", "llm_count", "md_count"}
    """
    documents = fetch_documents(knowledge_base_path)
    total = len(documents)
    llm_count = md_count = None

    if mode in ("llm", "both"):
        llm_chunks = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_document, doc, avg_chunk_size) for doc in documents]
            for i, future in enumerate(as_completed(futures), 1):
                llm_chunks.extend(future.result())
                yield i, total, "llm", None  # 문서 1개 완료마다 caller에서 yield

        create_embeddings(llm_chunks)
        llm_count = len(llm_chunks)
        yield total, total, "embedding", None

    if mode in ("markdown", "both"):
        md_chunks = create_md_chunks(documents, md_chunk_size, md_chunk_overlap)
        create_md_embeddings(md_chunks)
        md_count = len(md_chunks)
        yield total, total, "md_embedding", None

    yield total, total, "done", {"doc_count": total, "llm_count": llm_count, "md_count": md_count}



def chunk_markdown_document(document, md_chunk_size=800, md_chunk_overlap=100):
    """마크다운 헤더 기반 2단계 청킹

    1단계: # ## ### 헤더 기준으로 1차 분할 (의미 단위 보존)
    2단계: md_chunk_size 초과 청크는 RecursiveCharacterTextSplitter로 2차 분할
    각 청크 앞에 헤더 경로 접두사 추가 (검색 품질 향상)
    """
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=md_chunk_size,
        chunk_overlap=md_chunk_overlap,
    )

    md_splits = md_splitter.split_text(document["text"])

    chunks = []
    for split in md_splits:
        header_path = " > ".join(
            split.metadata[k] for k in ["h1", "h2", "h3"] if k in split.metadata
        )

        base_metadata = {
            "source": document["source"],
            "type": document["type"],
            "headers": header_path,
        }

        if len(split.page_content) > md_chunk_size:
            sub_splits = text_splitter.split_text(split.page_content)
            for sub in sub_splits:
                content = f"{header_path}\n\n{sub}" if header_path else sub
                chunks.append(Result(page_content=content, metadata=base_metadata.copy()))
        else:
            content = f"{header_path}\n\n{split.page_content}" if header_path else split.page_content
            chunks.append(Result(page_content=content, metadata=base_metadata.copy()))

    return chunks


def create_md_chunks(documents, md_chunk_size=800, md_chunk_overlap=100):
    """전체 문서를 마크다운 헤더 기반으로 청킹"""
    chunks = []
    for doc in tqdm(documents, desc="Markdown chunking"):
        chunks.extend(chunk_markdown_document(doc, md_chunk_size, md_chunk_overlap))
    print(f"Markdown chunking: {len(chunks)} chunks created")
    return chunks


def create_md_embeddings(chunks):
    """마크다운 청크를 임베딩하여 vector_db_md에 저장"""
    chroma = PersistentClient(path=DB_NAME_MD)
    if md_collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(md_collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(md_collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Markdown vectorstore created with {collection.count()} documents")


def create_embeddings(chunks):
    """청크들을 임베딩하여 ChromaDB에 저장 (기존 컬렉션이 있으면 삭제 후 재생성)"""
    chroma = PersistentClient(path=DB_NAME)
    # 기존 컬렉션 삭제 (clean rebuild)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # OpenAI Embedding API로 벡터 생성
    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    # ChromaDB에 텍스트 + 벡터 + 메타데이터 일괄 저장
    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")



def run_ingest(
    mode: str = "both",
    avg_chunk_size: int = AVERAGE_CHUNK_SIZE,
    workers: int = WORKERS,
    md_chunk_size: int = 800,
    md_chunk_overlap: int = 100,
    knowledge_base_path: Path = KNOWLEDGE_BASE_PATH,
) -> tuple[int, int | None, int | None]:
    """외부에서 호출 가능한 ingest 파이프라인 (CLI용)

    mode: "llm" | "markdown" | "both"
    """
    documents = fetch_documents(knowledge_base_path)
    llm_count, md_count = None, None

    if mode in ("llm", "both"):
        llm_chunks = create_chunks(documents, avg_chunk_size, workers)
        create_embeddings(llm_chunks)
        llm_count = len(llm_chunks)

    if mode in ("markdown", "both"):
        md_chunks = create_md_chunks(documents, md_chunk_size, md_chunk_overlap)
        create_md_embeddings(md_chunks)
        md_count = len(md_chunks)

    return len(documents), llm_count, md_count


if __name__ == "__main__":
    doc_count, llm_count, md_count = run_ingest()
    print(f"Ingestion complete: {doc_count} docs → LLM {llm_count} chunks, MD {md_count} chunks")
