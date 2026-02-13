from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential



load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"  # 청킹용 LLM 모델

DB_NAME = str(Path(__file__).parent / "vector_db")  # ChromaDB 저장 경로
collection_name = "docs"
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


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents




def make_prompt(document):
    """문서를 청킹하기 위한 LLM 프롬프트 생성 (25% 오버랩 포함 지시)"""
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1  # 문서 길이 기반 권장 청크 수
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


def make_messages(document):
    """LLM API 호출용 메시지 리스트 생성"""
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry(wait=wait)
def process_document(document):
    """단일 문서를 LLM으로 청킹하여 Result 리스트로 반환 (재시도 포함)"""
    messages = make_messages(document)
    # Structured Output (response_format)으로 Chunks 스키마 강제
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks



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



def run_ingest():
    """외부에서 호출 가능한 ingest 파이프라인"""
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    return len(documents), len(chunks)


if __name__ == "__main__":
    run_ingest()
    print("Ingestion complete")
