"""
문서 임베딩 모듈

knowledge_base 폴더의 마크다운 문서를 읽어
벡터 데이터베이스(ChromaDB)에 저장합니다.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
# 환경 변수 로드
load_dotenv(override=True)

# 경로 설정
CURRENT_DIR = Path(__file__).parent
print(f"현재 디렉토리: {CURRENT_DIR}")
BASE_DIR = CURRENT_DIR.parent.parent.parent
print(f"베이스 디렉토리: {BASE_DIR}")
KNOWLEDGE_BASE_DIR = BASE_DIR / "00_test_data" /  "knowledge_base"
CHROMA_DB_DIR = BASE_DIR / "00_test_data" / "chroma_db"

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# 청킹 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents() -> list:
    """
    knowledge_base 폴더에서 모든 마크다운 문서를 로드합니다.
    각 문서에 카테고리 메타데이터를 추가합니다.
    """
    all_documents = []

    # knowledge_base 하위 폴더 순회
    for category_folder in KNOWLEDGE_BASE_DIR.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        print(f"📂 카테고리 로드 중: {category_name}")

        # 해당 카테고리의 마크다운 파일 로드
        loader = DirectoryLoader(
            str(category_folder),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        docs = loader.load()

        # 메타데이터에 카테고리 추가
        for doc in docs:
            doc.metadata["category"] = category_name
            doc.metadata["filename"] = Path(doc.metadata["source"]).name
            all_documents.append(doc)

        print(f"   └─ {len(docs)}개 문서 로드 완료")

    return all_documents


def split_documents(documents: list) -> list:
    """
    문서를 적절한 크기의 청크로 분할합니다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"📄 총 {len(chunks)}개 청크 생성")

    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    청크를 벡터화하여 ChromaDB에 저장합니다.
    """
    # 기존 DB가 있으면 삭제
    if CHROMA_DB_DIR.exists():
        print("🗑️ 기존 벡터 DB 삭제 중...")
        shutil.rmtree(CHROMA_DB_DIR)

    # 임베딩 모델 초기화
    print(f"🔧 임베딩 모델 로드 중: {EMBEDDING_MODEL_NAME}")
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 벡터 스토어 생성
    print("💾 벡터 DB 생성 중...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DB_DIR)
    )

    # 저장 정보 출력
    collection = vector_store._collection
    doc_count = collection.count()


    # 개선
    if doc_count > 0:
        sample = collection.get(limit=1, include=["embeddings"])
        embedding_dim = len(sample["embeddings"][0])
    else:
        embedding_dim = 0

    print(f"✅ 벡터 DB 생성 완료!")
    print(f"   └─ 벡터 수: {doc_count:,}개")
    print(f"   └─ 벡터 차원: {embedding_dim:,}")
    print(f"   └─ 저장 경로: {CHROMA_DB_DIR}")

    return vector_store


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("📚 문서 임베딩 시작")
    print("=" * 50)

    # 1. 문서 로드
    documents = load_documents()
    print(f"\n총 {len(documents)}개 문서 로드 완료\n")

    # 2. 청크 분할
    chunks = split_documents(documents)

    # 3. 벡터 스토어 생성
    create_vector_store(chunks)

    print("\n" + "=" * 50)
    print("✨ 임베딩 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
