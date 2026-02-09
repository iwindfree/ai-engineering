# RAG 고객 상담 챗봇 예제

LangChain과 ChromaDB를 사용한 RAG(Retrieval-Augmented Generation) 기반 고객 상담 챗봇입니다.

## 프로젝트 구조

```
example/
├── app.py                    # Gradio 웹 애플리케이션
├── src/
│   ├── embed_documents.py    # 문서 임베딩 모듈
│   └── rag_chain.py          # RAG 체인 모듈
└── chroma_db/                # 벡터 DB (자동 생성)

# 지식베이스는 basic/rag_data/에 위치
basic/rag_data/
├── knowledge_base/           # 하늘여행사 지식베이스
│   ├── company/              # 회사 정보
│   ├── contracts/            # 계약 정보
│   ├── employees/            # 직원 정보
│   ├── faq/                  # FAQ
│   ├── policies/             # 정책 문서
│   └── products/             # 상품 정보
└── evaluation/               # RAG 평가용 데이터
    └── tests.jsonl           # 테스트 질문
```

## 실행 방법

### 1. 환경 설정

```bash
# .env 파일에 OpenAI API 키 설정
OPENAI_API_KEY=sk-your-api-key
```

### 2. 문서 임베딩

```bash
cd basic/day6/example
python src/embed_documents.py
```

### 3. 챗봇 실행

```bash
python app.py
```

브라우저에서 http://localhost:7860 접속
