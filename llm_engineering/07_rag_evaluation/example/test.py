import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent.parent.parent / "00_test_data" / "test_dataset" / "tests.jsonl")

class TestQuestion(BaseModel):
    """RAG 평가용 테스트 질문 모델"""

    question: str = Field(description="RAG 시스템에 던질 질문")
    keywords: list[str] = Field(description="검색 결과에 포함되어야 하는 키워드들")
    reference_answer: str = Field(description="정답 (Ground Truth)")
    category: str = Field(description="질문 카테고리")
    source_docs: list[str] = Field(default_factory=list, description="정답이 포함된 원본 문서 경로들")

def load_tests() -> list[TestQuestion]:
    """Load test questions from the JSONL file."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tests.append(TestQuestion(**data))
    return tests

def main():
    tests = load_tests()
    print(f"Loaded {len(tests)} tests")
    test = tests[0]
    print(f"Question: {test.question}")
    print(f"Expected Keywords: {test.keywords}")
    print(f"Reference Answer: {test.reference_answer}")
    print(f"Category: {test.category}")
    print(f"Source Docs: {test.source_docs}")


if __name__ == "__main__":
    main()
