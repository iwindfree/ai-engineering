import json
from pathlib import Path
from pydantic import BaseModel, Field

# 경로 설정
BASIC_DIR = Path(__file__).resolve().parents[3]  # basic/ 폴더
TEST_FILE = str(BASIC_DIR / "rag_data" / "evaluation" / "tests.jsonl")

class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")

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
    test = tests[0]
    print(f"Question: {test.question}")
    print(f"Expected Keywords: {test.keywords}")
    print(f"Reference Answer: {test.reference_answer}")
    print(f"Category: {test.category}")


if __name__ == "__main__":
    main()