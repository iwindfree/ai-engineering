import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from test import TestQuestion, load_tests
from answer import answer_question_basic, fetch_context_multi, RETRIEVAL_K

load_dotenv(override=True)
MODEL = "openai/gpt-4.1-nano"

class RetrievalEval(BaseModel):
    """검색 평가 결과 모델"""

    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg: float = Field(description="Normalized DCG")
    precision_at_k: float = Field(description="Precision@K")
    recall_at_k: float = Field(description="Recall@K")
    keywords_found: int = Field(description="찾은 키워드 수")
    total_keywords: int = Field(description="전체 키워드 수")
    keyword_coverage: float = Field(description="키워드 커버리지 (0~1)")


class AnswerEval(BaseModel):
    """LLM-as-a-judge 방식의 답변 품질 평가 결과."""

    feedback: str = Field(
        description="답변 품질에 대한 간결한 피드백. 참조 답변과 비교하고 검색된 컨텍스트를 기반으로 평가"
    )
    accuracy: float = Field(
        description="참조 답변 대비 사실적 정확도. 1점(오답 - 틀린 답변은 반드시 1점)부터 5점(이상적 - 완벽히 정확)까지. 허용 가능한 답변은 3점"
    )
    completeness: float = Field(
        description="질문의 모든 측면을 얼마나 완전히 다루는가. 1점(매우 부족 - 핵심 정보 누락)부터 5점(이상적 - 참조 답변의 모든 정보 완벽 포함)까지. 참조 답변의 모든 정보가 포함된 경우에만 5점"
    )
    relevance: float = Field(
        description="질문에 대한 답변의 관련성. 1점(매우 부족 - 주제 벗어남)부터 5점(이상적 - 질문에 직접 답하며 불필요한 정보 없음)까지. 질문에 완전히 관련되고 추가 정보가 없는 경우에만 5점"
    )


def normalize_source(source_path: str) -> str:
    """청크 metadata의 source 경로를 정규화하여 knowledge_base/ 이하 상대 경로로 변환"""
    if "knowledge_base/" in source_path:
        return source_path.split("knowledge_base/")[-1]
    return source_path


def calculate_mrr(source_docs: list[str], retrieved_docs: list) -> float:
    """문서 ID 기반 MRR(Reciprocal Rank) 계산"""
    source_set = set(source_docs)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if normalize_source(doc.metadata.get("source", "")) in source_set:
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(source_docs: list[str], retrieved_docs: list, k: int = 10) -> float:
    """문서 ID 기반 nDCG 계산"""
    source_set = set(source_docs)
    relevances = [
        1 if normalize_source(doc.metadata.get("source", "")) in source_set else 0
        for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    idcg = calculate_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(source_docs: list[str], retrieved_docs: list, k: int) -> float:
    """Precision@K: 상위 K개 청크 중 정답 문서에서 온 청크의 비율"""
    if k <= 0:
        return 0.0
    source_set = set(source_docs)
    top_k = retrieved_docs[:k]
    relevant = sum(
        1 for doc in top_k
        if normalize_source(doc.metadata.get("source", "")) in source_set
    )
    return relevant / k


def calculate_recall_at_k(source_docs: list[str], retrieved_docs: list, k: int) -> float:
    """Recall@K: 정답 문서 중 상위 K개에서 1개라도 청크가 검색된 문서의 비율"""
    if not source_docs:
        return 1.0
    source_set = set(source_docs)
    found_sources = set(
        normalize_source(doc.metadata.get("source", ""))
        for doc in retrieved_docs[:k]
    )
    found = len(source_set & found_sources)
    return found / len(source_set)


def evaluate_retrieval(test: TestQuestion, k: int = RETRIEVAL_K) -> RetrievalEval:
    """
    단일 테스트에 대한 검색 평가 수행 (Reranking 포함)

    검색 파이프라인: 벡터 유사도 검색 → Reranking → 평가

    Args:
        test: TestQuestion object containing question, source_docs and keywords
        k: Number of top documents to retrieve

    Returns:
        RetrievalEval object with MRR, nDCG, Precision@K, Recall@K, and keyword coverage metrics
    """
    # Retrieve documents using basic strategy (original query only)
    retrieved_docs = fetch_context_multi(test.question, [test.question])

    # 문서 ID 기반 지표
    mrr = calculate_mrr(test.source_docs, retrieved_docs)
    ndcg = calculate_ndcg(test.source_docs, retrieved_docs, k)
    precision = calculate_precision_at_k(test.source_docs, retrieved_docs, k)
    recall = calculate_recall_at_k(test.source_docs, retrieved_docs, k)

    # 키워드 기반 보조 지표
    all_content = " ".join(doc.page_content.lower() for doc in retrieved_docs[:k])
    keywords_found = sum(1 for kw in test.keywords if kw.lower() in all_content)
    total_keywords = len(test.keywords)
    coverage = keywords_found / total_keywords if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=float(mrr), ndcg=float(ndcg),
        precision_at_k=float(precision), recall_at_k=float(recall),
        keywords_found=keywords_found, total_keywords=total_keywords,
        keyword_coverage=coverage,
    )

def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    # Get RAG response using basic strategy
    generated_answer, retrieved_docs = answer_question_basic(test.question)

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    # Call LLM judge with structured outputs
    judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)

    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return answer_eval, generated_answer, retrieved_docs



def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """Evaluate all answers to tests using batched async execution."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test (async helper for CLI)."""
    # Load tests
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Get the test
    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Source Docs: {test.source_docs}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Precision@K: {retrieval_result.precision_at_k:.4f}")
    print(f"Recall@K: {retrieval_result.recall_at_k:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1%}")

    # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage: python eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
