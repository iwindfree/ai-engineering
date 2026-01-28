import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from test import TestQuestion, load_test
from answer import answer_question, fecth_context

load_dotenv(override=True)
MODEL = "gpt-4.1-nano"
db_name = "vector_db"

class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


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

