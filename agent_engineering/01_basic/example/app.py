from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import gradio as gr


load_dotenv(override=True)

# 알림 함수 — Pushover 설정 시 푸시 알림, 아니면 print 로깅
pushover_user = os.getenv("PUSHOVER_USER", "")
pushover_token = os.getenv("PUSHOVER_TOKEN", "")
pushover_enabled = bool(pushover_user and pushover_token)


def push(text):
    """알림 전송"""
    print(f"[알림] {text}", flush=True)
    if pushover_enabled:
        try:
            import requests
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={"token": pushover_token, "user": pushover_user, "message": text},
            )
        except Exception as e:
            print(f"  → Pushover 전송 실패: {e}")


# --- Tool 함수 ---

def record_user_details(email, name="이름 미제공", notes=""):
    """관심 있는 사용자의 연락처 정보를 기록합니다."""
    push(f"새로운 관심 사용자! 이름: {name}, 이메일: {email}, 메모: {notes}")
    return {"status": "success", "message": f"{name}님의 정보가 기록되었습니다."}


def record_unknown_question(question):
    """챗봇이 답변할 수 없는 질문을 기록합니다."""
    push(f"답변 불가 질문: {question}")
    return {"status": "logged", "message": "질문이 기록되었습니다."}


# --- Tool JSON 스키마 ---

record_user_details_json = {
    "name": "record_user_details",
    "description": "사용자가 연락처 정보(이메일 등)를 제공하거나, 연락 받기를 원하거나, 관심을 표현할 때 이 함수를 호출하세요.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "사용자의 이메일 주소",
            },
            "name": {
                "type": "string",
                "description": "사용자의 이름 (선택)",
            },
            "notes": {
                "type": "string",
                "description": "추가 메모 — 관심 분야, 대화 맥락 등",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "프로필 정보에 없는 내용을 질문받아 답변할 수 없을 때 이 함수를 호출하세요.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "답변할 수 없는 질문의 내용",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


# --- 커리어 어시스턴트 클래스 ---


class CareerAssistant:

    def __init__(self):
        self.openai = OpenAI()
        self.model = "gpt-4o-mini"

        # 프로필 데이터 — 자신의 정보로 교체하세요!
        self.name = "김철수"

        self.linkedin = """김철수
시니어 소프트웨어 엔지니어 | AI/ML 전문가

경력:
- ABC테크 (2020-현재): AI 플랫폼 개발 리드
  - LLM 기반 고객 서비스 자동화 시스템 구축
  - RAG 파이프라인 설계 및 운영
  - 팀 규모: 5명 → 12명 성장 리드

- XYZ소프트 (2017-2020): 백엔드 개발자
  - Python/FastAPI 기반 마이크로서비스 아키텍처 구축
  - 실시간 데이터 파이프라인 개발

학력:
- 서울대학교 컴퓨터공학과 석사 (2017)
- 서울대학교 컴퓨터공학과 학사 (2015)

기술 스택:
Python, PyTorch, LangChain, FastAPI, PostgreSQL, Docker, AWS

관심 분야:
LLM 에이전트, RAG 시스템, MLOps"""

        self.summary = """김철수는 AI와 소프트웨어 엔지니어링에 열정적인 개발자입니다.
현재 LLM 기반 에이전트 시스템 개발에 집중하고 있으며,
실제 비즈니스 문제를 AI로 해결하는 것에 관심이 있습니다.
기술 커뮤니티 활동과 오픈소스 기여를 즐기며,
새로운 기술을 배우고 공유하는 것을 좋아합니다."""

    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"  [Tool 호출] {tool_name}({arguments})", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "tool_call_id": tool_call.id,
            })
        return results

    def system_prompt(self):
        return f"""당신은 {self.name}을 대신하여 대화하는 AI 어시스턴트입니다.
당신의 역할은 {self.name}의 경력, 기술, 경험에 대한 질문에 친절하고 전문적으로 답변하는 것입니다.

## 참고 정보

### 이력/프로필
{self.linkedin}

### 추가 요약
{self.summary}

## 행동 규칙

1. 위 정보를 기반으로 {self.name}에 대한 질문에 성실히 답변하세요.
2. 답변은 항상 긍정적이고 전문적인 톤을 유지하세요.
3. 사용자가 연락처(이메일 등)를 제공하면 record_user_details 함수를 호출하세요.
4. 프로필에 없는 정보를 질문받으면 record_unknown_question 함수를 호출한 후, 모른다고 솔직히 답변하세요.
5. 한국어로 답변하세요.
"""

    def chat(self, message, history):
        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + history
            + [{"role": "user", "content": message}]
        )
        while True:
            response = self.openai.chat.completions.create(
                model=self.model, messages=messages, tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                results = self.handle_tool_calls(msg.tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                return response.choices[0].message.content


if __name__ == "__main__":
    assistant = CareerAssistant()
    gr.ChatInterface(
        assistant.chat,
        title=f"{assistant.name} 커리어 어시스턴트",
        description=f"{assistant.name}의 경력과 기술에 대해 물어보세요!",
        examples=[
            "어떤 경력을 가지고 계신가요?",
            "주요 기술 스택은 무엇인가요?",
            "AI 관련 프로젝트 경험이 있나요?",
        ],
    ).launch()
