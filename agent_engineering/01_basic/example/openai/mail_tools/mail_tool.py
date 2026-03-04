from agents import function_tool
from mail_factory import AgentFactory

sent_emails = []


@function_tool
def send_email(to: str, subject: str, body: str) -> dict[str, str]:
    """영업 대상에게 이메일을 전송합니다."""
    email_record = {"to": to, "subject": subject, "body": body}
    sent_emails.append(email_record)
    print(f"\n{'='*50}")
    print(f"[이메일 전송됨]")
    print(f"수신: {to}")
    print(f"제목: {subject}")
    print(f"{'='*50}")
    print(body)
    print(f"{'='*50}\n")
    return {"status": "success", "message": f"Email sent to {to}."}


@function_tool
def send_html_email(to: str, subject: str, html_body: str) -> dict[str, str]:
    """제목과 HTML 본문으로 이메일을 전송합니다."""
    email_record = {"to": to, "subject": subject, "html_body": html_body}
    sent_emails.append(email_record)
    print(f"\n{'='*50}")
    print(f"[HTML 이메일 전송됨]")
    print(f"수신: {to}")
    print(f"제목: {subject}")
    print(f"{'='*50}")
    print(html_body[:500] + "..." if len(html_body) > 500 else html_body)
    print(f"{'='*50}\n")
    return {"status": "success", "message": f"{to}에게 HTML 이메일이 전송되었습니다."}


def create_email_generator_tools():
    """3종 이메일 생성 에이전트를 도구로 변환하여 반환"""
    desc = "콜드 영업 이메일을 작성합니다."
    professional = AgentFactory.create_agent("professional")
    humorous = AgentFactory.create_agent("humorous")
    concise = AgentFactory.create_agent("concise")
    tools = [
        professional.as_tool(tool_name="professional_email_generator", tool_description="전문적이고 격식 있는 " + desc),
        humorous.as_tool(tool_name="humorous_email_generator", tool_description="유머러스하고 친근한 " + desc),
        concise.as_tool(tool_name="concise_email_generator", tool_description="간결하고 핵심적인 " + desc),
    ]
    for t in tools:
        print(f"{'─'*60}\nRegistered tool: {t.name} - {t.description}\n{'─'*60}")
    return tools
