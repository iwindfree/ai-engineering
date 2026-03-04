from mail_tools.mail_tool import create_email_generator_tools, send_email, send_html_email
from mail_factory import AgentFactory
from agents import trace, Runner
from dotenv import load_dotenv


class SalesManager:
    def __init__(self):
        load_dotenv(override=True)
        generator_tools = create_email_generator_tools()
        self.agent = AgentFactory.create_agent("sales_manager", tools=generator_tools + [send_email])

    async def send_email(self):
        message = "'대표님께' 로 시작하는 콜드 영업 이메일을 보내주세요. 발신자는 '김영업'입니다."
        with trace("영업 매니저 파이프라인"):
            result = await Runner.run(self.agent, message, max_turns=10)
        print(f"\n{'='*60}\n[최종 결과]\n{'='*60}\n{result.final_output}")


class SalesManagerWithHandoff:
    def __init__(self):
        load_dotenv(override=True)
        generator_tools = create_email_generator_tools()
        emailer_agent = self._create_emailer_agent()
        self.agent = AgentFactory.create_agent("sales_manager_v2", tools=generator_tools, handoffs=[emailer_agent])

    def _create_emailer_agent(self):
        subject_writer = AgentFactory.create_agent("mail_subject_writer")
        html_converter = AgentFactory.create_agent("html_converter")
        subject_tool = subject_writer.as_tool(
            tool_name="subject_writer",
            tool_description="콜드 영업 이메일의 제목을 작성합니다.",
        )
        html_tool = html_converter.as_tool(
            tool_name="html_converter",
            tool_description="텍스트 이메일 본문을 HTML 이메일로 변환합니다.",
        )
        return AgentFactory.create_agent("emailer_agent", tools=[subject_tool, html_tool, send_html_email])

    async def send_email(self):
        message = "'대표님께' 로 시작하는 콜드 영업 이메일을 보내주세요. 발신자는 '김영업'입니다."
        with trace("영업 매니저 V2 파이프라인 (핸드오프)"):
            result = await Runner.run(self.agent, message, max_turns=10)
        print(f"\n{'='*60}\n[최종 결과]\n{'='*60}\n{result.final_output}")
