from mail_agents.mail_agent import MailSelectorAgent
from manager.sales_manager import SalesManager, SalesManagerWithHandoff
import asyncio


async def main():
    # await select_email()          # 병렬 생성 + 투표 선택
    # await send_email()            # 도구 방식 (V1)
    await send_email_with_handoff()  # 핸드오프 방식 (V2)


async def select_email():
    mail_selector_agent = MailSelectorAgent()
    await mail_selector_agent.select_email()


async def send_email():
    sales_manager = SalesManager()
    await sales_manager.send_email()


async def send_email_with_handoff():
    sales_manager = SalesManagerWithHandoff()
    await sales_manager.send_email()


if __name__ == "__main__":
    asyncio.run(main())
