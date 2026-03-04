from agents import Runner, trace
from dotenv import load_dotenv
from mail_factory import AgentFactory
import asyncio
import os


class MailSelectorAgent:
    def __init__(self):
        print("Initializing MailSelectorAgent...")

    async def select_email(self):
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("API key found.")
        else:
            print("Failed to load API key. Please check your .env file.")
            exit(1)

        professional_agent = AgentFactory.create_agent("professional")
        humorous_agent = AgentFactory.create_agent("humorous")
        concise_agent = AgentFactory.create_agent("concise")

        mail_picker_agent = AgentFactory.create_agent("mail_picker")
        
        message = "콜드 영업 이메일을 작성해주세요."
        with trace("병렬 콜드 영업 이메일 작성"):
            print(f"{'*' * 60}")
            print("Running agents in parallel...")
            print(f"{'*' * 60}")
            results = await asyncio.gather(
                Runner.run(professional_agent, message),
                Runner.run(humorous_agent, message),
                Runner.run(concise_agent, message)
            )
            outputs = [result.final_output for result in results]
            for idx, output in enumerate(outputs):
                print(f"\nAgent {idx + 1} Output:\n{output}")

            emails = "콜드영업 이메일 후보들:\n\n" + "\n\n ---이메일---\n\n".join(outputs)
            best_email = await Runner.run(mail_picker_agent, emails)

            print(f"{'*' * 60}")
            print(f"\nBest Email Selected:\n{best_email.final_output}")
            print(f"{'*' * 60}")


