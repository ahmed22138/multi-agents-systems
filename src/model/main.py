import asyncio
from agents import OpenAIChatCompletionsModel, RunConfig, Agent, Runner
from openai  import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    tracing_disabled=True
)

spanish = Agent(
    name="Spanish agent",
    instructions="You are a Spanish agent. Convert English text into Spanish."
)

urdu = Agent(
    name="Urdu agent",
    instructions="You are an Urdu agent. Convert English text into Urdu."
)



main_agent = Agent(
    name="Main agent",
    instructions="You are the main agent. Your task is to decide and handoff tasks to the right agent.",
    handoffs=[spanish, urdu]
)

async def main():
    results = await Runner.run(
        main_agent,
        "I love learning books into english",
        run_config=config
    )
    print(results.final_output)


def start():

    asyncio.run(main())
