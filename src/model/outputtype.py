import asyncio
from agents import OpenAIChatCompletionsModel, RunConfig, Agent, Runner, function_tool
from openai  import AsyncOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel


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



class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)



async def main():


  result = await Runner.run(
        agent,
        "Extract calendar events from the following text: 'Meeting with Alice on 2024-07-01 and Bob on 2024-07-02.'",
        run_config=config
    )

  print(result.final_output)

def start():
    asyncio.run(main())