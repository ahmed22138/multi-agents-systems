import asyncio
from agents import OpenAIChatCompletionsModel, RunConfig, Agent, RunContextWrapper, Runner, function_tool, handoff
from openai  import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
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

class NewsRequest(BaseModel):
    topic: str
    reason: str

@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."

def on_news_transfer(ctx: RunContextWrapper, input_data: NewsRequest) -> None:
    print(f"\nTransferring to for news updates. input_data:", input_data, "\n")

news_agent: Agent = Agent(
    name="NewsAgent",
    instructions="You get latest news about tech community and share it with me.",
    tools=[get_weather],
)

weather_agent: Agent = Agent(
    name="WeatherAgent",
    instructions="You are weather expert - share weather updates as I travel a lot. For all Tech and News let the NewsAgent handle that part by delegation.",
    tools=[get_weather],
    handoffs=[handoff(agent=news_agent, on_handoff=on_news_transfer, input_type=NewsRequest)]
)

async def main():

 res =await Runner.run(weather_agent, "Check if there's any news about OpenAI after GPT-5 launch?", run_config=config)
 print("\nAGENT NAME", res.last_agent.name)
 print("\n[RESPONSE:]", res.final_output)

def start():
   asyncio.run(main())