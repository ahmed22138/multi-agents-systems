import asyncio
from agents import OpenAIChatCompletionsModel, RunConfig, Agent,  Runner, ModelSettings
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

base_agent =  Agent(
    name="BaseAssistant",
    instructions="You are a helpful Assistant.",
    model_settings=ModelSettings(temperature=0.1)
)

friendly_agent = base_agent.clone(
    name="FriendlyAssistant",
    instructions="You are a friendly and warm Assistant.",
    model_settings=ModelSettings(temperate=0.9)
    
)

query  = "Hello, I'm Ahmed and how are you."

async def main():

    res = await Runner.run(
        base_agent,
        query,
        run_config=config
    )

    res1 = await Runner.run(
        friendly_agent,
        query,
        run_config=config
    )

    print(res.final_output,"/n")
    print(res1.final_output)

def start():
    asyncio.run(main())
