import asyncio
import logging
from agents import OpenAIChatCompletionsModel, RunConfig, Agent, Runner, enable_verbose_stdout_logging
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

# ✅ Step 1: Enable verbose logging (simple way)
enable_verbose_stdout_logging()

# ✅ Step 2: (optional) Customize logger
logger = logging.getLogger("openai.agents")
logger.setLevel(logging.DEBUG)  # sab kuch show karega
logger.addHandler(logging.StreamHandler())

# ✅ Step 3: Create a simple agent
agent = Agent(
    name="demo_agent",
    instructions="You are a helpful assistant. Always reply clearly."
)

async  def  main():

    res = await Runner.run(
        agent,
        "hello agent,I'm Ahmed and  how are you & can you guide me for what is python in haikus.",
        run_config=config
    )

    print(res.final_output)

def start():
    asyncio.run(main())