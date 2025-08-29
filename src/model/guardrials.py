import asyncio
from agents import GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OpenAIChatCompletionsModel, OutputGuardrailTripwireTriggered, RunConfig, Agent, RunContextWrapper, Runner, TResponseInputItem, function_tool, input_guardrail, output_guardrail
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


class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

# Guardrail Agent (detect if input is math homework request)
guardrail_agent_input = Agent(
    name="Guardrail check (Input)",
    instructions="Check if the user is asking for math homework help.",
    output_type=MathHomeworkOutput,
)

@input_guardrail
async def math_input_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent_input, input, context=ctx.context,run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_homework,
    )


# ------------------------
# Output Guardrail Example
# ------------------------
class MessageOutput(BaseModel):
    response: str

class MathOutput(BaseModel):
    reasoning: str
    is_math: bool

# Guardrail Agent (detect if output contains math content)
guardrail_agent_output = Agent(
    name="Guardrail check (Output)",
    instructions="Check if the output contains any math solution.",
    output_type=MathOutput,
    
)

@output_guardrail
async def math_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent_output, output.response, context=ctx.context,run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )


# ------------------------
# Main Agent with Guardrails
# ------------------------
agent = Agent(
    name="Customer Support Agent",
    instructions="You are a customer support agent. Help customers with questions (not math homework!).",
    input_guardrails=[math_input_guardrail],
    output_guardrails=[math_output_guardrail],
    output_type=MessageOutput,
)


# ------------------------
# Run Test
# ------------------------
async def main():

  
    try:
        print("\n--- Test 1: Safe input ---")
        result = await Runner.run(agent, "Hello, I have a billing issue.",run_config=config)
        print("Agent output:", result.final_output.response)

        print("\n--- Test 2: Math Homework input ---")
        await Runner.run(agent, "Can you solve 2x + 3 = 11?",run_config=config)
    except InputGuardrailTripwireTriggered:
        print("🚨 Input Guardrail Blocked: Math homework detected!")

    try:
        print("\n--- Test 3: Math Output ---")
        # Forcing math output (simulate)
        result = await Runner.run(agent, "Please explain a math solution in your answer.",run_config=config)
        print("Agent output:", result.final_output.response)
    except OutputGuardrailTripwireTriggered:
        print("🚨 Output Guardrail Blocked: Math content detected!")


def start():
    asyncio.run(main())