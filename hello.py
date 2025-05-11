import os
import chainlit as cl

from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Model
model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash",
)

# Config: Defined at Run Level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Step 3: Agent
agent1 = Agent(
    instructions="You are a helpful assistant that can answer questions and Stream responses in real-time as they're generated.",
    name="Gemini Assistant",
)

# Step 4: Runner
result = Runner.run_sync(
    input=history,
    run_config=run_config,
    starting_agent=agent1,
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome to the Gemini Assistant! How can I help you today?",
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    msg = cl.Message(content="")
    await msg.send()

# Standard Interface [{"role":"user","content":"Hello!"}, role":"assistant","content":"Hello! How can I assist you today?"}]
    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )

    async for event in result.output_stream():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        await msg.stream(event.data.delta)

    # history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(
        content=result.final_output,
    ).send()




