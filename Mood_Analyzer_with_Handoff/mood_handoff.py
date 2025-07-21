from agents import  Agent , Runner , OpenAIChatCompletionsModel , RunConfig , AsyncOpenAI , function_tool
import os 
from dotenv import load_dotenv
import asyncio

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent1 = Agent(
    name="Mood Analyzer",
    instructions=("You are a mood analyzer that checks user's mood from their message eg.happy or sad or anything else"),
)

agent2 = Agent(
    name="Suggest Activity",
    instructions=("You are helpful assistant that suggests activity based on mood if sad , stressed or anything feels bad ")
)

async def main():
    user_input = input("🧑 How are you feeling today? ")

    result1 = await Runner.run(
        agent1,
        user_input,
        run_config=config,
    )
    mood_analysis = result1.final_output
    print("Mood Analysis:", mood_analysis)

    if "sad" in mood_analysis.lower() or "stressed" in mood_analysis.lower():
        result2 = await Runner.run(agent2, user_input, run_config=config)
        print("Suggested Activity:", result2.final_output )

asyncio.run(main())

