### check chuk-llm & ollama are working
# uvx chuk-llm test ollama

### initialize project
# uv init

### Add chuk-llm to venv
# uv add chuk-llm

## Pull an LLM & test
# ollama pull granite3.3
# ollama run granite3.3 "hi"

## HOW TO RUN THIS
# uv run python main-async.py

## imports
import asyncio
from chuk_llm import stream_ollama_granite

async def stream_example():
    print("Hello from uv-granite3-3!")
    persona = "you are a rapper call All Pink who always speaks like the Hood"
    
    async for chunk in stream_ollama_granite("describe a bird party", system_prompt = persona):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(stream_example())
