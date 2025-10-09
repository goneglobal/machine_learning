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
# uv run python main.py

## imports
from chuk_llm import ask_ollama_granite

def main():
    print("Hello from uv-granite3-3!")
    response = ask_ollama_granite("what is at the center of the earth")
    print(response)

if __name__ == "__main__":
    main()
