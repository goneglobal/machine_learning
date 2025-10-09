from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import logging
import math
import os
import random
import httpx
import subprocess
import json
from functools import wraps

load_dotenv("../.env")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP_Server")

# --- Environment variables ---
MCP_NAME = os.getenv("MCP_NAME", "UltimateMCP")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", 8050))
API_KEY = os.getenv("MCP_API_KEY", "changeme")

# --- Create MCP server ---
mcp = FastMCP(name=MCP_NAME, host=MCP_HOST, port=MCP_PORT)

# --- Authentication decorator ---
def require_api_key(func):
    @wraps(func)  # <--- preserves original name & docstring
    async def wrapper(*args, api_key=None, **kwargs):
        if api_key != API_KEY:
            logger.warning("Unauthorized access attempt")
            raise PermissionError("Invalid API key")
        return await func(*args, **kwargs) if callable(func) else func(*args, **kwargs)
    return wrapper

# -------------------------------
# Numeric tools
# -------------------------------
@mcp.tool()
async def add(a: float, b: float) -> float:
    return a + b

@mcp.tool()
async def subtract(a: float, b: float) -> float:
    return a - b

@mcp.tool()
@require_api_key
async def multiply(a: float, b: float) -> float:
    return a * b

@mcp.tool()
@require_api_key
async def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.tool()
@require_api_key
async def sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("Cannot take square root of negative number")
    return math.sqrt(x)

@mcp.tool()
@require_api_key
async def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Cannot take factorial of negative number")
    return math.factorial(n)

# -------------------------------
# String tools
# -------------------------------
@mcp.tool()
@require_api_key
async def reverse_text(text: str) -> str:
    return text[::-1]

@mcp.tool()
@require_api_key
async def uppercase(text: str) -> str:
    return text.upper()

@mcp.tool()
@require_api_key
async def word_count(text: str) -> int:
    return len(text.split())

@mcp.tool()
@require_api_key
async def is_palindrome(text: str) -> bool:
    clean_text = ''.join(filter(str.isalnum, text)).lower()
    return clean_text == clean_text[::-1]

# -------------------------------
# Utility tools
# -------------------------------
@mcp.tool()
@require_api_key
async def random_choice(options: list) -> str:
    if not options:
        raise ValueError("Options list cannot be empty")
    return random.choice(options)

# -------------------------------
# External tools
# -------------------------------
@mcp.tool()
@require_api_key
async def get_crypto_price(symbol: str, convert: str = "USD") -> float:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies={convert}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10)
        data = response.json()
    if symbol in data and convert.lower() in data[symbol]:
        return data[symbol][convert.lower()]
    else:
        raise ValueError("Invalid symbol or conversion currency")

@mcp.tool()
@require_api_key
async def run_shell_command(command: str) -> str:
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return result
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.output}"

@mcp.tool()
@require_api_key
async def read_json_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    with open(file_path, "r") as f:
        return json.load(f)

@mcp.tool()
@require_api_key
async def write_json_file(file_path: str, data: dict) -> str:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return f"Written to {file_path}"

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    logger.info(f"Starting MCP server '{MCP_NAME}' on {MCP_HOST}:{MCP_PORT} with SSE transport")
    mcp.run(transport="sse")
