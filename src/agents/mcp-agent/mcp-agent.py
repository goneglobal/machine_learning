# AI Agent Using an MCP Server

# ---- requirements ----
# pip install mcp-client openai

from mcp_client import MCPClient
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Connect to an MCP server running locally (e.g., exposes search, calendar, etc.)
mcp = MCPClient("ws://localhost:8000")   # or mcp.connect_stdio("path/to/server")

# Discover available tools
tools = mcp.list_tools()
print("Tools available from MCP server:", tools)

def agent_loop(user_question: str):
    prompt = f"You are a helpful agent. The user asked: '{user_question}'.\n" \
             f"You can call any of these tools: {tools}"

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        tools=tools       # tools discovered at runtime!
    )

    if response.output[0].type == "tool_call":
        tool_name = response.output[0].content[0].name
        params = response.output[0].content[0].input
        result = mcp.call_tool(tool_name, params)
        print(f"[Tool Result from MCP] {result}")
    else:
        print(response.output[0].content[0].text)

agent_loop("Who is the current Prime Minister of Australia?")
