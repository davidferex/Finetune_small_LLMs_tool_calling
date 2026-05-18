import os
import sys
from preset_tools import mcp

if __name__ == "__main__":
    # Eliminamos el print de "🚀 MCP server running..."
    mcp.run(transport="stdio")