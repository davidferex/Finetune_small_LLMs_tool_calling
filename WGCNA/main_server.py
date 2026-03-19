import os
import sys
from gene_tools4 import mcp

if __name__ == "__main__":
    # Eliminamos el print de "🚀 MCP server running..."
    mcp.run(transport="stdio")