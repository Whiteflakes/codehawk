"""Model Context Protocol (MCP) server for CodeHawk."""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from codehawk.context import ContextEngine
from codehawk.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeHawk MCP Server",
    description="Model Context Protocol server for code context",
    version="0.1.0",
)

# Global context engine instance
engine: Optional[ContextEngine] = None


class MCPRequest(BaseModel):
    """MCP request model."""

    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


class MCPResponse(BaseModel):
    """MCP response model."""

    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the context engine on startup."""
    global engine
    logger.info("Starting CodeHawk MCP server")
    engine = ContextEngine()
    engine.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the context engine."""
    global engine
    if engine:
        engine.shutdown()
    logger.info("Shutdown CodeHawk MCP server")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "CodeHawk MCP Server",
        "version": "0.1.0",
        "protocol": "MCP",
    }


@app.websocket("/mcp")
async def mcp_endpoint(websocket: WebSocket):
    """
    MCP WebSocket endpoint.

    Handles MCP protocol messages over WebSocket.
    """
    await websocket.accept()
    logger.info("MCP client connected")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                request = MCPRequest(**request_data)
                
                # Process request
                response = await process_mcp_request(request)
                
                # Send response
                await websocket.send_text(response.model_dump_json())
            
            except json.JSONDecodeError as e:
                error_response = MCPResponse(
                    error={"code": -32700, "message": "Parse error"},
                )
                await websocket.send_text(error_response.model_dump_json())
            
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_response = MCPResponse(
                    error={"code": -32603, "message": "Internal error"},
                    id=request.id if 'request' in locals() else None,
                )
                await websocket.send_text(error_response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("MCP client disconnected")


async def process_mcp_request(request: MCPRequest) -> MCPResponse:
    """
    Process an MCP request.

    Args:
        request: MCP request

    Returns:
        MCP response
    """
    if not engine:
        return MCPResponse(
            error={"code": -32603, "message": "Engine not initialized"},
            id=request.id,
        )

    try:
        if request.method == "search":
            # Search for code chunks
            query = request.params.get("query", "")
            limit = request.params.get("limit", 10)
            repository_id = request.params.get("repository_id")
            
            results = engine.search(query, limit, repository_id)
            
            return MCPResponse(
                result={"chunks": results},
                id=request.id,
            )

        elif request.method == "get_context":
            # Get context pack
            query = request.params.get("query", "")
            limit = request.params.get("limit", 5)
            include_relations = request.params.get("include_relations", True)
            include_lineage = request.params.get("include_lineage", True)
            
            context_pack = engine.get_context_pack(
                query,
                limit,
                include_relations,
                include_lineage,
            )
            
            return MCPResponse(
                result=context_pack,
                id=request.id,
            )

        elif request.method == "index_repository":
            # Index a repository
            repo_path = Path(request.params.get("repository_path", ""))
            repo_url = request.params.get("repository_url")
            
            if not repo_path.exists():
                return MCPResponse(
                    error={"code": -32602, "message": "Repository path not found"},
                    id=request.id,
                )
            
            repository_id = engine.index_repository(repo_path, repo_url)
            
            return MCPResponse(
                result={"repository_id": repository_id},
                id=request.id,
            )

        elif request.method == "list_methods":
            # List available methods
            methods = [
                {
                    "name": "search",
                    "description": "Search for code chunks",
                    "params": ["query", "limit", "repository_id"],
                },
                {
                    "name": "get_context",
                    "description": "Get context pack for LLM",
                    "params": ["query", "limit", "include_relations", "include_lineage"],
                },
                {
                    "name": "index_repository",
                    "description": "Index a repository",
                    "params": ["repository_path", "repository_url"],
                },
                {
                    "name": "list_methods",
                    "description": "List available methods",
                    "params": [],
                },
            ]
            
            return MCPResponse(
                result={"methods": methods},
                id=request.id,
            )

        else:
            return MCPResponse(
                error={"code": -32601, "message": f"Method not found: {request.method}"},
                id=request.id,
            )

    except Exception as e:
        logger.error(f"Error processing MCP request: {e}")
        return MCPResponse(
            error={"code": -32603, "message": str(e)},
            id=request.id,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.mcp_host, port=settings.mcp_port)
