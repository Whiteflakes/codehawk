"""FastAPI REST API for CodeHawk."""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from codehawk.context import ContextEngine
from codehawk.config import settings

logger = logging.getLogger(__name__)

# Global context engine instance
engine: Optional[ContextEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global engine
    logger.info("Starting CodeHawk API")
    engine = ContextEngine()
    engine.initialize()
    yield
    if engine:
        engine.shutdown()
    logger.info("Shutdown CodeHawk API")


app = FastAPI(
    title="CodeHawk API",
    description="Code context engine API",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    repository_id: Optional[int] = Field(None, description="Optional repository filter (deprecated, use repository_ids)")
    repository_ids: Optional[List[int]] = Field(None, description="Optional repository filters")
    languages: Optional[List[str]] = Field(None, description="Language filters")
    since: Optional[datetime] = Field(None, description="Only include chunks created after this timestamp")
    use_lexical: bool = Field(False, description="Blend vector search with lexical/BM25")
    lexical_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for lexical scoring")


class SearchResponse(BaseModel):
    """Search response model."""

    results: List[Dict[str, Any]]
    total: int


class ContextPackRequest(BaseModel):
    """Context pack request model."""

    query: str = Field(..., description="Query for context")
    limit: int = Field(5, ge=1, le=20, description="Number of chunks to include")
    include_relations: bool = Field(True, description="Include related chunks")
    include_lineage: bool = Field(True, description="Include commit lineage")


class ContextPackResponse(BaseModel):
    """Context pack response model."""

    query: str
    chunks: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    lineage: List[Dict[str, Any]]


class IndexRequest(BaseModel):
    """Index repository request model."""

    repository_path: str = Field(..., description="Path to repository")
    repository_url: Optional[str] = Field(None, description="Repository URL")


class IndexResponse(BaseModel):
    """Index repository response model."""

    repository_id: int
    message: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "CodeHawk API",
        "version": "0.1.0",
        "description": "Open-source code context engine",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for code chunks.

    Args:
        request: Search request

    Returns:
        Search results
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        results = engine.search(
            query=request.query,
            limit=request.limit,
            repository_id=request.repository_id,
            repository_ids=request.repository_ids,
            languages=request.languages,
            since=request.since,
            use_lexical=request.use_lexical,
            lexical_weight=request.lexical_weight,
        )
        
        return SearchResponse(
            results=results,
            total=len(results),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context", response_model=ContextPackResponse)
async def get_context(request: ContextPackRequest):
    """
    Get context pack for LLM.

    Args:
        request: Context pack request

    Returns:
        Context pack
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        context_pack = engine.get_context_pack(
            query=request.query,
            limit=request.limit,
            include_relations=request.include_relations,
            include_lineage=request.include_lineage,
        )
        
        return ContextPackResponse(**context_pack)
    except Exception as e:
        logger.error(f"Context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_repository(request: IndexRequest):
    """
    Index a repository.

    Args:
        request: Index request

    Returns:
        Index response
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        repo_path = Path(request.repository_path)
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail="Repository path not found")

        repository_id = engine.index_repository(repo_path, request.repository_url)
        
        return IndexResponse(
            repository_id=repository_id,
            message=f"Successfully indexed repository: {repo_path}",
        )
    except Exception as e:
        logger.error(f"Index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
