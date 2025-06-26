from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import pandas as pd
from mindsdb import initialize_mindsdb, server, kb_initialized
from models import HealthResponse, SemanticSearchRequest, SemanticSearchResponse
import openai
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = openai.OpenAI(api_key=openai_api_key)

# Startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: Initializing MindsDB...")
    await initialize_mindsdb()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Movie Semantic Search API",
    description="API for semantic search over IMDB movie database using MindsDB and OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

async def answer_question_with_llm(question: str, limit: int = 100) -> str:
    """Perform semantic search using MindsDB and generate an answer with OpenAI GPT-4o"""
    if not server or not kb_initialized:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    try:
        # Step 1: Query MindsDB for relevant chunks by replacing the single quotes in the question with double single quotes
        search_query = f"""
            SELECT id as movie_id, 
                   chunk_content as content,
                   metadata_json->>'$.genre' as genre,
                   metadata_json->>'$.expanded_genres' as expanded_genres,
                   CAST(metadata_json->>'.rating' as DECIMAL(3,1)) as rating,
                   relevance
            FROM movies_kb 
            WHERE content = '{question.replace("'", "''")}' 
            ORDER BY relevance DESC 
            LIMIT {limit}
        """
        relevant_chunks_df = server.query(search_query).fetch()
        logger.info(f"Found {len(relevant_chunks_df)} relevant chunks for query: {question}")

        # Step 2: Concatenate chunk_content to form context
        context = "\n---\n".join(relevant_chunks_df['content'])

        # Step 3: Create prompt for GPT-4o
        prompt = f"""
        You are a movie expert assistant. Based *only* on the following movie summaries (context),
        answer the user's question. If the context doesn't contain the answer,
        state that you cannot answer based on the provided information.

        CONTEXT:
        {context}

        QUESTION:
        {question}
        """

        # Step 4: Call OpenAI API
        logger.info("Sending request to GPT-4o...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about movies using only the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Semantic Search API",
        "endpoints": {
            "semantic_search": "/semantic-search - POST - Semantic search for movie-related questions",
            "health": "/health - GET - Check API and knowledge base status",
            "search_examples": "/search/examples - GET - Example search queries"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not server:
        return HealthResponse(status="unhealthy", kb_status="disconnected")
    
    if not kb_initialized:
        return HealthResponse(status="degraded", kb_status="not_initialized")
    
    try:
        row_count_df = server.query("""
            SELECT COUNT(*) AS cnt
            FROM (SELECT id FROM movies_kb) AS t;
        """).fetch()
        row_count = int(row_count_df.at[0, 'cnt'])
        return HealthResponse(status="healthy", kb_status="ready", kb_row_count=row_count)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(status="degraded", kb_status="error")

@app.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Perform semantic search on the movie database using OpenAI GPT-4o
    
    - **query**: The search question (e.g., "Who a boy must defend his home against on Christmas eve?")
    - **limit**: Maximum number of context chunks to retrieve (default: 100)
    """
    if not kb_initialized:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")
    
    try:
        answer = await answer_question_with_llm(request.query, request.limit)
        return SemanticSearchResponse(query=request.query, answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.get("/search/examples")
async def search_examples():
    """Get example search queries"""
    return {
        "examples": [
            "Who a boy must defend his home against on Christmas eve?",
            "What Anakin was lured into by Chancellor Palpatine?",
            "Christmas movies with family themes",
            "Space adventure and sci-fi",
            "Romantic comedies",
            "Action movies with superheroes",
            "Horror movies with supernatural elements",
            "Drama about family relationships",
            "Movies about time travel",
            "War movies based on true stories"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")