# app.py
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import logging
import uvicorn
from datetime import datetime
import os
from rag.rag_pipeline import CourseRAGPipeline
from llm_service import LLMService

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_logs.log')  # log output
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Initialize FastAPI app
app = FastAPI(
    title="Brown Courses RAG API",
    description="Semantic course search and question-answering API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",        
        "http://localhost:8000",   
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_pipeline = None
llm_service = None
start_time = None
query_history = []
MAX_HISTORY_SIZE = 1000

# Request/Response Models
class QueryRequest(BaseModel):
    q: str
    department: Optional[str] = None
    source: Optional[str] = None
    instructor: Optional[str] = None
    k: Optional[int] = 5
    semantic_weight: Optional[float] = 0.7
    keyword_weight: Optional[float] = 0.3

class CourseResult(BaseModel):
    course_code: str
    title: str
    department: str
    similarity: float
    source: str
    description: Optional[str] = None
    instructor: Optional[str] = None
    meeting_times: Optional[str] = None
    prerequisites: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    courses: List[CourseResult]
    retrieval_count: int
    latency_ms: float
    timestamp: str

class EvaluationMetrics(BaseModel):
    total_queries: int
    average_latency_ms: float
    total_retrievals: int
    average_retrievals_per_query: float
    uptime_seconds: float

class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    llm_loaded: bool
    timestamp: str
    total_courses: int

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline and LLM service on startup"""
    global rag_pipeline, llm_service, start_time
    
    logger.info("Starting up application...")
    start_time = time.time()
    
    try:
        data_dir = os.path.join(PROJECT_ROOT, "data")
        embeddings_dir = os.path.join(PROJECT_ROOT, "rag", "embeddings")

        csv_path = os.path.join(data_dir, "courses.csv")
        index_path = os.path.join(embeddings_dir, "faiss_index.bin")
        metadata_path = os.path.join(embeddings_dir, "metadata.pkl")

        logger.info(f"Using data path: {csv_path}")
        logger.info(f"Using index path: {index_path}")
        logger.info(f"Using metadata path: {metadata_path}")
        
        # Initialize RAG pipeline
        logger.info("Loading RAG pipeline...")
        rag_pipeline = CourseRAGPipeline(
            csv_path=csv_path,
            model_name="BAAI/bge-base-en-v1.5",
            index_path=index_path,
            metadata_path=metadata_path
        )
        rag_pipeline.load_index()
        logger.info(f"RAG pipeline loaded successfully with {len(rag_pipeline.df)} courses")

        # Initialize LLM service
        logger.info("Loading LLM service...")
        llm_service = LLMService(
            provider="transformers",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        logger.info("LLM service loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Project root: {PROJECT_ROOT}")
        # Allow the app to start with degraded functionality
        logger.warning("Application starting with degraded functionality")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    rag_loaded = rag_pipeline is not None and hasattr(rag_pipeline, 'df')
    llm_loaded = llm_service is not None
    
    status = "healthy" if rag_loaded else "degraded"
    if not rag_loaded and not llm_loaded:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        rag_loaded=rag_loaded,
        llm_loaded=llm_loaded,
        timestamp=datetime.now().isoformat(),
        total_courses=len(rag_pipeline.df) if rag_loaded else 0
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Brown University Course Search API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Search for courses with natural language",
            "/evaluate": "GET - System performance metrics", 
            "/health": "GET - System health check"
        },
        "status": "operational" if rag_pipeline else "initializing"
    }

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_courses(request: QueryRequest, req: Request):
    """Main query endpoint for course search"""
    start_time = time.time()
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available. Please try again later.")
    
    try:
        # Retrieve context and courses using RAG pipeline
        context, raw_results = rag_pipeline.retrieve_context(
            query=request.q,
            k=request.k,
            department=request.department,
            source=request.source,
            instructor=request.instructor
        )
        
        # Generate answer using LLM service
        if llm_service:
            try:
                generated_answer = llm_service.generate_answer(
                    query=request.q,
                    context=context,
                    courses=raw_results,
                    max_tokens=300
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                # Fallback to template if LLM fails
                generated_answer = generate_template_answer(request.q, raw_results)
        else:
            # Use template-based answers if no LLM service
            generated_answer = generate_template_answer(request.q, raw_results)
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        # Format course results
        formatted_courses = []
        for course in raw_results:
            formatted_courses.append(CourseResult(
                course_code=course["course_code"],
                title=course["title"],
                department=course["department"],
                similarity=round(course["similarity_score"], 4),
                source=course["source"],
                description=course.get("description"),
                instructor=course.get("instructor"),
                meeting_times=course.get("meeting_times"),
                prerequisites=course.get("prerequisites")
            ))
        
        # Log query
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": request.q,
            "department": request.department,
            "source": request.source,
            "instructor": request.instructor,
            "retrieval_count": len(formatted_courses),
            "latency_ms": latency_ms,
            "client_ip": req.client.host if req.client else "unknown"
        }
        
        query_history.append(log_entry)
        if len(query_history) > MAX_HISTORY_SIZE:
            query_history.pop(0)
        
        logger.info(f"Query: '{request.q}' | Dept: {request.department} | "
                   f"Results: {len(formatted_courses)} | Latency: {latency_ms}ms")
        
        return QueryResponse(
            query=request.q,
            answer=generated_answer,
            courses=formatted_courses,
            retrieval_count=len(formatted_courses),
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query '{request.q}': {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Evaluation endpoint
@app.get("/evaluate", response_model=EvaluationMetrics)
async def evaluate():
    """System performance metrics endpoint"""
    if not query_history:
        return EvaluationMetrics(
            total_queries=0,
            average_latency_ms=0.0,
            total_retrievals=0,
            average_retrievals_per_query=0.0,
            uptime_seconds=time.time() - start_time if start_time else 0.0
        )
    
    total_queries = len(query_history)
    total_retrievals = sum(entry["retrieval_count"] for entry in query_history)
    average_latency = sum(entry["latency_ms"] for entry in query_history) / total_queries
    average_retrievals = total_retrievals / total_queries
    
    logger.info(f"Evaluation metrics - Queries: {total_queries}, "
               f"Avg latency: {average_latency:.2f}ms, "
               f"Avg retrievals: {average_retrievals:.1f}")
    
    return EvaluationMetrics(
        total_queries=total_queries,
        average_latency_ms=round(average_latency, 2),
        total_retrievals=total_retrievals,
        average_retrievals_per_query=round(average_retrievals, 2),
        uptime_seconds=round(time.time() - start_time, 2) if start_time else 0.0
    )

# Recent queries endpoint
@app.get("/queries/recent")
async def get_recent_queries(limit: int = 10):
    """Get recent query history"""
    recent = query_history[-limit:] if query_history else []
    return {
        "recent_queries": recent,
        "total_queries": len(query_history)
    }

# Template-based answer generation (fallback)
def generate_template_answer(query: str, courses: List[Dict]) -> str:
    """Fallback template-based answer generation"""
    if not courses:
        return "I couldn't find any courses matching your query. Please try different search terms or filters."
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['who teaches', 'instructor', 'professor', 'taught by']):
        return instructor_template(query, courses)
    elif any(word in query_lower for word in ['when', 'time', 'schedule', 'meet', 'class meet']):
        return schedule_template(query, courses)
    elif any(word in query_lower for word in ['recommend', 'suggest', 'find', 'interested', 'which', 'good']):
        return recommendation_template(query, courses)
    elif any(word in query_lower for word in ['similar', 'like', 'comparable']):
        return similarity_template(query, courses)
    else:
        return general_template(query, courses)

def instructor_template(query, courses):
    answers = []
    for course in courses[:3]:
        code = course['course_code']
        title = course['title']
        instructor = course.get('instructor', 'Instructor information not available')
        times = course.get('meeting_times', 'Meeting times not specified')
        answers.append(f"**{code}: {title}**\n- Instructor: {instructor}\n- Meeting Times: {times}")
    return "Based on your query about instructors:\n\n" + "\n\n".join(answers)

def schedule_template(query, courses):
    answers = [f"I found {len(courses)} course(s) matching your schedule criteria:\n"]
    for course in courses[:5]:
        line = f"- **{course['course_code']}: {course['title']}**"
        if course.get('meeting_times'):
            line += f"\n  Time: {course['meeting_times']}"
        if course.get('instructor'):
            line += f"\n  Instructor: {course['instructor']}"
        answers.append(line)
    return "\n\n".join(answers)

def recommendation_template(query, courses):
    answers = [f"Here are {min(len(courses), 5)} recommended courses:\n"]
    for i, c in enumerate(courses[:5], 1):
        desc = c.get('description', '')
        desc_short = desc[:120] + "..." if len(desc) > 120 else desc
        answers.append(f"{i}. **{c['course_code']}: {c['title']}** ({c['department']})\n   {desc_short}")
    return "\n\n".join(answers)

def similarity_template(query, courses):
    answers = [f"Here are courses similar to your query:\n"]
    for i, c in enumerate(courses[:5], 1):
        sim = c.get('similarity_score', 0)
        answers.append(f"{i}. **{c['course_code']}: {c['title']}** ({c['department']})\n   Similarity: {sim:.1%}")
    return "\n\n".join(answers)

def general_template(query, courses):
    answers = [f"I found {len(courses)} relevant course(s):\n"]
    for i, c in enumerate(courses[:5], 1):
        answers.append(f"{i}. **{c['course_code']}: {c['title']}** ({c['department']}, from {c['source']})")
    return "\n\n".join(answers)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )