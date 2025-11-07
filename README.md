# Brown AI Powered Course Searcher

## Setup Prerequisites

- Docker
- Docker Compose
- Git

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/NeenaX/Impiricus-Take-Home.git
cd Impiricus-Take-Home
```

### 2. Run the Application

```bash
docker-compose up --build
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## First Run Important Notes

Initial startup will take 5-15 minutes

On first run, the backend automatically downloads two AI models:

- **TinyLlama** (1.1B parameter language model)
- **BGE Embeddings** (for RAG functionality)

### Stopping the Application

```bash
# Stop containers
docker-compose down
```

## System Architecture

### 1. Frontend (React)

**Purpose**: Provides an interactive UI that allows user to use natural language to find Brown courses. Users input natural language queries and recieve AI generated answers with displayed relevant course with metadata.

#### Key Components

- **`QueryForm`** - Gets user query and optional department filter
- **`CourseList`** - Displays the course code, title, instructor, and description of course results

### 2. Backend (FastAPI)

**Purpose**: Connects the RAG pipeline with the LLM service. FastAPI recieves a query from the frontend which is then inputted into the `CourseRAGPipeline` to perform a hybrid search (semantic and keyword). An answer is then generated with an LLM or a template resoponse if the LLM is not available. The system finally sends back a response to the frontend that contains the query, the generated answer, and the ranked list of relavant courses.

#### Main Components

- **`CourseRAGPipeline`** - Handles vector search and retrieval.
- **`LLMService`** - Generates answers using a transformer model or fallback templates

#### API Endpoints

- **`GET /health`** - Health check and system status
- **`POST /query`** - Submit natural language course queries
- **`GET /evaluate`** - System performance metrics
- **`GET /queries/recent`** - View recent query history

### 3. RAG Pipeline

**Purpose**: Performs information retrieval over Brown Courses

#### Components

- **Embedding Model** - The `BAAI/bge-base-en-v1.5` model was used to encode course information into dense vectors
- **Vector Index** - FAISS index was used for cosine similarity search
- **TFIDF** - Used for keyword based search
- **Hybrid Search** - Combined TFIDF and the vector index or a more robust result

The course data was stored in a CSV with their metadata (title, description, department, instructor, etc.). Additionally, embeddings and metadata was also stored for quicker reload.

## Design Decision and Trade-offs

The LLM used for answer generation was `TinyLlama/TinyLlama-1.1B-Chat-v1.0` becuase it is ligthweight and didn't require any authentication. This means that with the limited resources of this project, the answers will generate faster and it makes deployment easier on other machines because there is no need to authenticate. The trade-off here is that the answers generated will be of lower quality than other bigger models.

The backend is capable of using template responses which is reliable when an LLM is not avaiable to generate an answer. The trade-off here is that the response will be very rigid and increases the maintenance cost if anything changes in the course schema.

The department filter is a textbox with no restriction which made it easier to implement in the limited time, but it doesn't check for whether that department exists or the submission of multiple departments. This filter option could be changed in the future to auto complete with the aviable departments and submit more than one department.

## How to add additional datasets

Additional datasets can be added to the `backend.data` folder. The `combine.py` file in the will `backend.scrapers` folder will need to be updated to acomodate the new dataset and create a new `courses.csv` file that could then be fed into the RAG pipeline. From there as long as the data is in the same directory and has the same schema, the RAG pipeline will need to rebuild and save the new embeddings before it'll run again.
