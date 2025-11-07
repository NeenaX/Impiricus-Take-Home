# Brown AI Powered Course Searcher

## Setup Prerequisites

- Docker
- Docker Compose
- Git

## 🚀 Quick Start

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

**Initial startup will take 5-15 minutes**

On first run, the backend automatically downloads two AI models:

- **TinyLlama** (1.1B parameter language model) - ~2GB
- **BGE Embeddings** (for RAG functionality) - ~400MB

### Stopping the Application

```bash
# Stop containers
docker-compose down
```

## 🔌 API Endpoints

### Main Endpoints

- **`GET /health`** - Health check and system status
- **`POST /query`** - Submit natural language course queries
- **`GET /evaluate`** - System performance metrics
- **`GET /queries/recent`** - View recent query history
