# Multimodal Retrieval Pipeline Evaluation Framework

Automated testbed for evaluating retrieval pipeline accuracy and performance across different embedders, vector databases, and modalities.

## Overview

This system evaluates:
- **Embedders**: Late Interaction and Dense embedding models for text and multimodal content and and sparse pipelines for benchmarking
- **Vector Databases**: Vector database factory to support plug-and-play evalaution of different vector store providers
- **Modalities**: Text-based, multimodal based approaches

## Architecture

```
main/
├── embedders/           # Embedding model implementations with factory pattern
│   ├── base_embedder.py # Abstract base class and EmbedderFactory
│   ├── text/           # Text embedders (LinQ)
│   └── multimodal/     # Vision-language embedders (ColQwen, BiQwen, Voyage)
├── pipelines/          # Retrieval pipeline implementations
│   ├── base/           # Base indexing and retrieval classes
│   ├── text/           # Text-based retrieval (single/multi vector)
│   ├── visual/         # Visual retrieval (single/multi vector)
│   └── baselines/      # Sparse retrieval baselines (BM25, SPLADE)
├── vector_databases/   # Vector database abstraction layer
└── generation/         # LLM integration modules (currently not used)

evaluation/             # Evaluation framework and statistical testing
utils/                  # Latency extraction and performance analysis tools
results/                # Generated evaluation results and reports
scripts/                # Shell scripts for ease of use 
```

## Supported Models

### Text Embedders
- **LinQ**: `Linq-AI-Research/Linq-Embed-Mistral`
- **ColBERT**: Multi-vector text embedder (available for implementation)

### Multimodal Embedders
- **ColQwen2.5**: `Metric-AI/ColQwen2.5-7b-multilingual-v1.0`
- **BiQwen2.5**: Bidirectional multimodal embedder
- **Voyage Multimodal**: `voyage-multimodal-3`

### Baselines
- **BM25**: Term-frequency sparse retrieval 
- **SPLADE**: Learned sparse retrieval using YASEM library

## Setup

### Install Dependencies
```bash
poetry install
```

### Environment Configuration

Create a `.env` file:

```bash
# Hugging Face
HF_TOKEN=your_hf_token

# Vector Database Configuration
TEXT_SINGLE_QDRANT_URL=your_qdrant_url
TEXT_SINGLE_QDRANT_API_KEY=your_qdrant_api_key
TEXT_MULTI_QDRANT_URL=your_qdrant_url
TEXT_MULTI_QDRANT_API_KEY=your_qdrant_api_key
...

# Embedding Model APIs
VOYAGEAI_API_KEY=your_voyage_api_key
UNSTRUCTURED_API_KEY=your_unstructured_api_key

# Pipeline Configuration
TASK=TechReport  # or FinReport
EMBEDDER_NAME=your_embedder_name
PIPELINE_NAME=your_pipeline_name

# Embedder Selection (Factory Pattern)
TEXT_EMBEDDER=linq
MULTIMODAL_EMBEDDER=colqwen
MULTIMODAL_DENSE_EMBEDDER=voyage

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
```

## Usage

### 1. Indexing Documents

```bash
# Text single vector
export TASK=TechReport
export TEXT_EMBEDDER=linq
python -m main.pipelines.text.single.indexing

# Visual single vector  
export MULTIMODAL_DENSE_EMBEDDER=voyage
python -m main.pipelines.visual.single.indexing

# Multi vector approaches
python -m main.pipelines.text.multi.indexing
python -m main.pipelines.visual.multi.indexing
```

### 2. Running Retrieval

```bash
# Dense methods
python -m main.pipelines.text.single.retrieval
python -m main.pipelines.visual.single.retrieval

# Sparse baselines
python -m main.pipelines.baselines.bm25.retrieval
python -m main.pipelines.baselines.splade.retrieval
```

### 3. Evaluation

```bash
python -m evaluation.retrieval
```

Generates performance metrics, significance tests, and Excel reports.

## Adding New Components

### Creating a New Embedder

```python
# main/embedders/text/my_embedder.py
from ..base_embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory

class MyEmbedder(BaseEmbedder):
    def _initialize_model(self):
        # Initialize your model here
        self.model = YourModel(self.model_name)
        
    async def embed_query(self, query):
        # Embed a query
        return self.model.encode_query(query)
    
    async def embed_document(self, document):
        # Embed a document  
        return self.model.encode_document(document)

# Register with factory
EmbedderFactory.register_embedder("my_embedder", MyEmbedder)
```

Then add to `main/embedders/text/__init__.py`:
```python
from .my_embedder import MyEmbedder
__all__ = ['LinqEmbedder', 'MyEmbedder']
```

### Creating a New Vector Database

```python
# main/vector_databases/my_database.py
from .base_database import BaseVectorDatabase, DatabaseConfig, SearchResults, VectorDatabaseFactory

class MyDatabase(BaseVectorDatabase):
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        # Initialize your database client
        
    async def initialize_collection(self):
        # Create collection if it doesn't exist
        pass
        
    async def index_document(self, embedding, metadata):
        # Store document embedding with metadata
        pass
        
    async def search(self, query_embedding, limit=5):
        # Search and return SearchResults
        pass
        
    async def get_indexed_files(self):
        # Return set of already indexed filenames
        pass

# Register with factory
VectorDatabaseFactory.register_database("my_db", MyDatabase)
```

### Using Your New Components

Update your `.env`:
```bash
TEXT_EMBEDDER=my_embedder
DATABASE_TYPE=my_db
```

The factory pattern will automatically use your registered components.

## Evaluation Metrics

- **nDCG@5**: Normalized Discounted Cumulative Gain
- **MRR@5**: Mean Reciprocal Rank  
- **Precision@5** and **Recall@5**

Statistical tests include Friedman test, paired t-tests, and effect size analysis.

## Performance Analysis

Extract latency metrics from LangSmith:
```python
from utils.extract_latency_results import LatencyExtractor

extractor = LatencyExtractor()
extractor.extract_leaf_nodes_from_run(root_run_id="your_run_id")
```

## Notes

- The system uses factory patterns for easy extension
- Some pipeline scripts have legacy imports - use environment variables with the factory pattern
- Results are saved to `results/retrieval/{dataset}/`
- Comprehensive logging and tracing available via LangSmith

## License

Apache 2.0 License

## Author

Emre Kuru <emre.kuru@ozu.edu.tr>