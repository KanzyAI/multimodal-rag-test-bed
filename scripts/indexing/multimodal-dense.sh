python -c "from dotenv import load_dotenv; load_dotenv()"

# EMBEDDER CONFIGS
export MULTIMODAL_DENSE_EMBEDDER="voyage"
export MULTIMODAL_MULTI_EMBEDDER="colqwen"
export TEXT_DENSE_EMBEDDER="voyage"
export TEXT_MULTI_EMBEDDER="voyage"

# TAGS
export EMBEDDER_NAME=$MULTIMODAL_DENSE_EMBEDDER
export PIPELINE_NAME="multimodal-dense"
export DATABASE_NAME="qdrant"
export TASK=TechReport

python -m main.pipelines.visual.single.indexing
