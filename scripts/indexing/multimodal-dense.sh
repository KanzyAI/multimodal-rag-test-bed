python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# EMBEDDER CONFIGS
export MULTIMODAL_DENSE_EMBEDDER="voyage"

# TAGS
export EMBEDDER_NAME=$MULTIMODAL_DENSE_EMBEDDER
export PIPELINE_NAME="multimodal-dense"
export DATABASE_NAME="pinecone"
export TASK=TechReport
export API_KEY=$PINECONE_API_KEY

python -m main.pipelines.visual.single.indexing
