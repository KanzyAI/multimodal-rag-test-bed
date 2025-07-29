python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# TAGS
export PIPELINE_NAME="multimodal-single"
export MULTIMODAL_SINGLE_DATABASE_NAME="pinecone"
export MULTIMODAL_SINGLE_EMBEDDER="voyage"
export DATASET=ibm-research/REAL-MM-RAG_TechReport
export TASK=techreport
export API_KEY=$PINECONE_API_KEY

python -m main.pipelines.indexing
