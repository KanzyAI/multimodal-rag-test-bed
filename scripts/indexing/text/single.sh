python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# TAGS
export PIPELINE_NAME="text-single"
export TEXT_SINGLE_DATABASE_NAME="pinecone"
export TEXT_SINGLE_EMBEDDER="voyage_text"
export DATASET=ibm-research/REAL-MM-RAG_TechReport
export TASK=techreport
export API_KEY=$PINECONE_API_KEY

python -m main.pipelines.indexing
