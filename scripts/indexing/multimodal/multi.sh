python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# TAGS
export PIPELINE_NAME="MULTIMODAL-MULTI"
export MULTIMODAL_MULTI_EMBEDDER="colqwen"

export TASK=finreport
export DATASET=ibm-research/REAL-MM-RAG_FinReport

python -m main.pipelines.indexing
