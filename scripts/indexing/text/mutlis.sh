python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# TAGS
export PIPELINE_NAME="TEXT-SINGLE"
export TEXT_SINGLE_EMBEDDER="linq"

export TASK=finreport
export DATASET=ibm-research/REAL-MM-RAG_FinReport

python -m main.pipelines.indexing
