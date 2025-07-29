python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export TASK=finreport
export DATASET=ibm-research/REAL-MM-RAG_FinReport

python -m main.preprocessing.run
