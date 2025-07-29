python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export TASK=finslides
export DATASET=ibm-research/REAL-MM-RAG_FinSlides

python -m main.preprocessing.run
