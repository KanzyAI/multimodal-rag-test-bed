python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export TASK=vidore_test
export DATASET=vidore/tatdqa_test

python -m main.preprocessing.run
