python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export TASK=tatdqa
export DATASET=vidore/tatdqa_train

python -m main.preprocessing.run
