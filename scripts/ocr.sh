python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export TASK=vqaonbd
export DATASET=emrekuruu/VQAonBD

python -m main.preprocessing.run
