python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# TAGS
export PIPELINE_NAME="TEXT-SINGLE"
export TEXT_SINGLE_EMBEDDER="linq"

# export TASK=finreport
# export DATASET=ibm-research/REAL-MM-RAG_FinReport

# export TASK=finslides
# export DATASET=ibm-research/REAL-MM-RAG_FinSlides

# export TASK=finqa
# export DATASET=emrekuruu/FinQA

# export TASK=convfinqa
# export DATASET=emrekuruu/ConvFinQA

# export TASK=vqaonbd
# export DATASET=emrekuruu/VQAonBD

# export TASK=tatdqa
# export DATASET=vidore/tatdqa_test

# export TASK=tatdqa
# export DATASET=vidore/tatdqa_train

python -m main.pipelines.retrieval