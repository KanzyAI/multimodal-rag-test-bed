python -c "from dotenv import load_dotenv; load_dotenv('.env')"

# TAGS
export PIPELINE_NAME="TEXT-MULTI"
export TEXT_MULTI_EMBEDDER="colbert"

export TASK=finreport
export DATASET=ibm-research/REAL-MM-RAG_FinReport

python -m main.pipelines.retrieval

export TASK=finslides
export DATASET=ibm-research/REAL-MM-RAG_FinSlides

python -m main.pipelines.retrieval

export TASK=finqa
export DATASET=emrekuruu/FinQA

python -m main.pipelines.retrieval

export TASK=convfinqa
export DATASET=emrekuruu/ConvFinQA

python -m main.pipelines.retrieval

export TASK=vqaonbd
export DATASET=emrekuruu/VQAonBD

python -m main.pipelines.retrieval

export TASK=tatdqa
export DATASET=emrekuruu/TATDQA

python -m main.pipelines.retrieval

