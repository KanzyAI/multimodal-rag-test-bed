import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://5faae072-f569-4c46-881c-65a142ca7eae.eu-central-1-0.aws.cloud.qdrant.io:6333/collections/"

collections = [
    # Multimodal collections
    "multimodal-multi-finslides",
    "multimodal-multi-finreport", 
    "multimodal-multi-finqa",
    "multimodal-multi-convfinqa",
    "multimodal-multi-vqaonbd",
    "multimodal-multi-tatdqa",
    
    "text-multi-finslides",
    "text-multi-finreport",
    "text-multi-finqa",
    "text-multi-convfinqa",
    "text-multi-vqaonbd",
    "text-multi-tatdqa",

    # Single collections
    "multimodal-single-finslides",
    "multimodal-single-finreport",
    "multimodal-single-finqa", 
    "multimodal-single-convfinqa",
    "multimodal-single-vqaonbd",
    "multimodal-single-tatdqa",

    "text-single-finslides",
    "text-single-finreport",
    "text-single-finqa",
    "text-single-convfinqa",
    "text-single-vqaonbd",
    "text-single-tatdqa",
]

headers = {
    "api-key": os.environ.get("QDRANT_API_KEY")
}

results_multimodal_multi = []
results_text_multi = []
results_multimodal_single = []
results_text_single = []

for name in collections:
    try:
        response = requests.get(f"{BASE_URL}{name}", headers=headers)
        data = response.json().get("result", {})
        record = {
            "Collection": name,
            "Status": data.get("status", ""),
            "Points": data.get("points_count", ""),
            "Indexed": data.get("indexed_vectors_count", ""),
            "Segments": data.get("segments_count", "")
        }
        if name.startswith("multimodal-multi-"):
            results_multimodal_multi.append(record)
        elif name.startswith("text-multi-"):
            results_text_multi.append(record)
        elif name.startswith("multimodal-single-"):
            results_multimodal_single.append(record)
        elif name.startswith("text-single-"):
            results_text_single.append(record)
    except Exception as e:
        error_record = {
            "Collection": name,
            "Status": f"ERROR: {e}",
            "Points": None,
            "Indexed": None,
            "Segments": None
        }
        if name.startswith("multimodal-multi-"):
            results_multimodal_multi.append(error_record)
        elif name.startswith("text-multi-"):
            results_text_multi.append(error_record)
        elif name.startswith("multimodal-single-"):
            results_multimodal_single.append(error_record)
        elif name.startswith("text-single-"):
            results_text_single.append(error_record)

# Save all to separate Excel files
pd.DataFrame(results_multimodal_multi).to_excel("qdrant_multimodal_multi_indexing_progress.xlsx", index=False)
pd.DataFrame(results_text_multi).to_excel("qdrant_text_multi_indexing_progress.xlsx", index=False)
pd.DataFrame(results_multimodal_single).to_excel("qdrant_multimodal_single_indexing_progress.xlsx", index=False)
pd.DataFrame(results_text_single).to_excel("qdrant_text_single_indexing_progress.xlsx", index=False)
