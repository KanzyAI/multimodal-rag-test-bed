import json
import pandas as pd
import os

pipelines = ["TEXT-MULTI", "MULTIMODAL-MULTI", "MULTIMODAL-SINGLE"]

for task in ["finqa", "finreport", "finslides", "vqaonbd", "convfinqa"]:

    df = pd.DataFrame(columns=["pipeline", "mean_runtime"], index=pipelines)

    for pipeline in pipelines:
        try:
            with open(f"results/{task}/{pipeline}/query.json", "r") as f:
                data = json.load(f)

            total_runtime = 0
            for k, v in data.items():
                total_runtime += v["retrieval_latency"] 

            mean_runtime = total_runtime / len(data)
            df.loc[pipeline] = [pipeline, mean_runtime]

        except Exception as e:
            print(f"Error processing {task}/{pipeline}: {str(e)}")
            df.loc[pipeline] = [pipeline, None]

    os.makedirs("analysis/latency", exist_ok=True)
    df.to_excel(f"analysis/latency/latency_analysis_{task}.xlsx", index=True)