import json
import os 
import pandas as pd

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_queries(task, pipeline):
    with open(f"results/{task}/{pipeline}/per_query/query.json_ndcg_per_query.json", "r") as f:
        data = json.load(f)
    return data


tasks = ["finreport", "finqa", "convfinqa", "vqaonbd","finslides","tatdqa"]

pipelines = ["MULTIMODAL-MULTI", "MULTIMODAL-SINGLE", "TEXT-SINGLE", "TEXT-MULTI"]

def main():

    performance_df = pd.DataFrame(columns=pipelines)

    for task in tasks:

        for pipeline in pipelines:

            try:
                print(f"Loading data for {task} {pipeline}")
                performance_data = load_queries(task, pipeline)

                for query, performance in performance_data.items():

                        performance_df.loc[query, pipeline] = performance

            except Exception as e:
                print(f"Error loading data for {task} {pipeline}: {e}")

    performance_df.to_excel("analysis/performance_data.xlsx")

if __name__ == "__main__":
    main()