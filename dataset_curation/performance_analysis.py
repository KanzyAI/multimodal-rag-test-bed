import os 
import json
import pandas as pd

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_queries(task, pipeline, query_column):
    with open(f"results/{task}/{pipeline}/per_query/{query_column}.json_ndcg_per_query.json", "r") as f:
        data = json.load(f)
    return data

def load_latencies(task, pipeline, query_column):
    with open(f"results/{task}/{pipeline}/{query_column}.json", "r") as f:
        data = json.load(f)
    data = {k: v["retrieval_latency"] for k, v in data.items()}
    return data

tasks = ["finreport","finslides","finqa", "convfinqa", "vqaonbd","tatdqa"]
pipelines = ["MULTIMODAL-MULTI", "MULTIMODAL-SINGLE"]

def main():
    columns = []

    for pipeline in pipelines:
        columns.append(pipeline + "_performance")
        columns.append(pipeline + "_latency")
    
    performance_df = pd.DataFrame(columns=columns)

    for task in tasks:
        for pipeline in pipelines:
            for query_column in ["query", "rephrase_level_1", "rephrase_level_2", "rephrase_level_3"]:
                try:
                    print(f"Loading data for {task} {pipeline} {query_column}")
                    performance_data = load_queries(task, pipeline, query_column)
                    latency_data = load_latencies(task, pipeline, query_column)

                    for query, performance in performance_data.items():
                        performance_df.loc[query, pipeline + "_performance"] = performance
                        performance_df.loc[query, pipeline + "_latency"] = latency_data[query]

                except Exception as e:
                    print(f"Error loading data for {task} {pipeline}: {e}")

    performance_df.to_excel("dataset_curation/performance_data.xlsx")

if __name__ == "__main__":
    main()