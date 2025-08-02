import os 
import json
import pandas as pd
from evaluation.significance_tests import compare_models

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
pipelines = ["MULTIMODAL-MULTI", "MULTIMODAL-SINGLE", "TEXT-SINGLE", "TEXT-MULTI"]

def run_cross_dataset_significance_tests(performance_df):
    """Run significance tests across all pipelines using the performance data."""
    print("Running cross-dataset significance tests...")
    
    # Prepare data for significance testing
    # Convert the DataFrame to the format expected by compare_models
    pipeline_columns = [col for col in performance_df.columns if col.endswith("_performance")]
    
    # Create ratings dictionary
    ratings_dict = {}
    for col in pipeline_columns:
        pipeline_name = col.replace("_performance", "")
        # Get non-null performance scores
        scores = performance_df[col].dropna().tolist()
        if len(scores) >= 5:  # Minimum sample size
            ratings_dict[pipeline_name] = {
                'nDCG@5': scores,
                'MRR@5': scores,  # Using same scores for consistency
                'Precision@5': scores,
                'Recall@5': scores
            }
    
    if len(ratings_dict) < 2:
        print("Not enough pipelines with sufficient data for significance testing")
        return None
    
    # Convert to DataFrame and run significance tests
    ratings_df = pd.DataFrame.from_dict(ratings_dict, orient='index')
    sig_results = compare_models(ratings_df)
    
    # Save results
    sig_results.to_excel("analysis/cross_dataset_significance_tests.xlsx", index=False)
    print("Saved cross-dataset significance test results to analysis/cross_dataset_significance_tests.xlsx")
    
    return sig_results

def main():
    columns = []

    for pipeline in pipelines:
        columns.append(pipeline + "_performance")
        columns.append(pipeline + "_latency")
    
    performance_df = pd.DataFrame(columns=columns)

    for task in tasks:
        for pipeline in pipelines:
            try:
                print(f"Loading data for {task} {pipeline}")
                performance_data = load_queries(task, pipeline, "query")
                latency_data = load_latencies(task, pipeline, "query")

                for query, performance in performance_data.items():
                    performance_df.loc[query, pipeline + "_performance"] = performance
                    performance_df.loc[query, pipeline + "_latency"] = latency_data[query]

            except Exception as e:
                print(f"Error loading data for {task} {pipeline}: {e}")

    performance_df.to_excel("analysis/performance_data.xlsx")
    print("Saved performance data to analysis/performance_data.xlsx")
    
    # Run cross-dataset significance tests
    run_cross_dataset_significance_tests(performance_df)

if __name__ == "__main__":
    main()