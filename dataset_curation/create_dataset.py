import pandas as pd
import numpy as np

def create_best_pipeline_labels():
    """
    Read performance data and create labels for best performing pipeline per query.
    
    Labels:
    0 - MULTIMODAL-MULTI
    1 - MULTIMODAL-SINGLE  
    2 - TEXT-MULTI
    3 - TEXT-SINGLE
    
    Best pipeline is determined by highest performance, with lowest latency as tiebreaker.
    """
    
    # Load the performance data
    print("Loading performance data...")
    df = pd.read_excel("analysis/performance_data.xlsx", index_col=0)
    
    print(f"Loaded data with {len(df)} queries and {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    
    # Define pipeline mapping
    pipeline_mapping = {
        'MULTIMODAL-MULTI': 0,
        'MULTIMODAL-SINGLE': 1, 
        'TEXT-MULTI': 2,
        'TEXT-SINGLE': 3
    }
    
    # Extract performance and latency columns
    performance_cols = [col for col in df.columns if col.endswith('_performance')]
    latency_cols = [col for col in df.columns if col.endswith('_latency')]
    
    print(f"Found {len(performance_cols)} performance columns and {len(latency_cols)} latency columns")
    
    # Create results dataframe
    results = []
    
    for query_id in df.index:
        query_data = df.loc[query_id]
        
        # Get performance scores for each pipeline
        pipeline_scores = {}
        pipeline_latencies = {}
        
        for perf_col in performance_cols:
            pipeline_name = perf_col.replace('_performance', '')
            latency_col = pipeline_name + '_latency'
            
            if pipeline_name in pipeline_mapping:
                performance = query_data[perf_col]
                latency = query_data[latency_col] if latency_col in query_data else np.inf
                
                # Only consider pipelines with valid performance data
                if not pd.isna(performance):
                    pipeline_scores[pipeline_name] = performance
                    pipeline_latencies[pipeline_name] = latency if not pd.isna(latency) else np.inf
        
        # Find best pipeline
        if pipeline_scores:
            # Find maximum performance
            max_performance = max(pipeline_scores.values())
            
            # Get all pipelines with max performance
            best_pipelines = [name for name, score in pipeline_scores.items() 
                            if score == max_performance]
            
            # If tie, choose the one with lowest latency
            if len(best_pipelines) > 1:
                best_pipeline = min(best_pipelines, 
                                  key=lambda x: pipeline_latencies[x])
            else:
                best_pipeline = best_pipelines[0]
            
            label = pipeline_mapping[best_pipeline]
            
            results.append({
                'query': query_id,
                'label': label,
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_excel("analysis/dataset.xlsx", index=False)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total queries processed: {len(results_df)}")
    print(f"Queries with valid data: {len(results_df[results_df['label'] >= 0])}")
    print(f"Queries with missing data: {len(results_df[results_df['label'] == -1])}")
    
    print("\nLabel distribution:")
    label_counts = results_df[results_df['label'] >= 0]['label'].value_counts().sort_index()

    for label, count in label_counts.items():
        pipeline_name = [k for k, v in pipeline_mapping.items() if v == label][0]
        print(f"  {label} ({pipeline_name}): {count} queries")
    
    
    return results_df

if __name__ == "__main__":
    results = create_best_pipeline_labels()
