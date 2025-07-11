import time
from langsmith import Client
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class LatencyExtractor:
    def __init__(self):
        self.client = Client()
    
    def extract_metadata_from_run(self, run) -> Dict[str, Any]:
        """Extract metadata from a run's extra field"""
        metadata = {}
        if hasattr(run, 'extra') and run.extra:
            if 'metadata' in run.extra:
                metadata.update(run.extra['metadata'])
        return metadata
    
    def calculate_latency_ms(self, run) -> Optional[float]:
        """Calculate latency in milliseconds for a run"""
        if run.start_time and run.end_time:
            latency = (run.end_time - run.start_time).total_seconds() 
            return round(latency, 2)
        return None
    
    def is_leaf_node(self, run) -> bool:
        """Check if a run is a leaf node (has no children)"""
        max_retries = 10
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                child_runs = list(self.client.list_runs(parent_run_id=run.id, limit=1))
                return len(child_runs) == 0
                
            except Exception as e:
                if ("429" in str(e) or "rate limit" in str(e).lower()) and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit checking children for {run.id}, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    print(f"Error checking children for {run.id}: {e}")
                    return True  # Assume it's a leaf if we can't check
        
        return True
    
    def collect_all_leaf_nodes(self, run, parent_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Recursively find all leaf nodes"""
        results = []
        
        # Get current run metadata
        current_metadata = self.extract_metadata_from_run(run)
        
        # Merge with parent metadata
        combined_metadata = {}
        if parent_metadata:
            combined_metadata.update(parent_metadata)
        combined_metadata.update(current_metadata)
        
        # Check if this is a leaf node
        if self.is_leaf_node(run):
            latency = self.calculate_latency_ms(run)
            
            # Extract leaf node information
            leaf_result = {
                'run_name': run.name,
                'latency_ms': latency,
                'metadata': combined_metadata
            }
            results.append(leaf_result)
        else:
            # Continue recursively searching child runs
            max_retries = 10
            base_delay = 1.0
            
            for attempt in range(max_retries + 1):
                try:
                    child_runs = list(self.client.list_runs(parent_run_id=run.id))
                    for child_run in child_runs:
                        child_results = self.collect_all_leaf_nodes(child_run, combined_metadata)
                        results.extend(child_results)
                    break
                    
                except Exception as e:
                    if ("429" in str(e) or "rate limit" in str(e).lower()) and attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit for {run.id}, retrying in {delay}s - {e}")
                        time.sleep(delay)
                    else:
                        print(f"Error getting child runs for {run.id}: {e}")
                        break
        
        return results
    
    def extract_leaf_nodes_from_run(self, root_run_id: str) -> List[Dict[str, Any]]:
        """Extract all leaf nodes from a specific root run ID"""
        
        try:    
            root_run = self.client.read_run(root_run_id)
            print(f"Processing root run: {root_run.name} ({root_run.id})")
            
            results = self.collect_all_leaf_nodes(root_run)
            
            return results
            
        except Exception as e:
            print(f"Error extracting leaf nodes from run {root_run_id}: {e}")
            return []
    
    def has_filename_in_metadata(self, metadata: Dict[str, Any]) -> str:
        """Check if metadata contains a filename and return it"""
        if 'filename' in metadata:
            return metadata['filename']
        elif 'file_name' in metadata:
            return metadata['file_name']
        elif 'file' in metadata:
            return metadata['file']
        return None
    
    def filter_nodes_with_filename(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter nodes to only include those with filename in metadata"""
        filtered_nodes = []
        for node in results:
            filename = self.has_filename_in_metadata(node.get('metadata', {}))
            if filename:
                node_with_filename = node.copy()
                node_with_filename['filename'] = filename
                filtered_nodes.append(node_with_filename)
        
        print(f"Filtered to {len(filtered_nodes)} nodes with filename in metadata")
        return filtered_nodes
    
    def create_excel_matrix(self, filtered_nodes: List[Dict[str, Any]], root_run_id: str):
        """Create Excel file with filenames as rows and run names as columns"""
        if not filtered_nodes:
            print("No filtered nodes available for Excel creation")
            return
        
        # Get unique filenames and run names
        filenames = set()
        run_names = set()
        
        for node in filtered_nodes:
            filenames.add(node['filename'])
            run_names.add(node['run_name'])
        
        filenames = sorted(list(filenames))
        run_names = sorted(list(run_names))
        
        print(f"Creating Excel matrix: {len(filenames)} files × {len(run_names)} run types")
        
        # Create matrix data structure
        matrix_data = {}
        latency_data = {}
        count_data = {}
        
        # Initialize matrices
        for filename in filenames:
            matrix_data[filename] = {}
            latency_data[filename] = {}
            count_data[filename] = {}
            for run_name in run_names:
                matrix_data[filename][run_name] = []
                latency_data[filename][run_name] = None
                count_data[filename][run_name] = 0
        
        # Fill matrix with data
        for node in filtered_nodes:
            filename = node['filename']
            run_name = node['run_name']
            latency = node.get('latency_ms')
            
            if latency is not None:
                matrix_data[filename][run_name].append(latency)
        
        # Calculate average latencies and counts
        for filename in filenames:
            for run_name in run_names:
                latencies = matrix_data[filename][run_name]
                if latencies:
                    latency_data[filename][run_name] = round(sum(latencies) / len(latencies), 2)
                    count_data[filename][run_name] = len(latencies)
        
        # Create DataFrames
        latency_df = pd.DataFrame(latency_data).T
        latency_df.index.name = 'Filename'
        
        count_df = pd.DataFrame(count_data).T
        count_df.index.name = 'Filename'
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"latency_matrix_{root_run_id}_{timestamp}.xlsx"
        
        # Save to Excel with multiple sheets
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Average latency sheet
                latency_df.to_excel(writer, sheet_name='Average_Latency_ms')
                
                # Count sheet
                count_df.to_excel(writer, sheet_name='Run_Counts')
                
                # Raw data sheet for reference
                raw_df = pd.DataFrame(filtered_nodes)
                raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            print(f"Excel file created: {output_file}")
            print(f"Sheets: Average_Latency_ms, Run_Counts, Raw_Data")
            
            # Print summary
            total_data_points = sum(count_data[f][r] for f in filenames for r in run_names if count_data[f][r] > 0)
            print(f"Total data points: {total_data_points}")
            
            return output_file
            
        except Exception as e:
            print(f"Error creating Excel file: {e}")
            return None
    
def main(root_run_id: str = None):
    extractor = LatencyExtractor()
    
    results = extractor.extract_leaf_nodes_from_run(root_run_id)
    
    if not results:
        print(f"No leaf nodes found for run ID: {root_run_id}")
        return
    
    filtered_nodes = extractor.filter_nodes_with_filename(results)
    
    if filtered_nodes:
        extractor.create_excel_matrix(filtered_nodes, root_run_id)
    else:
        print("No nodes with filename found - skipping Excel creation")
        
if __name__ == "__main__":
    main(root_run_id="f719e5cb-03ac-4cde-a00c-cafa4632cb19") 