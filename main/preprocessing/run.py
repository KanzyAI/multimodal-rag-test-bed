import os
import asyncio
import json
from datasets import load_dataset
from typing import Any
from tqdm.asyncio import tqdm
from main.preprocessing.ocr import OCR_Engine
from main.pipelines import TASK

class Preprocessing:    
    """Handles preprocessing of documents before indexing"""
    
    def __init__(self, **kwargs):
        self.semaphore = asyncio.Semaphore(20)
        self.ocr_engine = OCR_Engine(f"main/preprocessing/texts/{TASK}_full_text.json")
    
    async def preprocess_document(self, filename: str, image: Any) -> str:
        """Preprocess a single document"""
        async with self.semaphore:
            processed_text = await self.ocr_engine(image, filename, apply_captioning=True)
            return processed_text
    
    async def process_all_files(self, processed_files: set = None):
        """Process all files from the dataset that haven't been preprocessed yet"""
        
        dataset = load_dataset(os.getenv("DATASET"), token=os.getenv("HF_TOKEN"), split="test")
        all_keys = set(dataset["image_filename"])
        
        if processed_files is None:
            processed_files = set()
        
        keys_to_process = all_keys - processed_files
        
        total_docs = len(keys_to_process)
        
        if total_docs == 0:
            print("No new files to preprocess")
            return
        
        print(f"Preprocessing {total_docs} documents...")
        
        with tqdm(total=total_docs, desc="Preprocessing documents", unit="docs") as pbar:
            tasks = []
            
            for row in dataset:
                if row["image_filename"] in keys_to_process:
                    keys_to_process.remove(row["image_filename"])
                    filename = row["image_filename"]
                    image = row["image"]
                    
                    async def update_progress(filename=filename, image=image):
                        result = await self.preprocess_document(filename, image)
                        pbar.update(1)
                        pbar.set_postfix({"file": filename})
                        return result
                    
                    tasks.append(update_progress())
            
            if tasks:
                await asyncio.gather(*tasks)
        
        print(f"Preprocessing completed for {total_docs} documents")
    
    def get_processed_files(self) -> set:
        preprocessed_file = f"main/preprocessing/texts/{TASK}_full_text.json"
        
        if not os.path.exists(preprocessed_file):
            return set()
        
        with open(preprocessed_file, 'r') as f:
            saved_texts = json.load(f)
            processed = {k for k, v in saved_texts.items() if v != {}}
        return processed
    
    async def __call__(self):
        processed_files = self.get_processed_files()
        await self.process_all_files(processed_files)

if __name__ == "__main__":
    preprocessing = Preprocessing()
    asyncio.run(preprocessing())    
