import io
import os
import base64
import json
import asyncio
import logging
import fcntl
from io import BufferedReader  
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.api import partition_via_api
from generation.Google import generate_context_for_chunk
# from unstructured.partition.image import partition_image

class Preprocessor:
    def __init__(self, task, output_path, advanced_ocr = False, captioning = False):
        self.task = task
        self.api_key = os.getenv("UNSTRUCTURED_API_KEY")
        self.output_path = output_path
        self.captioning = captioning
        self.advanced_ocr = advanced_ocr
        
        logging.basicConfig(
            filename=self.output_path,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    async def ocr(self, image, filename): 
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        file_obj = BufferedReader(img_bytes)

        if self.advanced_ocr:

            elements = await asyncio.to_thread(
                partition_via_api,
                file=file_obj,
                api_key=self.api_key,
                metadata_filename=filename,
                strategy="hi_res",
                pdf_infer_table_structure=True,
                api_url="https://api.unstructuredapp.io/general/v0/general",
                retries_initial_interval=10,
                retries_max_interval=300,
                retries_max_elapsed_time=120,
            )

        # else:
        #     elements = await asyncio.to_thread(
        #         partition_image,
        #         file=file_obj,
        #         strategy="ocr_only",
        #         infer_table_structure=False
        #     )

        return elements

    async def __call__(self, image, filename):

        if not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                saved_chunks = {}
                json.dump(saved_chunks, f, indent=4)

        with open(self.output_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            saved_chunks = json.load(f)
            
        if filename in saved_chunks:
            print(f"File {filename} exists in saved chunks. Returning chunks...", flush=True)
            return [chunk["text"] for chunk in saved_chunks[filename]]

        print(f"File {filename} does not exist in saved chunks. Starting Processing ...", flush=True)

        elements = await self.ocr(image, filename)

        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=400,
            max_characters=int(1e6),
        )

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        base64_image = base64.b64encode(img_bytes.read()).decode('utf-8')

        for chunk in chunks:
            if self.captioning:
                chunk.text = "CONTEXT: " + await generate_context_for_chunk(base64_image, chunk.text, "gemini-2.0-flash") + "\n" + chunk.text

        with open(self.output_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            saved_chunks = json.load(f)

            # Add our new chunks
            saved_chunks[filename] = [chunk.to_dict() for chunk in chunks]
            
            # Write back to file
            f.seek(0)
            f.truncate()
            json.dump(saved_chunks, f, indent=4)


        chunks = [chunk.text for chunk in chunks]
        return chunks
    
