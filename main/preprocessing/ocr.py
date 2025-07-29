import io
import os
import json
import asyncio
import logging
import fcntl
from mistralai import Mistral
from mistralai.models import File
from main.preprocessing.captioning import captioning, parse_image_references
from main.preprocessing.chunking import chunk_financial_markdown

class OCR_Engine:
    def __init__(self, output_path):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        self.client = Mistral(api_key=self.api_key) 
        self.output_path = output_path
        
        # Create chunks output path
        self.chunks_output_path = output_path.replace('_full_text.json', '_chunks.json')
        
        logging.basicConfig(
            filename=self.output_path,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    async def ocr(self, image, filename): 
        """Process image using Mistral OCR API - assumes only 1 page per image"""
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            
            uploaded_file = await asyncio.to_thread(
                self.client.files.upload,
                file=File(
                    file_name=f"{filename}.png",
                    content=img_bytes.read(),
                ),
                purpose="ocr"
            )
            
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            ocr_response = await asyncio.to_thread(
                self.client.ocr.process,
                model="mistral-ocr-latest",
                document={
                    "type": "document_url", 
                    "document_url": signed_url.url,
                },
                include_image_base64=False
            )
            
            page = ocr_response.pages[0]
            if page.markdown:
                return page.markdown.strip()
            
        except Exception as e:
            raise e
        
    async def __call__(self, image, filename, apply_captioning=True, apply_chunking=False):
        """Complete preprocessing pipeline: OCR → Captioning → Chunking → Save"""

        # Initialize files if they don't exist
        for file_path in [self.output_path, self.chunks_output_path]:
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump({}, f, indent=4)

        # Check if already processed
        with open(self.output_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            saved_texts = json.load(f)
            
        if filename in saved_texts:
            return saved_texts[filename]["text"]


        # Step 1: OCR
        markdown_text = await self.ocr(image, filename)
        
        # Step 2: Captioning (if needed)
        if apply_captioning:
            image_refs = parse_image_references(markdown_text)
        else:
            image_refs = []
        
        if image_refs and apply_captioning:
            try:
                captioned_text = await captioning(
                    document_image=image,
                    document_text=markdown_text,
                    model_type="gemini-2.5-flash"
                )
                final_text = captioned_text
            except Exception as e:
                final_text = markdown_text
        else:
            final_text = markdown_text
        
        # Step 3: Chunking
        if apply_chunking:
            try:
                chunk_documents = chunk_financial_markdown(final_text)
                # Extract just the text content (no metadata)
                chunks = [chunk.page_content for chunk in chunk_documents]
            except Exception as e:
                chunks = [final_text]  # Fallback to full text as single chunk
        else:
            chunks = [final_text]
        
        # Step 4: Save full text
        with open(self.output_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            saved_texts = json.load(f)
            saved_texts[filename] = {"text": final_text}
            f.seek(0)
            f.truncate()
            json.dump(saved_texts, f, indent=4)

        # Step 5: Save chunks  
        with open(self.chunks_output_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            saved_chunks = json.load(f)
            saved_chunks[filename] = {"chunks": chunks}
            f.seek(0)
            f.truncate()
            json.dump(saved_chunks, f, indent=4)

        return final_text
    
