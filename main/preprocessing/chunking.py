import re
import os
import json
import fcntl
from typing import List
from langchain.schema import Document
from main.pipelines import TASK

class ChunkingEngine:
    """Handles chunking of already processed full text documents"""
    
    def __init__(self):
        self.input_path = f"main/preprocessing/texts/{TASK}_full_text.json"
        self.output_path = f"main/preprocessing/chunks/{TASK}_chunks.json"
    
    def get_processed_files(self) -> set:
        """Get files that have already been chunked"""
        if not os.path.exists(self.output_path):
            return set()
        
        with open(self.output_path, 'r') as f:
            saved_chunks = json.load(f)
            processed = {k for k, v in saved_chunks.items() if v.get("chunks", [])}
        return processed
    
    def get_available_files(self) -> set:
        """Get files that have full text and are available for chunking"""
        if not os.path.exists(self.input_path):
            return set()
            
        with open(self.input_path, 'r') as f:
            saved_texts = json.load(f)
            available = {k for k, v in saved_texts.items() if v.get("text", "")}
        return available
    
    def chunk_document(self, filename: str) -> List[str]:
        """Chunk a single document by filename"""
        # Read the full text
        with open(self.input_path, 'r') as f:
            saved_texts = json.load(f)
        
        if filename not in saved_texts:
            raise ValueError(f"Document {filename} not found in full text file")
        
        full_text = saved_texts[filename]["text"]
        
        # Chunk the text
        chunk_documents = chunk_financial_markdown(full_text)
        chunks = [chunk.page_content for chunk in chunk_documents]
        
        return chunks
    
    def save_chunks(self, filename: str, chunks: List[str]):
        """Save chunks for a document"""
        # Initialize chunks file if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump({}, f, indent=4)
        
        # Save chunks
        with open(self.output_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            saved_chunks = json.load(f)
            saved_chunks[filename] = {"chunks": chunks}
            f.seek(0)
            f.truncate()
            json.dump(saved_chunks, f, indent=4)
    
    def process_document(self, filename: str) -> List[str]:
        """Process a single document: chunk and save"""
        chunks = self.chunk_document(filename)
        self.save_chunks(filename, chunks)
        return chunks
    
    def process_all_files(self):
        """Process all available files that haven't been chunked yet"""
        available_files = self.get_available_files()
        processed_files = self.get_processed_files()
        
        files_to_process = available_files - processed_files
        
        if not files_to_process:
            print("No new files to chunk")
            return
        
        print(f"Chunking {len(files_to_process)} documents...")
        
        for filename in files_to_process:
            try:
                chunks = self.process_document(filename)
                print(f"✓ Chunked {filename} into {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Failed to chunk {filename}: {e}")
        
        print(f"Chunking completed for {len(files_to_process)} documents")

def _merge_small_chunks_with_next(chunks: List[Document]) -> List[Document]:
    """
    Merge small chunks with the chunk below them if they appear to be related.
    This helps create more cohesive chunks with better context.
    """
    if not chunks:
        return chunks
    
    merged_chunks = []
    small_chunk_threshold = 150  # Characters
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        current_text = current_chunk.page_content.strip()
        
        # Check if current chunk is small and should be merged with next
        if (len(current_text) < small_chunk_threshold and 
            i + 1 < len(chunks) and 
            _should_merge_with_next(current_text, chunks[i + 1].page_content)):
            
            # Merge with next chunk
            next_chunk = chunks[i + 1]
            merged_text = current_text + '\n\n' + next_chunk.page_content
            merged_chunks.append(Document(page_content=merged_text))
            i += 2  # Skip the next chunk since we merged it
        else:
            # Keep chunk as is
            merged_chunks.append(current_chunk)
            i += 1
    
    return merged_chunks

def _should_merge_with_next(current_text: str, next_text: str) -> bool:
    """
    Determine if a small chunk should be merged with the next chunk.
    
    Merge if:
    - Current chunk is a header (starts with #)
    - Current chunk is a short section title
    - Current chunk is a brief introductory statement
    """
    current_lines = current_text.strip().split('\n')
    
    # Case 1: Markdown headers should be merged with content below
    if any(line.strip().startswith('#') for line in current_lines):
        return True
    
    # Case 2: Short standalone lines that look like section titles
    if len(current_lines) <= 2:
        for line in current_lines:
            line = line.strip()
            if line:
                # Common section patterns
                if (line.endswith(':') or 
                    any(word in line.lower() for word in ['section', 'chapter', 'part', 'overview', 'summary']) or
                    line.isupper() or  # ALL CAPS titles
                    (len(line.split()) <= 6 and not line.endswith('.'))):  # Short phrases without periods
                    return True
    
    # Case 3: Very short introductory text (less than 50 chars)
    if len(current_text) < 50:
        return True
    
    return False

def chunk_table_aware_markdown(text: str, config=None) -> List[Document]:
    """
    Advanced chunking that handles tables as special units.
    
    Strategy:
    1. Pre-scan to detect all tables and their positions
    2. Identify small text chunks around tables
    3. Create table units (table + surrounding small chunks)
    4. Chunk remaining text with simple chunking (no table splitting)
    5. Combine all chunks
    6. Merge small chunks with chunks below them when appropriate
    
    Args:
        text: Markdown text to chunk
        config: Optional configuration
        
    Returns:
        List of Document chunks
    """
    lines = text.split('\n')
    
    # Step 1: Detect all tables and their boundaries
    table_regions = _detect_table_regions(lines)
    
    if not table_regions:
        # No tables found, use simple chunking for everything
        chunks = _chunk_simple_text(text)
        # Merge small chunks with next chunks
        return _merge_small_chunks_with_next(chunks)
    
    # Step 2: Create table units with surrounding context
    table_units = _create_table_units(lines, table_regions)
    
    # Step 3: Extract non-table content
    non_table_lines = _extract_non_table_content(lines, table_regions)
    
    # Step 4: Chunk the non-table content with simple chunking (no table awareness needed)
    non_table_chunks = []
    if non_table_lines:
        non_table_text = '\n'.join(non_table_lines)
        non_table_chunks = _chunk_simple_text(non_table_text)
    
    # Step 5: Combine all chunks
    all_chunks = table_units + non_table_chunks
    
    # Step 6: Merge small chunks with chunks below them
    return _merge_small_chunks_with_next(all_chunks)

def _detect_table_regions(lines: List[str]) -> List[dict]:
    """
    Detect table regions in the text more robustly.
    
    Returns:
        List of dicts with 'start', 'end', 'type' for each table region
    """
    table_regions = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect markdown table start
        if _is_table_line(line):
            start_idx = i
            
            # Find the end of this table - be more conservative
            j = i + 1
            consecutive_empty = 0
            
            while j < len(lines):
                current_line = lines[j].strip()
                
                if _is_table_line(lines[j]):
                    # Reset empty line counter when we see table content
                    consecutive_empty = 0
                    j += 1
                elif current_line == '':
                    # Allow some empty lines within tables
                    consecutive_empty += 1
                    if consecutive_empty >= 2:  # Too many empty lines, probably end of table
                        break
                    j += 1
                else:
                    # Non-table, non-empty content - table has ended
                    break
            
            # Backtrack to remove trailing empty lines
            end_idx = j - 1
            while end_idx > start_idx and lines[end_idx].strip() == '':
                end_idx -= 1
            
            if end_idx >= start_idx:  # Valid table found
                table_regions.append({
                    'start': start_idx,
                    'end': end_idx,
                    'type': 'markdown_table'
                })
            
            i = j
        else:
            i += 1
    
    return table_regions

def _create_table_units(lines: List[str], table_regions: List[dict]) -> List[Document]:
    """
    Create table units that include tables with surrounding small chunks.
    """
    table_units = []
    small_chunk_threshold = 150  # Characters
    context_lines = 3  # Lines to check before/after
    
    for region in table_regions:
        start_idx = region['start']
        end_idx = region['end']
        
        # Look for small chunks before the table
        pre_context_start = start_idx
        for i in range(max(0, start_idx - context_lines), start_idx):
            line = lines[i].strip()
            if line:  # Non-empty line
                # Check if this and surrounding lines form a small chunk
                chunk_start = i
                chunk_text = []
                
                # Collect lines until we hit empty line or table
                for j in range(i, start_idx):
                    if lines[j].strip():
                        chunk_text.append(lines[j])
                    elif chunk_text:  # Hit empty line after collecting text
                        break
                
                if chunk_text:
                    chunk_content = '\n'.join(chunk_text)
                    if len(chunk_content) <= small_chunk_threshold:
                        pre_context_start = chunk_start
                        break
        
        # Look for small chunks after the table  
        post_context_end = end_idx
        for i in range(end_idx + 1, min(len(lines), end_idx + 1 + context_lines)):
            line = lines[i].strip()
            if line:  # Non-empty line
                # Check if this and surrounding lines form a small chunk
                chunk_text = []
                
                # Collect lines from current position forward
                for j in range(i, min(len(lines), i + context_lines)):
                    if lines[j].strip():
                        chunk_text.append(lines[j])
                    elif chunk_text:  # Hit empty line after collecting text
                        break
                
                if chunk_text:
                    chunk_content = '\n'.join(chunk_text)
                    if len(chunk_content) <= small_chunk_threshold:
                        post_context_end = min(len(lines) - 1, i + len(chunk_text) - 1)
                        break
        
        # Create the table unit
        table_unit_lines = lines[pre_context_start:post_context_end + 1]
        table_unit_text = '\n'.join(table_unit_lines).strip()
        
        if table_unit_text:
            table_units.append(Document(page_content=table_unit_text))
    
    return table_units

def _extract_non_table_content(lines: List[str], table_regions: List[dict]) -> List[str]:
    """
    Extract all content that's not part of table units.
    """
    if not table_regions:
        return lines
    
    non_table_lines = []
    last_end = -1
    
    for region in table_regions:
        # Add content between last table and current table
        if last_end + 1 < region['start']:
            non_table_lines.extend(lines[last_end + 1:region['start']])
        
        last_end = region['end']
    
    # Add content after the last table
    if last_end + 1 < len(lines):
        non_table_lines.extend(lines[last_end + 1:])
    
    return non_table_lines

def _is_table_line(line: str) -> bool:
    """Check if line contains table formatting"""
    stripped = line.strip()
    return bool(stripped and '|' in stripped and len(stripped.split('|')) >= 3)

def _chunk_simple_text(text: str) -> List[Document]:
    """
    Simple chunking for non-table content that doesn't split tables.
    Much simpler than the header-based chunking.
    """
    if not text.strip():
        return []
    
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    current_length = 0
    
    max_chunk_size = 800
    min_chunk_size = 0
    
    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        
        # If adding this line would make chunk too big, finalize current chunk
        if current_length + line_length > max_chunk_size and current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append(Document(page_content=chunk_text))
            
            # Start new chunk with current line
            current_chunk_lines = [line]
            current_length = line_length
        else:
            # Add line to current chunk
            current_chunk_lines.append(line)
            current_length += line_length
    
    # Handle final chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if len(chunk_text) >= min_chunk_size:
            chunks.append(Document(page_content=chunk_text))
        elif chunks:
            # Combine with last chunk if too small
            last_chunk = chunks.pop()
            combined_text = last_chunk.page_content + '\n\n' + chunk_text
            chunks.append(Document(page_content=combined_text))
    
    return chunks

# Keep the existing function name for compatibility
def chunk_financial_markdown(text: str) -> List[Document]:
    """
    Chunk markdown text using table-aware strategy.
    This now uses the improved table-aware chunking.
    """
    return chunk_table_aware_markdown(text)
