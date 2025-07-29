import re
from typing import List
from langchain.schema import Document

def chunk_by_title_markdown(text: str, config=None) -> List[Document]:
    """
    Chunk markdown text using chunk-by-title strategy.
    
    Rules:
    1. Split on headers (# ## ### ####)
    2. Keep tables together (never split lines with |)
    3. Minimum chunk size of 100 characters
    4. Maximum chunk size of 800 characters
    5. Combine small sections under 150 characters
    
    Args:
        text: Markdown text to chunk
        config: Optional configuration (ignored)
        
    Returns:
        List of Document chunks
    """
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line is a header
        if _is_header(line) and current_chunk_lines:
            # Before starting a new chunk, check if we're in a table
            if not _would_split_table(current_chunk_lines, lines, i):
                # Safe to break - finalize current chunk
                chunk_text = '\n'.join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(_create_chunk(chunk_text))
                
                # Start new chunk with this header
                current_chunk_lines = [line]
            else:
                # Would split a table - continue building current chunk
                current_chunk_lines.append(line)
        else:
            # Add line to current chunk
            current_chunk_lines.append(line)
            
            # Check if chunk is getting too large
            current_text = '\n'.join(current_chunk_lines)
            if len(current_text) > 800:
                # Try to find a safe break point
                break_point = _find_safe_break_point(current_chunk_lines)
                if break_point > 0:
                    # Split here
                    chunk_lines = current_chunk_lines[:break_point]
                    chunk_text = '\n'.join(chunk_lines).strip()
                    if chunk_text:
                        chunks.append(_create_chunk(chunk_text))
                    
                    # Continue with remaining lines
                    current_chunk_lines = current_chunk_lines[break_point:]
        
        i += 1
    
    # Handle final chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append(_create_chunk(chunk_text))
    
    # Post-process: combine small chunks
    return _combine_small_chunks(chunks)

def _is_header(line: str) -> bool:
    """Check if line is a markdown header"""
    stripped = line.strip()
    return bool(re.match(r'^#{1,6}\s+', stripped))

def _is_table_line(line: str) -> bool:
    """Check if line contains table formatting"""
    stripped = line.strip()
    return bool(stripped and '|' in stripped and len(stripped.split('|')) >= 3)

def _would_split_table(current_lines: List[str], all_lines: List[str], break_index: int) -> bool:
    """Check if breaking here would split a table"""
    # Look for table lines around the break point
    start_check = max(0, len(current_lines) - 3)
    
    # Check if we have table lines just before the break
    for i in range(start_check, len(current_lines)):
        if _is_table_line(current_lines[i]):
            # Found table before break, check if table continues after break
            for j in range(break_index, min(len(all_lines), break_index + 5)):
                if _is_table_line(all_lines[j]):
                    return True  # Would split a table
    
    return False

def _find_safe_break_point(lines: List[str]) -> int:
    """Find a safe place to break that doesn't split tables"""
    # Look backwards from the end for a safe break point
    for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
        line = lines[i]
        
        # Don't break on table lines
        if _is_table_line(line):
            continue
        
        # Good break points:
        # 1. After blank lines
        if line.strip() == '':
            return i + 1
        
        # 2. After paragraph endings
        if line.strip().endswith('.') and not _is_table_line(line):
            return i + 1
        
        # 3. After list items
        if re.match(r'^\s*[-*+]\s', line):
            return i + 1
    
    # If no safe break found, break at 75% to avoid infinite growth
    return int(len(lines) * 0.75)

def _create_chunk(text: str) -> Document:
    """Create a Document chunk from text"""
    return Document(page_content=text)

def _combine_small_chunks(chunks: List[Document]) -> List[Document]:
    """Combine chunks that are too small"""
    if not chunks:
        return chunks
    
    combined = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        current_text = current_chunk.page_content
        
        # If chunk is too small and we can combine with next
        if len(current_text) < 150 and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            combined_text = current_text + '\n\n' + next_chunk.page_content
            
            # Only combine if result isn't too large
            if len(combined_text) <= 800:
                combined.append(Document(page_content=combined_text))
                i += 2  # Skip next chunk since we combined it
                continue
        
        # If chunk is too small but can't combine, only add if >= 100 chars
        if len(current_text) >= 100:
            combined.append(current_chunk)
        elif combined:
            # Combine with previous chunk
            prev = combined.pop()
            combined_text = prev.page_content + '\n\n' + current_text
            combined.append(Document(page_content=combined_text))
        
        i += 1
    
    return combined

# Keep the existing function name for compatibility
def chunk_financial_markdown(text: str) -> List[Document]:
    """
    Chunk markdown text using chunk-by-title strategy.
    This is a simple wrapper for compatibility.
    """
    return chunk_by_title_markdown(text)
