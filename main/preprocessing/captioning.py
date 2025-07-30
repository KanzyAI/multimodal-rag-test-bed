
import os
import re
import base64
import io
import asyncio
import random
from typing import Dict, List, Any, Type
from pydantic import BaseModel, create_model
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def parse_image_references(text: str) -> List[str]:
    # Pattern to match ![img-N.jpeg](img-N.jpeg)
    pattern = r'!\[img-(\d+)\.jpeg\]\(img-\d+\.jpeg\)'
    matches = re.findall(pattern, text)
    
    # Convert to full image names and sort by number
    image_refs = [f"img-{match}.jpeg" for match in matches]
    # Sort by the number to ensure proper order
    image_refs.sort(key=lambda x: int(re.search(r'img-(\d+)\.jpeg', x).group(1)))
    
    return image_refs

def create_dynamic_caption_model(image_refs: List[str]) -> Type[BaseModel]:
    """
    Creates a dynamic Pydantic model with fields for each image reference.
    
    Args:
        image_refs: List of image references like ['img-0.jpeg', 'img-1.jpeg']
    
    Returns:
        A Pydantic model class with caption fields for each image
    """
    if not image_refs:
        # Return a simple model with no image fields
        return create_model('EmptyCaptionModel')
    
    # Create fields dictionary for the dynamic model
    fields = {}
    for img_ref in image_refs:
        # Clean the image reference to make it a valid field name
        field_name = img_ref.replace('.jpeg', '').replace('-', '_')
        fields[field_name] = (str, ...)  # Required string field
    
    # Create the dynamic model
    DynamicCaptionModel = create_model(
        'DynamicCaptionModel',
        **fields
    )
    
    return DynamicCaptionModel

def convert_image_to_base64(document_image: Any) -> str:
    """
    Convert document image to base64.
    
    Args:
        document_image: The document image (PIL Image or similar)
    
    Returns:
        Base64 encoded image string
    """
    # Convert PIL Image to base64 if needed
    if hasattr(document_image, 'save'):
        # It's a PIL Image
        img_bytes = io.BytesIO()
        document_image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    elif isinstance(document_image, str):
        # It's already base64
        return document_image
    else:
        raise ValueError(f"Unsupported image type: {type(document_image)}")

def replace_images_with_captions(text: str, captions: Dict[str, str]) -> str:
    """
    Replace image references in text with their captions.
    
    Args:
        text: The original text with image references
        captions: Dict mapping image references to their captions
    
    Returns:
        Text with image references replaced by captions
    """
    modified_text = text
    
    for img_ref, caption in captions.items():
        # Pattern to match the specific image reference
        pattern = f'!\\[{img_ref}\\]\\({img_ref}\\)'
        # Replace with the caption
        modified_text = re.sub(pattern, "\n\n" + caption + "\n\n", modified_text)
    
    return modified_text

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Factor to multiply delay by each retry
        jitter: Whether to add random jitter to delay
    
    Returns:
        Result of the function call
    
    Raises:
        The last exception encountered after all retries
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            result = await func()
            return result
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                print(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
                break
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    raise last_exception

CAPTIONING_PROMPT = PromptTemplate(
    input_variables=["document_text", "image_refs"],
    template="""
    You are a financial document analysis specialist. Your task is to generate dense, contextual captions for images within financial reports, SEC filings, and investor presentations.

    DOCUMENT CONTEXT:
    {document_text}

    IMAGE REFERENCES TO CAPTION: {image_refs}

    FINANCIAL ANALYSIS FRAMEWORK:
    1. **Chart/Graph Type Recognition**: Identify financial visualizations (revenue charts, margin trends, performance dashboards, etc.)
    2. **Financial Metrics Extraction**: Capture specific KPIs, financial ratios, monetary values, percentages, growth rates
    3. **Temporal Analysis**: Note time periods, trend directions, comparative analysis, year-over-year changes
    4. **Business Context Integration**: Connect visual data to financial performance, market conditions, strategic initiatives

    CAPTION STRUCTURE (for each image):
    - **Financial Content Type**: Chart type and primary financial focus
    - **Quantitative Details**: Specific metrics, values, ratios, growth rates visible
    - **Temporal Context**: Time periods covered, trend analysis, comparative timeframes
    - **Business Intelligence**: Financial insights, performance indicators, strategic implications
    - **Retrieval Keywords**: Financial terminology, accounting concepts, business metrics

    QUALITY REQUIREMENTS:
    ✓ Dense financial information content (3-5 sentences per image)
    ✓ Precise financial terminology and GAAP/IFRS concepts
    ✓ Quantitative accuracy and trend identification
    ✓ Business context and strategic relevance
    ✓ Optimal retrieval for financial analysis queries

    EXAMPLES OF SOTA FINANCIAL CAPTIONS:
    - img_0: "Revenue growth analysis chart displaying quarterly performance from Q1 2021 to Q4 2023. Shows consistent upward trend with 12% year-over-year growth in 2023, reaching $2.4B in Q4 2023 compared to $2.1B in Q4 2022. Revenue breakdown by segments indicates Services growing 18% while Products segment remained flat at 2% growth. Chart includes dotted trend line projecting continued growth trajectory into 2024."
    
    - img_1: "Executive compensation summary table presenting CEO total compensation structure for fiscal year 2023. Base salary of $1.5M represents 15% of total compensation, with performance-based equity awards comprising 70% ($7.0M) tied to TSR performance vs S&P 500. Annual cash incentive of $1.5M reflects achievement of 120% of target based on revenue growth and margin expansion metrics. Long-term incentive vesting schedule spans three years with cliff vesting in 2026."
    
    - img_2: "Balance sheet liquidity analysis dashboard showing cash position and debt metrics as of December 31, 2023. Cash and equivalents total $3.2B, up 15% from prior year, while total debt decreased to $8.5B representing debt-to-equity ratio of 0.65. Current ratio improved to 1.8x from 1.6x, indicating strengthened working capital position. Credit facility utilization remains low at 25% of $2B revolving credit line."

    Generate comprehensive, financially accurate captions that maximize retrieval effectiveness for investment analysis, due diligence, and financial research queries.
    """
)

async def captioning(document_image, document_text, model_type, max_retries: int = 20):
    """
    Caption all images referenced in a document and replace them with captions.
    Uses exponential backoff retry logic to handle errors gracefully.
    
    Args:
        document_image: The full document image (PIL Image or base64)
        document_text: The OCR text containing image references
        model_type: The model to use for captioning
        max_retries: Maximum number of retry attempts
    
    Returns:
        Text with image references replaced by captions
    """
    # Parse image references from the text
    image_refs = parse_image_references(document_text)
    
    if not image_refs:
        # No images to caption, return original text
        return document_text
    
    # Create dynamic Pydantic model for all images
    DynamicCaptionModel = create_dynamic_caption_model(image_refs)
    
    # Set up the LLM with structured output
    llm = ChatGoogleGenerativeAI(model=model_type, api_key=os.getenv("GOOGLE_API_KEY"))
    llm = llm.with_structured_output(DynamicCaptionModel)
    
    # Prepare the prompt
    prompt = CAPTIONING_PROMPT.format(
        document_text=document_text,
        image_refs=image_refs
    )
    
    # Convert document image to base64 if needed
    document_image_b64 = convert_image_to_base64(document_image)
    
    # Prepare content for the model
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{document_image_b64}"}
        },
        {
            "type": "text",
            "text": prompt
        }
    ]
    
    message = HumanMessage(content=content)
    
    async def attempt_captioning():
        """Inner function that performs the actual captioning attempt"""
        response = await llm.ainvoke([message])
        
        captions = {}
        for img_ref in image_refs:
            field_name = img_ref.replace('.jpeg', '').replace('-', '_')
            caption = getattr(response, field_name, f"Visual content from {img_ref}")
            
            # Validate caption quality
            if caption == img_ref or caption == field_name or len(caption.strip()) < 5:
                error_msg = f"Caption for {img_ref} is too short or invalid: {caption}"
                print(f"Caption quality issue: {error_msg}")
                raise ValueError(error_msg)
            
            captions[img_ref] = caption
        
        return captions
    
    # Attempt captioning with exponential backoff retry
    print(f"Starting captioning process for {len(image_refs)} images...")
    captions = await retry_with_exponential_backoff(
        attempt_captioning,
        max_retries=max_retries,
        base_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
        jitter=True
    )
    
    print(f"Successfully generated captions for all {len(image_refs)} images")
    captioned_text = replace_images_with_captions(document_text, captions)
    return captioned_text

