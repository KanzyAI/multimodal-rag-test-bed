from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import os 
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import asyncio 
import re 
import json

class Context(BaseModel):
    context: str

CONTEXTUAL_CHUNKING_PROMPT = PromptTemplate(
    input_variables=["chunk_content"],
    template="""
    Here is the chunk we want to situate within the whole document. 
    
    - Examine the provided image. locate where the chunk is in the image and provide a succinct context to situate the chunk within the overall document for the purposes of improving search retrieval of the chunk.
    - Do not decribe the location of the chunk in the image, just provide the context.
    - If the given chunk is a table, summarize it in a few sentences and provide the context.
    - If the given chunk is an image, like a chart or a graph, describe it in a few sentences and provide the context.
    
    <chunk>
    {chunk_content}
    </chunk>

    Again only return the context to situate the chunk within the overall document. Dont descibe it or add additional commentary.
    Answer only with the succinct context and nothing else.
    """
)

class QAOutput(BaseModel):
    reasoning: str
    answer: str

async def exponential_backoff(func, *args, retries=20, initial_wait=10, **kwargs):
    wait_time = initial_wait
    for attempt in range(retries):
        try:
            response =  await func(*args, **kwargs)
            if response:
                return response
            else:
                raise Exception("No response try again")
            
        except Exception as e:
            if attempt == retries - 1:
                raise e
            print(e)
            await asyncio.sleep(wait_time)
            wait_time += 10

JSONIFY_PROMPT = PromptTemplate(
    input_variables=["response", "schema"],
    template="""
    You are an AI JSON extractor. Your task is to strictly extract and return a JSON object based on the following schema:

    1. Expected JSON Schema:
    {schema}

    Input Response:
    {response}

    Extract and return the JSON object.
    """
)

async def jsonify(response_string):
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()
    prompt = JSONIFY_PROMPT.format(response=response_string, schema=schema)
    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

async def get_structured_output(response):
    json_pattern = r'({.*?})'
    matches = re.findall(json_pattern, ''.join(response), re.DOTALL)
    exception = None

    for match in matches:
        exception = None

        try:       
            reasoning_and_answer_json = match
            parsed_response = json.loads(reasoning_and_answer_json)
            reasoning = parsed_response["reasoning"]
            answer = parsed_response["answer"]

            if type(answer) != str:
                raise Exception("Answer must be a string")
            
            response = {"reasoning": reasoning, "answer": answer}
            return response
        
        except Exception as e:
            exception = e

    try: 
        response = await jsonify(response)
        return response
    
    except Exception as e:
        exception = e

    return {"reasoning": "Error", "answer": "Error", "response": response, "exception": str(exception)}


O1_IMAGE_PROMPT = PromptTemplate(
    input_variables=["query", "schema"],
    template="""
    You are tasked with acting as a financial analyst who excels in evaluating and interpreting financial documents. 
    You will be given context in the form of PDF pages that includes financial data, narratives, or supplementary information. 
    Your objective is to answer a specific financial question based solely on the given context.

    You must produce your answer in the following strict JSON format:

    {{"reasoning": str, "answer": str}}

    - The 'answer' field must be exceptionally concise. Provide numerical values, percentages, or a very brief statement, including units where applicable. Examples include: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Ensure your answer is based exclusively on the provided context.

    Question: {query}
    """
)

IMAGE_PROMPT = PromptTemplate(
    input_variables=["query", "schema"],
    template="""
    You are tasked with acting as a financial analyst who excels in evaluating and interpreting financial documents. 
    You will be given context in the form of PDF pages that includes financial data, narratives, or supplementary information. 
    Your objective is to answer a specific financial question based solely on this context.

    You must produce your answer in the following strict JSON format:
    
    {{"reasoning": str, "answer": str}}

    Instructions:
    - The 'reasoning' field should contain a detailed chain-of-thought as a numbered list, explaining how you utilize the given PDF-based context to derive your answer:
      
      1 - Pinpoint relevant financial data or metrics from the context
      2 - Employ financial reasoning or calculations
      3 - Verify context details
      4 - Conclude with a final decision

    - The 'answer' field must be exceptionally concise. Provide numerical values, percentages, or a very brief statement, including units where applicable. Examples include: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Ensure your reasoning and final answer are based exclusively on the provided context.

    Question: {query}
    """
)

O1_TEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
    You are tasked with acting as a financial analyst who excels in evaluating and interpreting financial documents. 
    You will be given a piece of textual context that includes financial data, narratives, or supplementary information. 
    Your objective is to answer a specific financial question based solely on this context.

    You must produce your answer in the following strict JSON format:

    {{"reasoning": str, "answer": str}}

    - The 'answer' field must be exceptionally concise. Provide numerical values, percentages, or a very brief statement, including units where applicable. Examples include: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Ensure your answer is based exclusively on the provided context.

    Context: {context}

    Question: {query}
    """
)

TEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
    You are tasked with acting as a financial analyst who excels in evaluating and interpreting financial documents. 
    You will be given a piece of textual context that includes financial data, narratives, or supplementary information. 
    Your objective is to answer a specific financial question based solely on this context.

    Your response must adhere to the following strict JSON format:
    
    {{"reasoning": str, "answer": str}}

    Instructions:
    - The 'reasoning' field should contain a detailed chain-of-thought as a numbered list, explaining how you utilize the given textual context to derive your answer:
      
      1 - Pinpoint relevant financial data or metrics from the context
      2 - Employ financial reasoning or calculations
      3 - Verify context details
      4 - Conclude with a final decision

    - The 'answer' field must be exceptionally concise. Provide numerical values, percentages, or a very brief statement, including units where applicable. Examples include: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Ensure your reasoning and final answer are based exclusively on the provided context.

    Context: {context}

    Question: {query}
    """
)

O1_HYBRID_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
    You are tasked with acting as a financial analyst who excels in evaluating and interpreting financial documents. 
    You will be given multimodal context in the form of both PDF pages and textual context that includes financial data, narratives, or supplementary information. 
    Your objective is to answer a specific financial question based solely on this context.

    You must produce your answer in the following strict JSON format:

    {{"reasoning": str, "answer": str}}

    - The 'answer' field must be exceptionally concise. Provide numerical values, percentages, or a very brief statement, including units where applicable. Examples include: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Ensure your reasoning and final answer are based exclusively on the provided multimodal context.

    Context: {context}

    Question: {query}
    """
)

HYBRID_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
    You are tasked with acting as a financial analyst who excels in evaluating and interpreting financial documents. 
    You will be given multimodal context in the form of both PDF pages and textual context that includes financial data, narratives, or supplementary information. 
    Your objective is to answer a specific financial question based solely on this context.

    Your response must adhere to the following strict JSON format:
    
    {{"reasoning": str, "answer": str}}

    Instructions:
    - The 'reasoning' field should contain a detailed chain-of-thought as a numbered list, explaining how you utilize the given multimodal context to derive your answer:
      
      1 - Pinpoint relevant financial data or metrics from the context
      2 - Employ financial reasoning or calculations
      3 - Verify context details
      4 - Conclude with a final decision

    - The 'answer' field must be exceptionally concise. Provide numerical values, percentages, or a very brief statement, including units where applicable. Examples include: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Ensure your reasoning and final answer are based exclusively on the provided multimodal context.

    Context: {context}

    Question: {query}
    """
)
