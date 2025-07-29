import os 
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from .prompts import QAOutput, exponential_backoff


async def image_based(query, pages, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=os.getenv("OPENAI_API_KEY"), disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = prompt_template.format(query=query, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]

    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

async def text_based(query, chunks, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=os.getenv("OPENAI_API_KEY"), disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()    

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}                                    
    return response

async def hybrid(query, pages, chunks, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=os.getenv("OPENAI_API_KEY"), disabled_params={"parallel_tool_calls": None, "temperature": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()    

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
