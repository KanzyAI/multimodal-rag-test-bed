from openai import OpenAI
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput, exponential_backoff, get_structured_output
import asyncio
import base64
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

schema = QAOutput.model_json_schema()

async def query_model_async(messages, model, structured = False):

    def sync_query_model():
    
        if structured:
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=QAOutput,
                )
                return completion.choices[0].message.parsed
        
            except Exception as e:
                print(e)
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print(completion)
        return completion.choices[0].message.content
        
    return await asyncio.to_thread(sync_query_model)

    
async def image_based(query, pages, model, prompt_template=IMAGE_PROMPT):
    prompt = prompt_template.format(query=query, schema=schema)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    for p in pages:

        try:
            base64.b64decode(p,validate=True)
        except Exception:
            p = p[0]

        messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    response = await exponential_backoff(query_model_async, messages, model)
    response = await get_structured_output(response)
    return response

async def text_based(query, chunks, model, prompt_template=TEXT_PROMPT):
    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    response = await exponential_backoff(query_model_async, messages, model)
    response = await get_structured_output(response)
    return response

async def hybrid(query, pages, chunks, model, prompt_template=HYBRID_PROMPT):

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    for p in pages:

        try:
            base64.b64decode(p,validate=True)
        except Exception:
            p = p[0]
    
        messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    response = await exponential_backoff(query_model_async, messages, model)
    response = await get_structured_output(response)
    return response

