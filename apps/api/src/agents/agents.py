from openai import OpenAI

from core.config import config

def run_llm(provider: str, model_name: str, messages: list, max_tokens: int = 2048) -> str:

    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_tokens,
        reasoning_effort="low"
    )

    return response.choices[0].message.content