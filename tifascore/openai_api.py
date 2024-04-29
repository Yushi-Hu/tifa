import openai
import time, sys


def openai_completion(prompt, engine="gpt-3.5-turbo", max_tokens=700, temperature=0):
    client = openai.OpenAI()
    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n", "<|endoftext|>"]
        )
    
    return resp.choices[0].message.content



