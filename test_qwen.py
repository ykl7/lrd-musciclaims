from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {"role": "user", "content": "What are the key differences between the Qwen2 and Qwen3 model families?"}
    ]
)

print(response.choices[0].message.content)
