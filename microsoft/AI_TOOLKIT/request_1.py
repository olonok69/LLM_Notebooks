from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5272/v1/",
    api_key="x",  # required for the API but not used
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "I had a car 20 years ago and at that time I was 37. Answer: How old I am now?",
        }
    ],
    model="Phi-3-mini-128k-cpu-int4-rtn-block-32-acc-level-4-onnx",
)

print(chat_completion.choices[0].message.content)
