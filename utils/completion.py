from openai import OpenAI


client = OpenAI(
    api_key="", #https://console.groq.com/keys
    base_url="https://api.groq.com/openai/v1"
)

def complete(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=60
    )

    return response.choices[0].message.content

