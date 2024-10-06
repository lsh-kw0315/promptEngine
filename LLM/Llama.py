from openai import OpenAI

# llama 세팅
# llama_server로 배포하면 OpenAI api로 사용 가능
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="None")


def chat(prompt, model):
    output = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1024,
    )

    answer = str(output.choices[0].text.split("\n")[-1])

    print(answer)
    return answer
