from datetime import datetime
from transformers import BartForConditionalGeneration, BartTokenizer
from django.shortcuts import render
from openai import OpenAI
import torch
from django.http import JsonResponse
from .models import Answer
from rest_framework import viewsets
from .serializer import RestApiSerializer
import google.generativeai as genai

# bart 모델 세팅
model = BartForConditionalGeneration.from_pretrained("restApiTest/model/chatgpt-prompt-generator", from_tf=True)
tokenizer = BartTokenizer.from_pretrained("restApiTest/model/chatgpt-prompt-generator")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


class RestApiViewSet(viewsets.ModelViewSet):
    queryset = Answer.objects.all()
    serializer_class = RestApiSerializer


# llama 세팅
client = OpenAI(base_url="http://localhost:8000/v1", api_key="None")


def prompt_page(request):
    # HTML 파일에 넘겨줄 데이터 정의
    now = datetime.now()

    # HTML 파일에 넘겨줄 추가 내역들 삽입하는 곳
    context = {
        "now": now
    }

    # request에 대해 main.html로 context데이터를 넘겨준다.
    return render(request, 'prompt_page.html', context)


def prompt_generator(request, query):
    persona, output = persona_generator(query).values()
    print(persona)
    print(output)
    bot_prompt = ("당신은 [persona] 일을 보조하는 역할을 맡게 되었습니다. "
                  "\n\nInput과 같이 [persona] 일을 하는 사람에게 명령들과 해당 명령에 관한 세부적인 사항을 출력하는 것을 3번 반복하시오. "
                  "명령은 \"하시오\" 로 마무리하시오.  세부사항을 작성할 때는 주제, 프로세스, 예시 순으로 출력하시오. "
                  "출력할 때는 한국어로 번역하여 출력하시오. \n\nInput: [PromptGenResult]").replace("[persona]", persona).replace("[PromptGenResult]", output)

    print(bot_prompt)

    #answer = llama(bot_prompt)
    answer = gemini(bot_prompt)
    data = {'query': persona, 'answer': answer}

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data)


# bart 실행 코드
def persona_generator(query):
    batch = tokenizer(query, return_tensors="pt")
    batch.to(device)
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    data = {'persona': query, 'output': output[0]}

    return data


# llama 실행 코드
def llama(prompt):
    output = client.completions.create(
        model="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        prompt=prompt,
        max_tokens=1024,
    )

    answer = str(output.choices[0].text.split("\n")[-1])
    return answer


def gemini(bot_prompt) :
    GOOGLE_API_KEY = "AIzaSyAiDzalf8J3i0qFRESAOr-dQ_cbdjXnSFU"

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(bot_prompt)

    return response.text
