from datetime import datetime
import threading

from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import viewsets
from .models import LlamaCpp
from .serializer import RestApiSerializer
from openai import OpenAI
from collections import deque
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

bot_prompt = "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."

client = OpenAI(base_url="http://localhost:8000/v1", api_key="None")

chat_log = deque()
log_maxlen = 3

model = BartForConditionalGeneration.from_pretrained("restApi/model/chatgpt-prompt-generator", from_tf=True)
tokenizer = BartTokenizer.from_pretrained("restApi/model/chatgpt-prompt-generator")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Create your views here.
class RestApiViewSet(viewsets.ModelViewSet):
    queryset = LlamaCpp.objects.all()
    serializer_class = RestApiSerializer


def llama(request):
    query = request.POST['query']
    data = {}
    answer = chat(query)
    data['query'] = query
    data['answer'] = answer
    data['chatLog'] = ''.join(chat_log)

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data)


def chat(query):
    chatting = "\n".join(chat_log)
    current_chat = f"\nQ : {query}.\nA : "
    chatting += current_chat
    output = client.completions.create(
        model="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        prompt=chatting,
        max_tokens=1024, stop=["Q:", "\n"],
    )

    answer = str(output.choices[0].text.split("\n")[-1]).replace("A : ", "")
    '''deque.append(query + ' ' + answer)
    if len(deque) > 3:
        deque.popleft()'''
    return answer


def prompt_test_page(request):
    # HTML 파일에 넘겨줄 데이터 정의
    now = datetime.now()

    # HTML 파일에 넘겨줄 추가 내역들 삽입하는 곳
    context = {
        "now": now
    }

    # request에 대해 main.html로 context데이터를 넘겨준다.
    return render(request, 'prompt_test_page.html', context)


def prompt_generator(request, query):
    batch = tokenizer(query, return_tensors="pt")
    batch.to(device)
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    data = {'query': query, 'answer': output[0], 'chatLog': ''}

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data)
