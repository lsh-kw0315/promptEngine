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

model = BartForConditionalGeneration.from_pretrained("merve/chatgpt-prompt-generator-v12", from_tf=True)
#model = BartForConditionalGeneration.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", from_tf=True)
tokenizer = BartTokenizer.from_pretrained("merve/chatgpt-prompt-generator-v12")
#tokenizer = BartTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
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


def llama2(request):
    query = request.POST['query']
    data = {}
    answer = chat2(query)
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


def chat2(query):
    chatting = "\n".join(chat_log)
    current_chat = f'The task for you is to rephrase user queries into hyperparameter task prompts that make it easier for other text generation AI to understand the requirements. For example, if I ask: \n\n"I\'m planning a trip to Japan next month. Can you provide some recommendations for must-see attractions and activities in Tokyo?",\n\n you should rephrase it as: \n\n"Provide recommendations on top attractions, activities, and experiences in Tokyo suitable for a one-month trip. Include information on convenient public transportation options, tips for efficient navigation, and any other helpful advice for first-time visitors. [Task: Travel Recommendations] [Destination: Tokyo, Japan] [Duration: 1 month] [Requirements: Top Attractions, Activities, Transportation, Travel Tips]"\n\nYour role is not(never) to *answer* the questions I ask but to rephrase them into *clear prompts with hyperparameters*(must) that make it easier for the text generation AI to understand and provide relevant responses. Now, here\'s my real question. \n Q: "{query}"'
    chatting += current_chat
    output = client.completions.create(
        model="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        prompt=chatting,
        max_tokens=2048, stop=["Q:", "\n"],
    )
    print(chatting)
    answer = str(output.choices[0].text)
    '''deque.append(query + ' ' + answer)
    if len(deque) > 3:
        deque.popleft()'''
    print(answer)
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
