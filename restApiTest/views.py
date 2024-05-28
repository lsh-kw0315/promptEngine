from datetime import datetime

from transformers import BartForConditionalGeneration, BartTokenizer
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import viewsets
from .models import LlamaCpp
from .serializer import RestApiTestSerializer
from openai import OpenAI
from collections import deque

import torch
import time

from .models import Answer
import google.generativeai as genai
import os

bot_prompt = "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."

client = OpenAI(base_url="http://localhost:8000/v1", api_key="None")

chat_log = deque()
log_maxlen = 3

'''model = BartForConditionalGeneration.from_pretrained("restApiTest/model/chatgpt-prompt-generator", from_tf=True)
tokenizer = BartTokenizer.from_pretrained("restApiTest/model/chatgpt-prompt-generator")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)'''

#genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# Create your views here.
class RestApiViewSet(viewsets.ModelViewSet):
    queryset = LlamaCpp.objects.all()
    serializer_class = RestApiTestSerializer


def llama(request):
    query = request.POST['query']
    data = {}
    answer = chat(query)
    data['query'] = query
    data['answer'] = answer
    data['chatLog'] = ''.join(chat_log)

    serializer = RestApiTestSerializer(data=data)
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
    '''batch = tokenizer(query, return_tensors="pt")
    batch.to(device)
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    data = {'query': query, 'answer': output[0], 'chatLog': ''}

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data)'''
    
def geval(request):
    origin_prompt = request.POST['origin']
    result_prompt = request.POST['result']

    coherence_instruction = open("restApiTest/geval/coherence/coherence_CoT_ko.txt",encoding="utf-8").read()
    consistency_instruction = open("restApiTest/geval/consistency/consistency_CoT_ko.txt",encoding="utf-8").read()
    fluency_instruction = open("restApiTest/geval/fluency/fluency_CoT_ko.txt",encoding="utf-8").read()
    relevance_instruction = open("restApiTest/geval/relevance/relevance_CoT_ko.txt",encoding="utf-8").read()
    
    coherence_assistant_example=open("restApiTest/geval/coherence/coherence_result_example_ko.txt",encoding="utf-8").read()
    consistency_assistant_example=open("restApiTest/geval/consistency/consistency_result_example_ko.txt",encoding="utf-8").read()
    fluency_assistant_example=open("restApiTest/geval/fluency/fluency_result_example_ko.txt",encoding="utf-8").read()
    relevance_assistant_example=open("restApiTest/geval/relevance/relevance_result_example_ko.txt",encoding="utf-8").read()

    ct, ignore = 0, 0

    coherence_input = open("restApiTest/geval/coherence/coherence_user_input_ko.txt",encoding="utf-8").read().replace('{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    consistency_input =open("restApiTest/geval/consistency/consistency_user_input_ko.txt",encoding="utf-8").read().replace('{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    fluency_input =open("restApiTest/geval/fluency/fluency_user_input_ko.txt",encoding="utf-8").read().replace('{{Summary}}', result_prompt)
    relevance_input = open("restApiTest/geval/relevance/relevance_user_input_ko.txt",encoding="utf-8").read().replace('{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    
    coherence={"system":coherence_instruction,"user":coherence_input,"assistant":coherence_assistant_example}
    consistency={"system":consistency_instruction,"user":consistency_input,"assistant":consistency_assistant_example}
    fluency={"system":fluency_instruction,"user":fluency_input,"assistant":fluency_assistant_example}
    relevance={"system":relevance_instruction,"user":relevance_input,"assistant":relevance_assistant_example}
    
    coherence_full_prompt=open("restApiTest/geval/coherence/coherence_full_prompt_ko.txt",encoding="utf-8").read().replace('{{Document}}',origin_prompt).replace('{{Summary}}',result_prompt)
    consistency_full_prompt=open("restApiTest/geval/consistency/consistency_full_prompt_ko.txt",encoding="utf-8").read().replace('{{Document}}',origin_prompt).replace('{{Summary}}',result_prompt)
    fluency_full_prompt=open("restApiTest/geval/fluency/fluency_full_prompt_ko.txt",encoding="utf-8").read().replace('{{Document}}',origin_prompt).replace('{{Summary}}',result_prompt)
    relevance_full_prompt=open("restApiTest/geval/relevance/relevance_full_prompt_ko.txt",encoding="utf-8").read().replace('{{Document}}',origin_prompt).replace('{{Summary}}',result_prompt)


    data = {}

    try:
        coherence_answer = geval_getAnswer(coherence, coherence_full_prompt)
        consistency_answer = geval_getAnswer(consistency, consistency_full_prompt)
        fluency_answer = geval_getAnswer(fluency, fluency_full_prompt)
        relevance_answer = geval_getAnswer(relevance, relevance_full_prompt)
        data['answer'] = {
            "coherence": coherence_answer,
            "consistency": consistency_answer,
            "fluency": fluency_answer,
            "relevance": relevance_answer
        }

    except Exception as e:
        print(e)
        if ("limit" in str(e)):
            time.sleep(2)
        else:
            ignore += 1
            print('ignored', ignore)

    serializer = RestApiTestSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data, json_dumps_params={'ensure_ascii': False})


def geval_getAnswer(prompt, full_prompt):
    print("받은 프롬프트:")
    #print(prompt)
    print(full_prompt)
    # llm_response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         # {"role":"system","content":full_prompt}
    #         {"role": "system", "content": prompt['system']},
    #         {"role": "user", "content": prompt['user']},
    #         # {"role":"assistant","content":prompt['assistant']},
    #     ],
    #     # prompt=full_prompt,
    #     temperature=1,
    #     max_tokens=200,
    #     top_p=1,
    #     frequency_penalty=2.0,
    #     presence_penalty=0,
    #     # stop='assistant',
    #     # logprobs=40,
    #     # n=5,
    #     # echo=False

    # )
    
    llm_response= genai.GenerativeModel("gemini-pro").generate_content(full_prompt)
    time.sleep(0.5)
    print("llm 응답:")
    print(llm_response)
    # response = [llm_response.choices[i].text for i in range(len(llm_response.choices))]
    #response = [llm_response.choices[i].message.content for i in range(len(llm_response.choices))]
    response=llm_response.text
    return response