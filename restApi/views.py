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
import time
import os
from django.views.decorators.csrf import csrf_exempt

# json serializer 세팅
class RestApiViewSet(viewsets.ModelViewSet):
    queryset = Answer.objects.all()
    serializer_class = RestApiSerializer


# bart 모델 세팅
model = BartForConditionalGeneration.from_pretrained("restApiTest/model/chatgpt-prompt-generator", from_tf=True)
tokenizer = BartTokenizer.from_pretrained("restApiTest/model/chatgpt-prompt-generator")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

API_KEY=os.environ['GEMINI_API_KEY']
genai.configure(api_key=API_KEY)
 
# llama 세팅
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="None")


def llama2(request):
    query = request.POST['query']
    data = {}
    answer = chat2(query)
    data['query'] = query
    data['answer'] = answer

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data)


def chat2(query):
    bot_prompt = (f'The task for you is to rephrase user queries into hyperparameter task prompts that make it easier '
                f'for other text generation AI to understand the requirements. For example, if I ask: \n\n"I\'m '
                f'planning a trip to Japan next month. Can you provide some recommendations for must-see attractions '
                f'and activities in Tokyo?",\n\n you should rephrase it as: \n\n"Provide recommendations on top '
                f'attractions, activities, and experiences in Tokyo suitable for a one-month trip. Include '
                f'information on convenient public transportation options, tips for efficient navigation, '
                f'and any other helpful advice for first-time visitors. [Task: Travel Recommendations] [Destination: '
                f'Tokyo, Japan] [Duration: 1 month] [Requirements: Top Attractions, Activities, Transportation, '
                f'Travel Tips]"\n\nYour role is not(never) to *answer* the questions I ask but to rephrase them into '
                f'*clear prompts with hyperparameters*(must) that make it easier for the text generation AI to '
                f'understand and provide relevant responses. Now, here\'s my real question. \n Q: "{query}"')

    answer = llama(bot_prompt)

    print(answer)
    return answer


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
                  "출력할 때는 한국어로 번역하여 출력하시오. \n\nInput: [PromptGenResult]").replace("[persona]", persona).replace(
        "[PromptGenResult]", output)

    print(bot_prompt)

    # answer = llama(bot_prompt)
    answer = gemini(bot_prompt)
    data = {'query': persona, 'answer': answer, 'intermedia':bot_prompt}

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

    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    response = model.generate_content(bot_prompt)

    return response.text

@csrf_exempt
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

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    return JsonResponse(data, json_dumps_params={'ensure_ascii': False})


def geval_getAnswer(prompt, full_prompt):
    print("받은 프롬프트:")
    print(prompt)
    # print(full_prompt)
    # llm_response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role":"system","content":full_prompt}
    #         #{"role": "system", "content": prompt['system']},
    #         #{"role": "user", "content": prompt['user']},
    #         #{"role":"assistant","content":prompt['assistant']},
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