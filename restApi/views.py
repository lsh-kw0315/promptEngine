from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from .models import Answer
from rest_framework import viewsets
from .serializer import RestApiSerializer
import time
from django.views.decorators.csrf import csrf_exempt
import LLM.LLM as llm
import asyncio
from asgiref.sync import ThreadSensitiveContext, async_to_sync
import time

# json serializer 세팅
class RestApiViewSet(viewsets.ModelViewSet):
    queryset = Answer.objects.all()
    serializer_class = RestApiSerializer


def prompt_page(request):
    # HTML 파일에 넘겨줄 데이터 정의
    now = datetime.now()

    # HTML 파일에 넘겨줄 추가 내역들 삽입하는 곳
    context = {
        "now": now
    }

    # request에 대해 main.html로 context데이터를 넘겨준다.
    return render(request, 'prompt_page.html', context)


def halfauto_generator(request, query):
    print("=================================================start halfauto_generator==============================")

    # 사용자가 입력한 값
    print(query)

    # hyper-parameter형 프롬프트로 기능.
    prompt = (
        "question:\"[input]\" 이 단어들 또는 문장을 분석해서 \"hyper-parameter: [임무:여행 계획 추천], [장소:하와이], [기간:2025년 5월 중순 3박4일], "
        "[추천 목록: 동선, 관광지, 음식 추천, 현지 특이사항, 챙겨야 할 물건], [프롬프트 작성 언어: 한국어]\"와 같이 간결하게 정리 한 뒤,"
        "(중요! 앞에 기술한 hyper-parameter는 이해를 돕기 위해 예시로 적은 것이니 무조건 question에 적힌 것들로만 추론 해서 새로운 hyper-parameter를 작성 할 것.)"
        "이 hyper-parameter에 의거해 의뢰인이 하고자 하는 명령과 명령에 관한 세부적인 사항과 hyper-parameter를 "
        "출력하는 것을 3번 독립적으로 반복하시오. 명령은 \"하시오\"로 마무리하시오. 세부사항을 작성할 때는 주제, 프로세스, 예시 순으로 출력하시오."
        "각 반복에 대한 구분자는 **donedonedone** 입니다."
        "출력할 때는 한국어로 번역하여 출력하시오.").replace("[input]", query)
    print(prompt)

    answer = llm.chat(prompt)
    data = {'query': query, 'answer': answer}

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()

    print("=================================================end halfauto_generator==============================")
    return JsonResponse(data)


# bart로 1차 생성 자동생성 후 LLM에게 인계
def auto_generator(request, query):
    print("=================================================start auto_generator==============================")

    print(query)

    # bart로 유저 input을 구체화시킨 후 LLM의 입력으로 줌
    persona = llm.chat(query, "Bart")[0]
    print(persona)

    prompt = ("당신은 [persona] 일을 보조하는 역할을 맡게 되었습니다. "
              "\n\nInput과 같이 [persona] 일을 하는 LLM에게 명령들과 해당 명령에 관한 세부적인 사항을 출력하는 것을 3번 독립적으로 반복하시오. "
              "\n(주의: 각 반복은 서로 연관된 주제나 순서가 아닌 독립적으로 명확히 구별되는 다른 명령입니다!)"
              "\n(주의: 명령에 대해 LLM이 해당 역할을 하는 봇이 될 수 있도록 프롬프트를 구성하시오.)"
              "\n\n세부사항을 작성할 때는 주제, 프로세스, 예시 순으로 출력하시오. "
              "\n\n사용자에게 입력 받을 텍스트가 있다면 입력받을 수 있게 양식을 만드시오. "
              "\n각 반복은 반드시 \"**donedonedone**\" 로 끝나야합니다."
              "출력할 때는 한국어로 번역하여 출력하시오. \n\nInput: [PromptGenResult]").replace("[persona]", query).replace(
        "[PromptGenResult]", persona)

    #print(prompt)

    answer = llm.chat(prompt)
    data = {'query': query, 'answer': answer, 'intermedia': persona}

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()

    print("=================================================end auto_generator==============================")
    return JsonResponse(data)


# 프롬프트 평가
# form 사용 시 csrf 필수
@csrf_exempt
def geval(request):
    print("=================================================start geval==============================")

    origin_prompt = request.POST['origin']
    result_prompt = request.POST['result']
    
    coherence_instruction = open("geval/coherence/coherence_CoT_ko.txt", encoding="utf-8").read()
    consistency_instruction = open("geval/consistency/consistency_CoT_ko.txt", encoding="utf-8").read()
    fluency_instruction = open("geval/fluency/fluency_CoT_ko.txt", encoding="utf-8").read()
    relevance_instruction = open("geval/relevance/relevance_CoT_ko.txt", encoding="utf-8").read()
    concrete_instruction = open("geval/concrete/concrete_CoT_ko.txt", encoding="utf-8").read()

    coherence_input = open("geval/coherence/coherence_user_input_ko.txt", encoding="utf-8").read().replace(
        '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    consistency_input = open("geval/consistency/consistency_user_input_ko.txt", encoding="utf-8").read().replace(
        '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    fluency_input = open("geval/fluency/fluency_user_input_ko.txt", encoding="utf-8").read().replace('{{Summary}}',
                                                                                                     result_prompt)
    relevance_input = open("geval/relevance/relevance_user_input_ko.txt", encoding="utf-8").read().replace(
        '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    concrete_input = open("geval/concrete/concrete_user_input_ko.txt", encoding="utf-8").read().replace(
        '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)

    # coherence = {"system": coherence_instruction, "user": coherence_input, "assistant": coherence_assistant_example}
    # consistency = {"system": consistency_instruction, "user": consistency_input,
    #                "assistant": consistency_assistant_example}
    # fluency = {"system": fluency_instruction, "user": fluency_input, "assistant": fluency_assistant_example}
    # relevance = {"system": relevance_instruction, "user": relevance_input, "assistant": relevance_assistant_example}

    # coherence_full_prompt = open("geval/coherence/coherence_full_prompt_ko.txt", encoding="utf-8").read().replace(
    #     '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    # consistency_full_prompt = open("geval/consistency/consistency_full_prompt_ko.txt", encoding="utf-8").read().replace(
    #     '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    # fluency_full_prompt = open("geval/fluency/fluency_full_prompt_ko.txt", encoding="utf-8").read().replace(
    #     '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    # relevance_full_prompt = open("geval/relevance/relevance_full_prompt_ko.txt", encoding="utf-8").read().replace(
    #     '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    # concrete_full_prompt = open("geval/concrete/concrete_full_prompt_ko.txt", encoding="utf-8").read().replace(
    #     '{{Document}}', origin_prompt).replace('{{Summary}}', result_prompt)
    
    # prompt_list = [
    #     coherence_full_prompt,
    #     consistency_full_prompt,
    #     fluency_full_prompt,
    #     relevance_full_prompt,
    #     concrete_full_prompt
    # ]
    
    
    prompt_list = [
        {"content":coherence_input,"system_instruction": coherence_instruction},
        {"content":consistency_input,"system_instruction": consistency_instruction},
        {"content":fluency_input,"system_instruction": fluency_instruction},
        {"content":relevance_input,"system_instruction": relevance_instruction},
        {"content":concrete_input,"system_instruction": concrete_instruction}
    ]
    async def geval_getAnswerAsync(prompt):
        return await asyncio.to_thread(geval_getAnswerSystemInst, prompt)

    async def geval_answerAsync(prompt_list):  
        answer_list = await asyncio.gather(
                geval_getAnswerAsync(prompt_list[0]),
                geval_getAnswerAsync(prompt_list[1]),
                geval_getAnswerAsync(prompt_list[2]),
                geval_getAnswerAsync(prompt_list[3]),
                geval_getAnswerAsync(prompt_list[4])
        )

        return {
            "coherence": answer_list[0],
            "consistency": answer_list[1],
            "fluency": answer_list[2],
            "relevance": answer_list[3],
            "concrete": answer_list[4]
        }

    
    
    startTime = time.time()
    data = {"answer":async_to_sync(geval_answerAsync)(prompt_list)}
    # data = {"answer":
    #         {
    #             "coherence": geval_getAnswer(prompt_list[0]),
    #             "consistency": geval_getAnswer(prompt_list[1]),
    #             "fluency": geval_getAnswer(prompt_list[2]),
    #             "relevance":geval_getAnswer(prompt_list[3]),
    #             "concrete": geval_getAnswer(prompt_list[4])
    #         }
    # }
    endTime = time.time()
    print("걸린 시간:", endTime - startTime)
    

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()

    print("=================================================end geval==============================")
    return JsonResponse(data, json_dumps_params={'ensure_ascii': False})


def geval_getAnswer(full_prompt):
    ignore = 0
    try:
        llm_response = llm.chat(full_prompt)
        response = llm_response

        return response

    except Exception as e:
        print(e)
        if ("limit" in str(e)):
            time.sleep(2)
        else:
            ignore += 1
            print('ignored', ignore)

def geval_getAnswerSystemInst(promptObj):
    ignore = 0
    try:
        llm_response = llm.chat(prompt=promptObj["content"], system_inst=promptObj["system_instruction"])
        response = llm_response
        return response

    except Exception as e:
        print(e)
        if ("limit" in str(e)):
            time.sleep(2)
        else:
            ignore += 1
            print('ignored', ignore)

@csrf_exempt
def analysis(request):
    print("=================================================start analysis=============================")
    origin_prompt = request.POST['origin']
    result_prompt = request.POST['result']

    analysis_prompt = open("prompt_templet/analysis.txt",
                           encoding="utf-8").read().replace("[INPUT_PROMPT]", result_prompt)

    data = {'answer': {"security": llm.chat(analysis_prompt)}}
    print(data)

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    print("=================================================end analysis=============================")
    return JsonResponse(data, json_dumps_params={'ensure_ascii': False})

@csrf_exempt
def improve(request):
    print("=================================================start improve=============================")
    origin_prompt = request.POST['origin']
    improve_text = request.POST['request']
    subject = request.POST['subject']

    improve_prompt = (open("prompt_templet/improve.txt",
                          encoding="utf-8").read()
                      .replace("[ORIGIN_PROMPT]", origin_prompt)
                      .replace("[IMPROVE_TEXT]", improve_text)
                      .replace("[SUBJECT]", subject))

    print(improve_prompt)

    data = {'answer': {"improve": llm.chat(improve_prompt)}}

    print(data)

    serializer = RestApiSerializer(data=data)
    if serializer.is_valid():
        serializer.save()

    print("=================================================end improve=============================")
    return JsonResponse(data, json_dumps_params={'ensure_ascii': False})
