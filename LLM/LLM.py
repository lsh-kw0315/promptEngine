from .Bart import chat as bart_chat
from .Llama import chat as llama_chat
from .Gemini import chat as gemini_chat

# 현재 사용 중인 LLM(Gemini, Llama, Bart), 기본은 Gemini
currentLLM = "Gemini"
# LLM에서 사용할 모델, 기본은 gemini의 1.5-flash-latest
# Bart는 한 가지 모델만 사용하므로 model 파라미터를 안 받음
currentModel = "gemini-1.5-flash-latest"


# 사용 모델 변경
def LLMSetting(llm):
    currentModel = llm


# 모델에 따라 다른 LLM이 사용됨
# model 파라미터를 안넘기면 currentModel로 동작
def chat(prompt, llm=currentLLM, model=currentModel):
    if llm == "Gemini":
        return gemini_chat(prompt, model)
    elif llm == "Llama":
        return llama_chat(prompt, model)
    elif llm == "Bart":
        return bart_chat(prompt)
