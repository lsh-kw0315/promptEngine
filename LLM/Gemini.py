from google import generativeai as genai
from google.generativeai import types
import sys
import os

# gemini 세팅
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)


def chat(prompt:str, model, system_inst:str=None):
    model = genai.GenerativeModel(model,system_instruction=system_inst)

    if(system_inst is not None):
        return evalChat(prompt, model)
    else:
      response = model.generate_content(prompt)
      return response.text

def evalChat(prompt:str, model:genai.GenerativeModel):
    response = model.generate_content(contents=prompt,
                                      generation_config=types.GenerationConfig(temperature=0, top_p=0))
    return response.candidates[0].content.parts[0].text