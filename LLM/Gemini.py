import google.generativeai as genai

# gemini 세팅
API_KEY = "AIzaSyCAQsDeXl1LWreDgeYPAbvlJNJhfr2n4Hc"
genai.configure(api_key=API_KEY)


def chat(prompt, model):
    model = genai.GenerativeModel(model)

    response = model.generate_content(prompt)

    return response.text
