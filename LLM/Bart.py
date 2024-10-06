from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# bart 모델 세팅
model = BartForConditionalGeneration.from_pretrained("restApiTest/model/chatgpt-prompt-generator", from_tf=True)
tokenizer = BartTokenizer.from_pretrained("restApiTest/model/chatgpt-prompt-generator")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# bart 실행 코드
def chat(prompt):
    batch = tokenizer(prompt, return_tensors="pt")
    batch.to(device)
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return output

