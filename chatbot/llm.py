import openai
import bentoml
from bentoml.io import JSON

LLM_MODEL_ID = "gpt2"

openai.api_key = "0K0YA"

class LLMService:
    def __init__(self):
        pass

    def generate(self, json_input):
        prompt = json_input["prompt"]
        response = openai.Completion.create(
            model=LLM_MODEL_ID,
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

llm_service = LLMService()

svc = bentoml.Service("llm_servic", runners=[])

@svc.api(input=JSON(), output=JSON())
def generate(json_input):
    return llm_service.generate(json_input)
