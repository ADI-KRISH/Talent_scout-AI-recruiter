from google import genai
from dotenv import load_dotenv
load_dotenv()
import os
s = os.getenv("google")
client = genai.Client(api_key = s)
models = client.models.list()
for model in models:
    print(model.name)
