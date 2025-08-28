from mistralai import Mistral
# Initialize client with your API key
import os
from dotenv import load_dotenv
load_dotenv()
m = os.getenv("mistral")
client = Mistral(api_key=m)

# List available models
models = client.models.list()

# Print model IDs
for model in models.data:
    print(model.id)
