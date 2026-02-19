from google import genai
from google.genai import types

client = genai.Client(api_key='gen-lang-client-0469293948')

model = "gemini-2.5-flash"

system_prompt = """
You are a helpful assistant that generates data for a success rate analysis.
"""

user_prompt = """
Generate a list of 1000 problems for a success rate analysis.
"""

response = client.models.generate_content(model=model, contents=[system_prompt, user_prompt])