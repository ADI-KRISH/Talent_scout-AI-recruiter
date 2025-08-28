from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY = os.getenv("mistral")
llm = ChatMistralAI(
    model="open-mixtral-8x22b",
    api_key=MISTRAL_API_KEY
)
template = """
You are an expert it hiring manager and recruiter .
You are given the job description skills , experience and the interview transcript of a candidate with another interview agent .{qualification} ,{transcript}
Your task is to evaluate the candidate based on the job description and provide a score out of 10 and a detailed feedback on the candidate's strengths and weaknesses based on the interview transcript 
Dont over think but be critical in your review ."""

def hiring_assistant():
    prompt = PromptTemplate(
        input_variables=['transcript','qualification'],
        template=template
    )
    brain  = llm
    res = brain.invoke(prompt)
    return res.content()

    
    