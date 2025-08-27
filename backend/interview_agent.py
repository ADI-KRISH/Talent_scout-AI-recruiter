from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("google")

# Reconnect to persisted Chroma DB
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
db = Chroma(
    persist_directory="./job_vector_db",
    embedding_function=embeddings
)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Build interview prompt
def agent_prompt():
    template = """
You are TalentScout, an AI Interviewer.  
Introduce yourself first and then start the interview.

Ask only 5 questions. Candidate’s qualifications:  
Name: {name}, Skills: {skills}, Experience: {experience}, Job: {job}  

Use the following job description context to guide your questions:  
{context}  

- Always keep track of the number of questions you have asked.  
- Critically analyse the candidate's answers and ask relevant follow-up questions.  
- Only keep track of the interview questions you have asked.  
- End the interview by thanking the candidate for their time.  

Conversation so far:
{chat_history}

Candidate: {question}
AI Interviewer:
"""
    prompt = PromptTemplate(
        input_variables=["name", "skills", "experience", "job", "context", "chat_history", "question"],
        template=template,
    )
    return prompt

def build_interview_agent(qualifications):
    prompt = agent_prompt()

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    interview_agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
        return_source_documents=True,
    )

    # Pre-fill static candidate info into chain
    interview_agent.combine_docs_chain.llm_chain.prompt = prompt.partial(
        name=qualifications["Name"],
        skills=qualifications["Skills"],
        experience=qualifications["Experience"],
        job=qualifications["Job"],
    )

    return interview_agent
