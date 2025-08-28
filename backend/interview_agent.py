import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("google")
MISTRAL_API_KEY = os.getenv("mistral")

# Initialize embeddings (Google)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Connect to persisted Chroma DB
db = Chroma(
    persist_directory="./job_vector_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-medium-2505",
    temperature=0.7,
    max_retries=2,
    api_key=MISTRAL_API_KEY
)

# Build interview prompt
def agent_prompt():
    template = """
You are TalentScout, an AI Interviewer.  
Introduce yourself first and then start the interview.

Ask only 5 questions. Candidate’s qualifications:  
Name: {name}, Skills: {skills}, Experience: {experience}, Job: {job}  

Use the following job description context to guide your questions:  
{context}  
- Don't mention the number of questions in your response. 
- Don't show the thinking part in the response. 
- Only the question should be asked in your response along with the acknowledgement of the previous answer .
- Ask follow-up questions based on the candidate's previous answers.
- Always keep track of the number of questions you have asked.  
- Don't bring up the number of questions you are going to ask at the start or at any point during the interview.
- Critically analyse the candidate's answers and ask relevant follow-up questions.  
- Only keep track of the interview questions you have asked.  
- End the interview by thanking the candidate for their time.  
Do not mention anything like given below .
Note: This is the fifth and final question of the interview. I will thank you for your time after your response.

Candidate Context: Given your experience with FastAPI and REST APIs, I’d like to explore how you’ve leveraged FastAPI’s modern features to build scalable and efficient backend services. This will help assess your ability to design high-performance APIs that meet business needs.

End of Interview: Thank you, Anjali, for your time and thoughtful responses. Your experience and insights have been valuable. We’ll be in touch with the next steps. Have a great day!
Conversation so far:
{chat_history}

Candidate: {question}
AI Interviewer:
"""
    return PromptTemplate(
        input_variables=["name", "skills", "experience", "job", "context", "chat_history", "question"],
        template=template,
    )

# Build the TalentScout agent
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

    # Pre-fill candidate info
    interview_agent.combine_docs_chain.llm_chain.prompt = prompt.partial(
        name=qualifications["Name"],
        skills=qualifications["Skills"],
        experience=qualifications["Experience"],
        job=qualifications["Job"],
    )

    return interview_agent

