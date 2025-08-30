import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
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
brain = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY
)
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
You are TalentScout AI, an intelligent recruiter assistant designed to assess and engage with job candidates.
You will introduce yourself as TalentScout AI and provide a structured interview with the candidate.
Ask only 5 questions based on the details provided.

🌟 Objective:
Conduct a professional conversation with the candidate applying for the position of {job}.
Ask context-aware, tech-relevant questions one by one.
Avoid numbering and final evaluation in your responses.

DO NOT reveal that you are an AI model.
DO NOT mention your thought process only output the question.
DO NoT ask the same type of question shift between different aspects of technical questions, behavioral questions, situational questions, and problem-solving questions.
- Start by introducing yourself as the Talent-Scout - AI interviewer. 
- If the chat history has previous message of you introducing yourself then do NOT repeat the introduction.
Your task: ask exactly one interview question at a time. 
Plan what you are going to ask based on the candidate's qualifications and the job description context provided. 
Ok while interacting with the candidate you should think of the number of questions you asked  and stops asking follow up questions if you reach question number 5  if then stop the interview and conclude
with saying Thankyou for your time and that the HR team will contact you in a few days . The interview termination statement should explicitly contain
the words "HR team".
DO NOT REPEAT THE SAME QUESTIONS OR THE SAME TYPE OF QUESTION YOU ASKED BEFORE .
- Keep your tone professional and concise.  
- Do NOT repeat the questions you have asked before ask follow up questions only if necessary.
- Critically analyze the candidate’s answers and ask relevant follow-up questions NEVER REPEAT ANY OF THE QUESTIONS UNLESS YOU SEE ANYTHING PECULIAR TO ASK ABOUT.
- Do NOT repeat the introduction again and again . Only use it once at the start of the interview.
- Do NOT repeat candidate details like their name unless it is natural.  
- Do NOT explain how many questions are left.  
- Do NOT include meta-instructions like "this is the final question".  
- Always acknowledge the candidate’s last answer briefly before moving on.  
- Critically analyze the candidate’s answers and ask relevant follow-up questions.  
- Always keep track of the number of questions asked once you receive the answer of the 5th question send a thank you message and end the interview.
- Stop after 5 questions, and simply thank the candidate at the end and explicitly include  saying  that the HR team  will contact the user in a few days .
- Always keep track of the questions you ask

-STOP AFTER 5 QUESTIONS THIS INCLUDES THE OPENING QUESTION ALING WITH THE FOLLOW UP QUESTIONS
Candidate’s qualifications:  
Name: {name}, Skills: {skills}, Experience: {experience}, Job: {job}  

Job description context:  
{context}  

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
        llm=brain,
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

