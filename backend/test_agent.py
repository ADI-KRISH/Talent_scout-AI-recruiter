import os, re, json
from dotenv import load_dotenv
from langchain_community.tools.jina_search import JinaSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from langchain_tavily import TavilySearch
# ------------------------
# Load API keys
# ------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("google")
JINA_API_KEY = os.getenv("JINA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not GOOGLE_API_KEY or not JINA_API_KEY:
    raise ValueError("❌ Missing API keys. Please set 'google' and 'jina' in your .env file.")

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Jina search setup
search_tool = JinaSearch(api_key=JINA_API_KEY)
search_agent = TavilySearch(api_key=TAVILY_API_KEY,
                            max_results=5,
                            topic="general",
                            include_answer=False,
                            include_raw_content=False,
                            search_depth="basic",
                            time_range="day")
# ------------------------
# Step 1: Search for Jobs (Top 3 Links)
# ------------------------
query = "Recent Python developer job postings in India"
results = search_agent.run(query)
print("🔎 Raw Search Results:", results)

# Extract top 3 links
job_links = [res["url"] for res in results["results"][:5]]
print("✅ Top 3 Links:", job_links)

# ------------------------
# Step 2: Scrape & Summarize
# ------------------------
all_jobs = []
for link in job_links:
    try:
        print(f"📥 Scraping: {link}")
        loader = WebBaseLoader(link)
        docs = loader.load()
        text = docs[0].page_content

        # Ask LLM to summarize into JD, Responsibilities, Qualifications
        summary_prompt = f"""
        Extract the following from this job posting:
        - Job Title
        - Job Description
        - Responsibilities
        - Qualifications / Skills Required

        Text: {text}
        """
        summary = llm.invoke(summary_prompt).content
        print("✅ Summary Extracted:\n", summary)
        all_jobs.append(summary)
    except Exception as e:
        print(f"⚠️ Failed to scrape {link}: {e}")

# ------------------------
# Step 3: Save to PDF
# ------------------------
print(all_jobs)
def save_to_pdf(text, filename="job_summary2.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Job Posting Summaries", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(text, styles['Normal']))
    doc.build(elements)

if all_jobs:
    save_to_pdf("\n\n---\n\n".join(all_jobs))
    print("📄 Saved job summaries to PDF as job_summary.pdf")

# ------------------------
# Step 4: Store in Vector DB
# ------------------------
if all_jobs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(all_jobs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./job_vector_db")

    print("🗄️ Data stored in vector DB at ./job_vector_db")
