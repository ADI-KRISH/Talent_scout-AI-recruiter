import os
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

if not GOOGLE_API_KEY or not JINA_API_KEY or not TAVILY_API_KEY:
    raise ValueError("❌ Missing API keys. Please set 'google', 'JINA_API_KEY', and 'TAVILY_API_KEY' in your .env file.")

# ------------------------
# LLM setup
# ------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# ------------------------
# Search setup
# ------------------------
search_agent = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general",
    include_answer=False,
    include_raw_content=False,
    search_depth="basic",
    time_range="day"
)

# ------------------------
# Step 1: Search for Jobs
# ------------------------
def fetch_job_links(query="Recent Python developer job postings in India", limit=3):
    results = search_agent.run(query)
    job_links = [res["url"] for res in results.get("results", [])[:limit]]
    return job_links

# ------------------------
# Step 2: Scrape & Summarize
# ------------------------
def scrape_and_summarize(job_links):
    all_jobs = []
    for link in job_links:
        try:
            print(f"📥 Scraping: {link}")
            loader = WebBaseLoader(link)
            docs = loader.load()
            text = docs[0].page_content

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
    return all_jobs

# ------------------------
# Step 3: Save to PDF
# ------------------------
def save_to_pdf(text, filename="job_summary.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Job Posting Summaries", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(text, styles['Normal']))
    doc.build(elements)
    print(f"📄 Saved job summaries to {filename}")

# ------------------------
# Step 4: Build Vector DB
# ------------------------
def vector_db():
    job_links = fetch_job_links(limit=3)   # get top 3 jobs
    all_jobs = scrape_and_summarize(job_links)

    if not all_jobs:
        raise ValueError("❌ No job postings could be scraped.")

    # Save PDF
    save_to_pdf("\n\n---\n\n".join(all_jobs))

    # Store in Chroma DB
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(all_jobs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    db = Chroma.from_documents(docs, embeddings, persist_directory="./job_vector_db")
    print("🗄️ Data stored in vector DB at ./job_vector_db")
    return db

# ------------------------
# Run as script (for debugging)
# ------------------------
if __name__ == "__main__":
    db = vector_db()
    print("✅ Vector DB built successfully.")
