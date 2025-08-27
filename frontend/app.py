import streamlit as st
import sys
sys.path.append(r"C:/Users/GS Adithya Krishna\Desktop\study/agentic ai/interview_agent\backend")
from interview_agent import build_interview_agent

st.set_page_config(page_title="Talent-Scout AI Interview Agent", page_icon="🤖", layout="centered")

st.title("🤖 Talent-Scout  AI Interview Agent")
st.write("Fill in your details to start the interview.")

with st.form("candidate_form"):
    name = st.text_input("Full Name")
    place = st.text_input("Place")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email")
    skills = st.text_area("Skills (comma separated)")
    job = st.text_input("Job Role You’re Interviewing For")
    experience = st.text_area("Experience (where, what role, how many years)")
    submitted = st.form_submit_button("Start Interview")

# Save candidate info in session
if submitted:
    st.session_state["candidate_info"] = {
        "name": name,
        "place": place,
        "phone": phone,
        "email": email,
        "skills": skills,
        "job": job,
        "experience": experience,
    }
    qualifications = {
        "Name": name,
        "Skills": skills,
        "Experience": experience,
        "Job": job,
    }
    st.session_state["interview_started"] = True
    st.session_state["chain"] = build_interview_agent(qualifications)
    st.session_state["chat_history"] = []

# Chat Interface
if st.session_state.get("interview_started", False):
    st.subheader("💬 Interview Chat")

    # Candidate input
    user_input = st.chat_input("Your response...")
    if user_input:
        chain = st.session_state["chain"]

        response = chain.invoke({
            "question": user_input,
            "chat_history": st.session_state["chat_history"],
            **st.session_state["candidate_info"]
        })

        # Update history
        st.session_state["chat_history"].append(("Candidate", user_input))
        st.session_state["chat_history"].append(("AI Interviewer", response["answer"]))

    # Display chat
    for role, msg in st.session_state["chat_history"]:
        if role == "Candidate":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)
