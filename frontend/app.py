
import streamlit as st
import sys
from langchain_core.messages import HumanMessage, AIMessage

# Add backend path
sys.path.append(r"C:/Users/GS Adithya Krishna/Desktop/study/agentic ai/interview_agent/backend")
from interview_agent import build_interview_agent

# Finishing keyword
KEY_WORD = "HR Team"

# Streamlit page setup
st.set_page_config(page_title="Talent-Scout AI Interview Agent", page_icon="🤖", layout="centered")

st.title("🤖 Talent-Scout AI Interview Agent")
st.write("Fill in your details to start the interview.")

# --- Session state setup ---
if "interview_started" not in st.session_state:
    st.session_state["interview_started"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None
if "candidate_info" not in st.session_state:
    st.session_state["candidate_info"] = {}
if "interview_over" not in st.session_state:
    st.session_state["interview_over"] = False

# --- Reset memory at the start ---
if not st.session_state["interview_started"]:
    st.session_state["chat_history"] = []
    st.session_state["chain"] = None
    st.session_state["candidate_info"] = {}
    st.session_state["interview_over"] = False

# --- Candidate info form ---
if not st.session_state["interview_started"]:
    with st.form("candidate_form"):
        name = st.text_input("Full Name")
        place = st.text_input("Place")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email")
        skills = st.text_area("Skills (comma separated)")
        job = st.text_input("Job Role You’re Interviewing For")
        experience = st.text_area("Experience (where, what role, how many years)")
        submitted = st.form_submit_button("Start Interview")

    if submitted:
        # Save candidate info
        st.session_state["candidate_info"] = {
            "name": name,
            "place": place,
            "phone": phone,
            "email": email,
            "skills": skills,
            "job": job,
            "experience": experience,
        }

        # Build the interview agent
        qualifications = {
            "Name": name,
            "Skills": skills,
            "Experience": experience,
            "Job": job,
        }
        st.session_state["chain"] = build_interview_agent(qualifications)
        st.session_state["interview_started"] = True

        # Trigger first AI response (self-introduction / first question)
        first_response = st.session_state["chain"].invoke({
            "question": "Introduce yourself as the AI interviewer and ask the first interview question.",
            "chat_history": [],  # start fresh
            **st.session_state["candidate_info"]
        })
        st.session_state["chat_history"].append(("AI Interviewer", first_response["answer"]))

# --- Chat interface ---
if st.session_state["interview_started"] and not st.session_state["interview_over"]:
    st.subheader("💬 Interview Chat")

    # Display previous chat
    for role, msg in st.session_state["chat_history"]:
        st.chat_message("user" if role == "Candidate" else "assistant").write(msg)

    # Candidate input
    user_input = st.chat_input("Your response...")
    if user_input:
        # Add candidate message immediately
        st.session_state["chat_history"].append(("Candidate", user_input))
        st.chat_message("user").write(user_input)

        # Generate AI follow-up (fresh context)
        response = st.session_state["chain"].invoke({
            "question": f"The candidate responded: {user_input}. Continue the interview with a relevant follow-up question.",
            "chat_history": [],  # do not use internal memory
            **st.session_state["candidate_info"]
        })

        # Add AI response
        ai_msg = response["answer"]
        st.session_state["chat_history"].append(("AI Interviewer", ai_msg))
        st.chat_message("assistant").write(ai_msg)

        # Check for finishing keyword
        if KEY_WORD.lower() in ai_msg.lower():
            st.session_state["interview_over"] = True

            # Clear chat memory / LLM buffer
            if st.session_state.get("chain"):
                if hasattr(st.session_state["chain"], "memory") and st.session_state["chain"].memory:
                    st.session_state["chain"].memory.clear()

            # Remove chain and candidate info from session state
            st.session_state["chain"] = None
            st.session_state["candidate_info"] = {}
            st.session_state["chat_history"] = []

            st.success(f"The interview has ended . Thankyou for your time .")

# import streamlit as st
# import sys
# from langchain_core.messages import HumanMessage, AIMessage

# # Add backend path
# sys.path.append(r"C:/Users/GS Adithya Krishna/Desktop/study/agentic ai/interview_agent/backend")
# from interview_agent import build_interview_agent
# KEY_WORD  = "HR Team"

# # Streamlit page setup
# st.set_page_config(page_title="Talent-Scout AI Interview Agent", page_icon="🤖", layout="centered")

# st.title("🤖 Talent-Scout AI Interview Agent")
# st.write("Fill in your details to start the interview.")

# # --- Session state setup ---
# if "interview_started" not in st.session_state:
#     st.session_state["interview_started"] = False
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
# if "chain" not in st.session_state:
#     st.session_state["chain"] = None
# if "candidate_info" not in st.session_state:
#     st.session_state["candidate_info"] = {}
# if "interview_over" not in st.session_state:
#     st.session_state["interview_over"] = False

# # --- Candidate info form ---
# if not st.session_state["interview_started"]:
#     with st.form("candidate_form"):
#         name = st.text_input("Full Name")
#         place = st.text_input("Place")
#         phone = st.text_input("Phone Number")
#         email = st.text_input("Email")
#         skills = st.text_area("Skills (comma separated)")
#         job = st.text_input("Job Role You’re Interviewing For")
#         experience = st.text_area("Experience (where, what role, how many years)")
#         submitted = st.form_submit_button("Start Interview")

#     if submitted:
#         # Save candidate info
#         st.session_state["candidate_info"] = {
#             "name": name,
#             "place": place,
#             "phone": phone,
#             "email": email,
#             "skills": skills,
#             "job": job,
#             "experience": experience,
#         }

#         # Build the interview agent
#         qualifications = {
#             "Name": name,
#             "Skills": skills,
#             "Experience": experience,
#             "Job": job,
#         }
#         st.session_state["chain"] = build_interview_agent(qualifications)
#         st.session_state["interview_started"] = True

#         # Trigger first AI response (self-introduction / first question)
#         first_response = st.session_state["chain"].invoke({
#             "question": "Introduce yourself as the AI interviewer and ask the first interview question.",
#             "chat_history": [],
#             **st.session_state["candidate_info"]
#         })
#         st.session_state["chat_history"].append(("AI Interviewer", first_response["answer"]))
# # --- Candidate input (only if interview is not over) ---
# if not st.session_state["interview_over"]:
#     user_input = st.chat_input("Your response...")
#     if user_input:
#         # Add candidate message immediately
#         st.session_state["chat_history"].append(("Candidate", user_input))
#         st.chat_message("user").write(user_input)

#         # Generate AI follow-up
#         response = st.session_state["chain"].invoke({
#             "question": f"The candidate responded: {user_input}. Continue the interview with a relevant follow-up question.",
#             "chat_history": st.session_state["chat_history"],
#             **st.session_state["candidate_info"]
#         })

#         # Add AI response
#         ai_msg = response["answer"]
#         st.session_state["chat_history"].append(("AI Interviewer", ai_msg))
#         st.chat_message("assistant").write(ai_msg)

#         # Check for finishing keyword
#         if KEY_WORD.lower() in ai_msg.lower():
#             st.session_state["interview_over"] = True

#     # Clear internal chat memory of the chain
#             if st.session_state.get("chain"):
#                 if hasattr(st.session_state["chain"], "memory") and st.session_state["chain"].memory:
#                     st.session_state["chain"].memory.clear()

#     # Remove chain and candidate info from session state
#         st.session_state["chain"] = None
#         st.session_state["candidate_info"] = {}

#         st.success(f"✅ The interview has ended. The AI concluded the session with: '{KEY_WORD}'.")

            
            
# import streamlit as st
# import sys
# from langchain_core.messages import HumanMessage, AIMessage

# # Add backend path
# sys.path.append(r"C:/Users/GS Adithya Krishna/Desktop/study/agentic ai/interview_agent/backend")
# from interview_agent import build_interview_agent
# KEY_WORD  = "HR Team"

# # Streamlit page setup
# st.set_page_config(page_title="Talent-Scout AI Interview Agent", page_icon="🤖", layout="centered")

# st.title("🤖 Talent-Scout AI Interview Agent")
# st.write("Fill in your details to start the interview.")

# # --- Session state setup ---
# if "interview_started" not in st.session_state:
#     st.session_state["interview_started"] = False
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
# if "chain" not in st.session_state:
#     st.session_state["chain"] = None
# if "candidate_info" not in st.session_state:
#     st.session_state["candidate_info"] = {}
# if "interview_over" not in st.session_state:
#     st.session_state["interview_over"] = False

# # --- Candidate info form ---
# if not st.session_state["interview_started"]:
#     with st.form("candidate_form"):
#         name = st.text_input("Full Name")
#         place = st.text_input("Place")
#         phone = st.text_input("Phone Number")
#         email = st.text_input("Email")
#         skills = st.text_area("Skills (comma separated)")
#         job = st.text_input("Job Role You’re Interviewing For")
#         experience = st.text_area("Experience (where, what role, how many years)")
#         submitted = st.form_submit_button("Start Interview")

#     if submitted:
#         # Save candidate info
#         st.session_state["candidate_info"] = {
#             "name": name,
#             "place": place,
#             "phone": phone,
#             "email": email,
#             "skills": skills,
#             "job": job,
#             "experience": experience,
#         }

#         # Build the interview agent
#         qualifications = {
#             "Name": name,
#             "Skills": skills,
#             "Experience": experience,
#             "Job": job,
#         }
#         st.session_state["chain"] = build_interview_agent(qualifications)
#         st.session_state["interview_started"] = True

#         # Trigger first AI response (self-introduction / first question)
#         first_response = st.session_state["chain"].invoke({
#             "question": "Introduce yourself as the AI interviewer and ask the first interview question.",
#             "chat_history": [],
#             **st.session_state["candidate_info"]
#         })
#         st.session_state["chat_history"].append(("AI Interviewer", first_response["answer"]))
# # --- Candidate input (only if interview is not over) ---
# if not st.session_state["interview_over"]:
#     user_input = st.chat_input("Your response...")
#     if user_input:
#         # Add candidate message immediately
#         st.session_state["chat_history"].append(("Candidate", user_input))
#         st.chat_message("user").write(user_input)

#         # Generate AI follow-up
#         response = st.session_state["chain"].invoke({
#             "question": f"The candidate responded: {user_input}. Continue the interview with a relevant follow-up question.",
#             "chat_history": st.session_state["chat_history"],
#             **st.session_state["candidate_info"]
#         })

#         # Add AI response
#         ai_msg = response["answer"]
#         st.session_state["chat_history"].append(("AI Interviewer", ai_msg))
#         st.chat_message("assistant").write(ai_msg)

#         # Check for finishing keyword
#         if KEY_WORD.lower() in ai_msg.lower():
#             st.session_state["interview_over"] = True
#             # Clear chat memory / LLM buffer
#             st.session_state["chain"] = None
#             st.session_state["candidate_info"] = {}
#             st.success(f"✅ The interview has ended. The AI has concluded the session with: '{KEY_WORD}'.")


# # import streamlit as st
# import sys
# from langchain_core.messages import HumanMessage, AIMessage

# # Add backend path
# sys.path.append(r"C:/Users/GS Adithya Krishna/Desktop/study/agentic ai/interview_agent/backend")
# from interview_agent import build_interview_agent

# # Streamlit page setup
# st.set_page_config(page_title="Talent-Scout AI Interview Agent", page_icon="🤖", layout="centered")

# st.title("🤖 Talent-Scout AI Interview Agent")
# st.write("Fill in your details to start the interview.")

# # --- Session state setup ---
# if "interview_started" not in st.session_state:
#     st.session_state["interview_started"] = False
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
# if "chain" not in st.session_state:
#     st.session_state["chain"] = None
# if "candidate_info" not in st.session_state:
#     st.session_state["candidate_info"] = {}

# # --- Candidate info form ---
# if not st.session_state["interview_started"]:
#     with st.form("candidate_form"):
#         name = st.text_input("Full Name")
#         place = st.text_input("Place")
#         phone = st.text_input("Phone Number")
#         email = st.text_input("Email")
#         skills = st.text_area("Skills (comma separated)")
#         job = st.text_input("Job Role You’re Interviewing For")
#         experience = st.text_area("Experience (where, what role, how many years)")
#         submitted = st.form_submit_button("Start Interview")

#     if submitted:
#         # Save candidate info
#         st.session_state["candidate_info"] = {
#             "name": name,
#             "place": place,
#             "phone": phone,
#             "email": email,
#             "skills": skills,
#             "job": job,
#             "experience": experience,
#         }

#         # Build the interview agent
#         qualifications = {
#             "Name": name,
#             "Skills": skills,
#             "Experience": experience,
#             "Job": job,
#         }
#         st.session_state["chain"] = build_interview_agent(qualifications)
#         st.session_state["interview_started"] = True

#         # Trigger first AI response
#         first_response = st.session_state["chain"].invoke({
#             "question": "Introduce yourself as the AI interviewer and ask the first interview question.",
#             "chat_history": [],
#             **st.session_state["candidate_info"]
#         })
#         st.session_state["chat_history"].append(("AI Interviewer", first_response["answer"]))

# # --- Chat interface ---
# if st.session_state["interview_started"]:
#     st.subheader("💬 Interview Chat")

#     # Candidate input
#     user_input = st.chat_input("Your response...")
#     if user_input:
#         # 1️⃣ Immediately display the user's message
#         st.session_state["chat_history"].append(("Candidate", user_input))

#         # Display updated chat so the user sees their message immediately
#         for role, msg in st.session_state["chat_history"]:
#             st.chat_message("user" if role == "Candidate" else "assistant").write(msg)

#         # 2️⃣ Generate AI response after showing user's message
#         response = st.session_state["chain"].invoke({
#             "question": f"The candidate responded: {user_input}. Continue the interview with a relevant follow-up question.",
#             "chat_history": st.session_state["chat_history"],
#             **st.session_state["candidate_info"]
#         })
#         st.session_state["chat_history"].append(("AI Interviewer", response["answer"]))

#     # Display the full chat
#     for role, msg in st.session_state["chat_history"]:
#         st.chat_message("user" if role == "Candidate" else "assistant").write(msg)
