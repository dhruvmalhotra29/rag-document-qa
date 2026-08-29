from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#model = genai.GenerativeModel("gemini-3.6-flash")
model = genai.GenerativeModel("gemini-3.1-flash-lite")

def generate_retrieval_query(question, history):

    if not history:
        return question

    # Only use recent conversation
    history_text = "\n".join(
        f"User: {msg['user']}\nAI: {msg['ai']}"
        for msg in history[-3:]
    )

    prompt = f"""
You are a query router for a document question-answering system.

Determine whether the current question is a follow-up to the recent
conversation or a completely new topic.

If it is a NEW TOPIC:
return exactly:
NEW_TOPIC

If it is a FOLLOW-UP:
rewrite the current question into a standalone search query.

Rules:
- Use the recent conversation to resolve references such as:
  "it", "this", "that", "they", "those", etc.
- Use previous AI answers when they help resolve the reference.
- Preserve the user's intended meaning.
- Do not answer the question.
- Do not add information that isn't present in the conversation.
- Return ONLY:
  NEW_TOPIC
  OR
  the standalone search query.

Recent conversation:
{history_text}

Current question:
{question}
"""

    response = model.generate_content(prompt)

    result = response.text.strip().replace("`", "")

    if result == "NEW_TOPIC":
        return question

    return result

def generate_answer(question, context, history):
    prompt = f"""

You are a helpful AI assistant.

Answer the user's question ONLY using the context below.
If the answer is not present in the context, reply:
"I couldn't find that information in the uploaded document."

Conversation history:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    answer = response.text.strip()

    not_found_message = "I couldn't find that information in the uploaded document."

    if answer == not_found_message:
        return answer, False

    return answer, True