# 🤖 AntiGPT 2.0 — AI Chat & Document Question Answering System

🔗 **Live Demo (Hosted on Render):**  
👉 https://antigpt2.onrender.com

> ⚠️ **Note:** This application is hosted on Render’s free tier.  
> When you open the link, **please wait 30–60 seconds** for the service to wake up automatically.

---

## 📌 Project Overview

**AntiGPT 2.0** is a **ChatGPT-like web application** that supports:

- 🔐 User Signup & Login
- 💬 Persistent chat threads (conversation history)
- 📄 Document upload (PDF, TXT, DOCX)
- 🧠 AI answers based on **uploaded documents (RAG)**
- 🌍 AI answers for **general questions** (ChatGPT-style)
- ☁️ Fully deployed on **Render** via **GitHub**

This project combines **general AI chat** + **document-based question answering** into a single system.

---

## ✨ Key Features

### 🧑‍💻 User System
- Secure user authentication (signup / login / logout)
- Each user has **their own chat threads**
- Chat history is saved and can be continued anytime

### 💬 ChatGPT-Like Chat Experience
- Create multiple chat threads
- Ask general questions like:
  - *“Who is the CEO of Meta?”*
  - *“Give Python if-else examples”*
- AI responds using **general knowledge** when no documents are relevant

### 📄 Document Upload + RAG (Retrieval Augmented Generation)
- Upload documents:
  - PDF
  - DOCX
  - TXT
- System automatically:
  - Extracts text
  - Splits into chunks
  - Creates embeddings
  - Stores them per user
- Ask questions **from your own documents**, e.g.:
  - *“What is my phone number in the resume?”*
  - *“Summarize this document”*

### 🧠 Smart Answer Logic
- If **document context exists** → AI answers from document
- If **no document context** → AI answers from general knowledge
- Works like **ChatGPT + File Upload combined**

---

## 🧱 System Architecture

### 🎨 Frontend
- Django Templates (HTML, CSS, JavaScript)
- AJAX-based chat & file upload
- Single **Upload** button (clean UI)
- Chat thread sidebar + chat window

### ⚙️ Backend
- Django (Python)
- REST APIs for:
  - Threads
  - Messages
  - Document upload
- FAISS / Vector storage for document embeddings
- Secure session-based authentication

### 🤖 AI Engine
- **Primary Generation:** DeepSeek API
- **Embeddings:** OpenAI Embeddings API
- **Fallback Logic:**  
  - Uses document context when available  
  - Uses general LLM knowledge otherwise

### ☁️ Deployment
- Hosted on **Render**
- Connected via **GitHub**
- Uses:
  - Gunicorn
  - WhiteNoise
  - Environment variables for secrets

---

## 🚀 How to Access the Project

1. Open the live URL:
https://antigpt2.onrender.com

yaml
Copy code

2. **Wait ~1 minute** (first load only – Render auto-deploy wake-up)

3. Sign up or log in

4. Start chatting:
- Ask general AI questions
- Upload documents
- Ask document-specific questions

---

## 🖼️ Render Auto-Deploy Notice

When you first open the site, you may see a screen like:

> “Service waking up…”  
> “Allocating compute resources…”

⏳ This is **normal behavior** for Render free tier.  
The app becomes fully active automatically.

---

## 🔐 Environment Variables Used

Configured in Render dashboard:

DJANGO_SECRET_KEY
DEBUG=False
ALLOWED_HOSTS=antigpt2.onrender.com
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
DATABASE_URL=render_postgres_url

yaml
Copy code

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Django |
| AI | DeepSeek API, OpenAI API |
| Vector Store | FAISS |
| Auth | Django Auth |
| Hosting | Render |
| Version Control | GitHub |

---

## 📂 Project Capabilities Summary

✔ ChatGPT-like chat  
✔ Multiple chat threads  
✔ Document upload & RAG  
✔ AI answers from documents  
✔ AI answers from general knowledge  
✔ Secure user accounts  
✔ Fully deployed & live  

---

## 👨‍💻 Author

**Madhesh SR**  
Final-Year BE CSE Student  
Specialization: IoT & Cybersecurity  
Interest: AI, Cloud, DevOps  

🔗 LinkedIn: *MADHESH SR*

---

## ⭐ Final Note

This project demonstrates a **real-world AI system** combining:

- LLM APIs
- Retrieval Augmented Generation (RAG)
- Full-stack development
- Cloud deployment

If you like this project, feel free to ⭐ star the repository!
