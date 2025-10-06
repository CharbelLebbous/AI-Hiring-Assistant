# 🤖 AI Hiring Assistant — Conversational Recruitment Support

An **AI-powered HR assistant** that:
- Indexes and understands **candidate resumes** (PDF, DOCX, TXT)
- Extracts **skills, summaries, and contact info**
- Answers HR questions in **natural language**
- Supports **multi-turn conversations** with chat memory
- Displays **source files** for every answer for full transparency

---

## 🚀 How it works

✅ **Architecture**  
- **Resume Parsing:** Extracts text using `PyMuPDF`, `docx2txt`, and OCR (`pytesseract` + `pdf2image`) for scanned resumes.  
- **Candidate Profiling:** Uses **Cohere’s LLM** to extract skills, contact details, and a short summary for each candidate.  
- **Vector Index:** Stores all candidates’ embeddings locally using **FAISS**.  
- **Conversational Assistant:** Powered by Cohere’s chat model (`command-a-03-2025`) for contextual HR dialogue.  
- **Chat Memory:** Keeps the HR conversation flowing naturally across multiple turns.

✅ **Tools Used**
- [Cohere](https://cohere.com) — for embeddings & LLM chat
- [FAISS](https://github.com/facebookresearch/faiss) — for local semantic search
- [Streamlit](https://streamlit.io) — for the HR chat interface
- **OCR & Parsing:** `PyMuPDF`, `pytesseract`, `pdf2image`, `docx2txt`
- **Python** — the entire stack runs locally, no external backend

---

## 🧩 Features

- 🗣️ **Conversational HR Assistant:** Ask natural questions like:
  - “Show me the top 3 data scientists.”
  - “Who has experience in NLP?”
  - “What is Charbel’s email address?”
- 🔍 **Smart Resume Search:** Uses semantic vector search to find best matches.
- 🧾 **Contact Info Extraction:** Automatically detects and stores emails & phone numbers.
- 🧠 **Multi-turn Context:** Understands follow-up questions within ongoing chat.
- 💾 **Local Indexing:** Build, reset, and re-index candidate data anytime.
- 📂 **Source Transparency:** Shows which resume(s) were used for each answer.

---

## 🏷️ Limitations & Improvements

- **Trial API Limits:** Free Cohere API tier has quota limits — use a paid plan for real HR scale.
- **Privacy:** All resumes stay local — only text embeddings and queries reach Cohere.
- **Ranking Logic:** Current ranking is semantic similarity — can be enhanced with scoring weights.
- **Multilingual Resumes:** Works best in English but supports other languages via Cohere embeddings.

---

## 🖥️ How to Run

1️⃣ **Install dependencies**
```bash
pip install -r requirements.txt

2️⃣ Set your API key
Add a .env file in your project root:
COHERE_API_KEY=your_cohere_api_key_here

3️⃣ Add your candidate resumes
Place all candidate files under:
data/resumes/
Supports .pdf, .docx, .doc, and .txt.

4️⃣ Run the app
streamlit run app.py

5️⃣ Open in your browser:
http://localhost:8501

📂 Project Structure

.
├── app.py              # Streamlit UI (chat + indexing)
├── query_engine.py     # Conversational assistant logic
├── index_builder.py    # Builds FAISS index from resumes
├── ingestion.py        # Extracts resume text, skills, contact info
├── vector_store.py     # Wrapper for FAISS storage & search
├── cohere_client.py    # Wrapper around Cohere's API (embeddings, chat)
├── config.py           # Environment variables and model configs
├── requirements.txt    # Dependencies
├── .gitignore
└── data/resumes/       # Candidate resumes folder

✅ Example Use Cases
Q: Who has experience in NLP?
A: Charbel Lebbous has experience in NLP with tools like spaCy, NLTK, and BERT.

Q: What is Charbel’s email address?
A: Charbel Lebbous’s email address is charbellebbousf@gmail.com.

Q: Show me the top 3 data scientists.
A: Based on available resumes, Charbel Lebbous ranks highest with strong ML and AI expertise.

Each answer includes:
📄 Source Files Used:

Charbel Lebbous CV.pdf

🧠 Internal Flow

Resumes Folder ──▶ Ingestion (text + metadata)
                 ├─▶ Embedding (Cohere)
                 ├─▶ FAISS Vector Index
User Question ──▶ Cohere Query Engine ──▶ Answer + Sources

🧰 Tech Stack
Python 3.10+

Cohere LLMs for embeddings and chat

FAISS for vector search

Streamlit for the interactive web UI

OCR Stack: PyMuPDF + pdf2image + pytesseract

🧹 Maintenance & Reset
🔄 Rebuild Index: Click “🔍 Build / Rebuild Index” in the sidebar to reprocess resumes.

🗑️ Delete Storage: Use “🗑️ Delete Stored Index & Metadata” to reset your database.

Storage files are located in:
./storage/faiss.index
./storage/metadata.json

🧩 Possible Future Upgrades
⏫ Export candidate lists to CSV/Excel

🔗 Integrate with ATS or company HR systems

🌐 Add multilingual resume parsing

💬 Richer chat UI (avatars, timestamps)

🧮 Weighted ranking (skills, experience, education)

👨‍💻 Author
Charbel Lebbous
AI Developer — Machine Learning | Deep Learning | Robotics | Computer Vision | Cybersecurity
📧 charbellebbousf@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/charbel-lebbous/) • 💻 [Github](https://github.com/CharbelLebbous)

📄 License
This project is provided as a technical demo for educational and portfolio purposes.
You are free to adapt, extend, and enhance it for your HR automation workflows.

Made with ❤️ and Cohere
