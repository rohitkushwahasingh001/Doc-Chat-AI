# 🌌 DOC-CHAT AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-8E75B2?style=for-the-badge&logo=google)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20%26%20Agents-green?style=for-the-badge)

**A Next-Gen AI Document Assistant capable of reading handwriting, analyzing Excel data, generating plots, and processing messy PDFs using the power of Google Gemini 2.5 Flash.**

---

## 🚀 Overview

This project is a sophisticated **RAG (Retrieval-Augmented Generation)** application wrapped in a futuristic **3D Sci-Fi Interface**. Unlike traditional chatbots that only read text, this workspace uses **Multimodal AI** to "see" your documents.

It can ingest scanned PDFs, handwritten notes, images, Word docs, and PowerPoint slides. Additionally, it features an autonomous **Data Agent** that can write Python code to analyze Excel sheets and generate visual graphs on the fly.

## ✨ Key Features

* **⚡ Native Multimodal Processing:** Uses Gemini 2.5 Flash to directly process PDFs and Images. It can read complex **handwriting** and interpret charts/diagrams without OCR libraries like Tesseract.
* **📊 Excel Data Agent:** Upload an `.xlsx` file and ask questions like *"Plot a bar chart of sales vs. year"* or *"What is the average of column X?"*. The agent writes code to create the visualization.
* **🏎️ Parallel Processing Engine:** Multi-threaded architecture processes multiple files simultaneously, reducing wait times by up to 80%.
* **🧠 Advanced RAG:** Uses FAISS Vector Store and Google Embeddings to provide accurate, context-aware answers from your knowledge base.
* **🌌 3D Immersive UI:** A custom-styled interface with deep-space backgrounds, glassmorphism, and 3D glowing buttons.

## 🎯 Use Cases

1.  **Student Research:** Upload 10 textbooks and handwritten class notes. Ask the bot to summarize concepts connecting both sources.
2.  **Financial Analysis:** Upload an Excel balance sheet. Ask the bot to plot the revenue trend and calculate year-over-year growth.
3.  **Legal/Medical:** Process scanned contracts or medical prescriptions (handwritten) and digitize/summarize them instantly.

---

## 🛠️ Installation & Local Setup

Follow these steps to run the project on your own machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/gemini-3d-chatbot.git](https://github.com/your-username/gemini-3d-chatbot.git)](https://github.com/rohitkushwahasingh001/Doc-Chat-AI)
cd Doc-Chat-AI
```
### 2. Create a Virtual Environment
```
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Setup Environment Variables
```
GOOGLE_API_KEY="AIzaSy...[Get this from Google AI Studio]"
```
### 5. Run the App
```
Bash

streamlit run app.py
```
☁️ How to Deploy (Streamlit Cloud)
You can make this app live on the internet for free using Streamlit Community Cloud.

Push to GitHub: Upload your code (app.py, requirements.txt) to a GitHub repository. Do not upload .env.

Go to Streamlit Cloud: Login to share.streamlit.io.

New App: Click "New App" and select your repository.

Add Secrets:

Before clicking deploy, go to "Advanced Settings".

Find the "Secrets" box.

Add your API key there:

Ini, TOML

GOOGLE_API_KEY = "AIzaSy...[Your Key]"
Deploy: Click "Deploy" and wait for the app to go live!

📂 Project Structure
Plaintext

Gemini-3D-Workspace/
├── app.py                # Main Application Code (UI + Logic)
├── requirements.txt      # List of Python libraries
├── .env                  # API Key (Local only, ignored by Git)
├── .gitignore            # Files to exclude from Git
├── faiss_index/          # Local storage for vectors (Auto-generated)
└── README.md             # Documentation
🧩 Tech Stack
Frontend: Streamlit (Custom CSS)

LLM: Google Gemini 2.5 Flash (via google-generativeai)

Orchestration: LangChain

Vector Store: FAISS (Facebook AI Similarity Search)

Data Tools: Pandas, Matplotlib, OpenPyXL

Document Parsing: PyPDF2, Python-Docx, Python-PPTX

🤝 Contributing
Contributions are welcome! Please fork this repository and submit a Pull Request.

📄 License
This project is licensed under the MIT License.

