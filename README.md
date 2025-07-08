# 📄 Chat with Your Notes – PDF Q&A Bot

This is a Generative AI-based app where you can **upload any PDF (like lecture notes, manuals, books)** and **ask questions** — and the AI will answer intelligently based on the content.

## 🔧 Built With

- [Streamlit](https://streamlit.io/) – for the web interface
- [LangChain](https://www.langchain.com/) – for chaining AI logic
- [OpenAI GPT-3.5](https://openai.com/) – to generate answers
- [ChromaDB](https://www.trychroma.com/) – to store PDF chunks as embeddings

## 📦 Features

- Upload any PDF and chat with its contents
- Uses GPT to return context-aware answers
- Fast, simple, and private (API key stays local)

## 🚀 Getting Started

### Step 1: Clone the repo or download files
```bash
git clone https://github.com/your-username/pdf-chatbot-genai.git
cd pdf-chatbot-genai
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

1. Push this code to GitHub
2. Visit https://streamlit.io/cloud
3. Connect GitHub and deploy the `app.py`

## 📬 License

MIT License – use freely, just give credit!