# AI Lab Final Project: Hybrid Intelligent Virtual Assistant
Functional deployment link:  https://ai-virtual-assistant-1-xh2u.onrender.com

## Key Features & Fixes Included
1. **Zero Port/Cache Conflict (`http://127.0.0.1:5050`):** Runs on port `5050` with `debug=False` so older cached servers on port `5000` never conflict.
2. **Tier-1 Ensemble NLP Intent Router (`scikit-learn`):** Combines **Logistic Regression** with **TF-IDF Cosine Similarity** for Live Weather, News Headlines, Safe AST Scientific Calculator, Unit Converter, System Diagnostics, Persistent To-Do Manager, Web Launcher, and Extractive Text Summarization.
3. **Tier-2 Open-Domain Knowledge, Dictionary & LLM Engine:**
   - **Free Dictionary API (`dictionaryapi.dev`):** Handles word meanings and definitions (e.g., *"Explain meaning of rigorous"*).
   - **DuckDuckGo Instant Answers + Smart Wikipedia Search:** Answers general everyday and academic questions with zero API keys.
   - **Optional Gemini 2.5 Flash (`google-genai`):** Paste your key in the Web UI sidebar or `.env` file—it validates live and auto-saves to `.env` so it stays active when your teacher runs it.

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
Open **`http://127.0.0.1:5050`** in your browser (or run `python assistant.py` for terminal mode).
