"""
AI Lab Final Project: Hybrid Intelligent Virtual Assistant
----------------------------------------------------------
Complete, bug-free implementation featuring:
1. Tier 1 — Local Ensemble NLP Router (Logistic Regression + TF-IDF Cosine Similarity):
   - Live Weather (wttr.in)
   - Live News Headlines (Google News RSS)
   - Safe AST Scientific Calculator (sin, cos, tan, sqrt, log, factorial, pi, e)
   - Unit Converter (Length, Weight, Temperature)
   - System Diagnostics (CPU, RAM, OS, Battery)
   - Persistent To-Do Manager (Add, List, Complete)
   - Web Automation (Open sites, Google search, YouTube search)
   - Extractive NLP Text Summarizer (TF-IDF sentence scoring)
2. Tier 2 — Open-Domain General Knowledge & Conversational Brain:
   - Gemini 2.5 Flash LLM (`google-genai`) when GEMINI_API_KEY is present in `.env`
     or entered via the Web UI (automatically persisted to `.env` for teacher runs!).
   - Keyless Smart Fallback Chain (works out-of-the-box with zero API keys):
     a) Free Dictionary API (`dictionaryapi.dev`) for word meanings & definitions
        (e.g., "explain meaning of rigorous", "define ephemeral")
     b) DuckDuckGo Instant Answer API for general facts & concepts
     c) Smart Wikipedia Search (`wikipedia.search` + summary) + "Tell me more" follow-ups
"""

import os
import re
import ast
import json
import math
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import platform
import random
import datetime
import webbrowser
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

try:
    import wikipedia
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class SafeMathEvaluator:
    """Evaluates mathematical expressions safely using Python's AST module."""

    OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": lambda x: math.sin(math.radians(x)),
        "cos": lambda x: math.cos(math.radians(x)),
        "tan": lambda x: math.tan(math.radians(x)),
        "log": math.log10,
        "ln": math.log,
        "abs": abs,
        "round": round,
        "factorial": math.factorial,
    }

    CONSTANTS = {"pi": math.pi, "e": math.e}

    @classmethod
    def evaluate(cls, expression: str):
        expr = expression.replace("^", "**").strip()
        node = ast.parse(expr, mode="eval").body
        return cls._eval_node(node)

    @classmethod
    def _eval_node(cls, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in cls.CONSTANTS:
            return cls.CONSTANTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in cls.OPERATORS:
            return cls.OPERATORS[type(node.op)](cls._eval_node(node.left), cls._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls.OPERATORS:
            return cls.OPERATORS[type(node.op)](cls._eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in cls.FUNCTIONS:
            args = [cls._eval_node(arg) for arg in node.args]
            return cls.FUNCTIONS[node.func.id](*args)
        raise ValueError("Unsupported mathematical operation")


class HybridVirtualAssistant:
    def __init__(self, intents_path="intents.json", state_path="assistant_state.json", use_voice_output=False):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.intents_path = os.path.join(self.base_dir, intents_path)
        self.state_path = os.path.join(self.base_dir, state_path)
        self.env_path = os.path.join(self.base_dir, ".env")
        self.use_voice_output = use_voice_output and TTS_AVAILABLE

        self.state = self._load_state()
        self.last_topic = None
        self.chat_history = []

        # Load GEMINI_API_KEY from environment or local .env file
        self.gemini_client = None
        self.gemini_key = self._read_env_key()
        if GENAI_SDK_AVAILABLE and self.gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
            except Exception:
                self.gemini_client = None

        # Optional offline TTS
        self.tts_engine = None
        if self.use_voice_output:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 175)
            except Exception:
                self.use_voice_output = False

        # Train Tier-1 Local Ensemble NLP Classifier (TF-IDF + Logistic Regression + Cosine Similarity)
        self.intents_data = self._load_intents()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.classifier = LogisticRegression(C=5.0, max_iter=500)
        self.patterns = []
        self.tags = []
        self._train_models()

    def _read_env_key(self) -> str:
        env_val = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_val:
            return env_val
        if os.path.exists(self.env_path):
            try:
                with open(self.env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
        return ""

    def set_api_key(self, api_key: str):
        """Validates the Gemini API key with a live test and saves it to `.env` so it persists across restarts."""
        key = api_key.strip()
        if not GENAI_SDK_AVAILABLE:
            return False, "SDK missing: run 'pip install google-genai' and restart app.py"
        if not key:
            return False, "Please paste a valid GEMINI_API_KEY."
        try:
            client = genai.Client(api_key=key)
            client.models.generate_content(model="gemini-2.5-flash", contents="Reply with OK")
            self.gemini_client = client
            self.gemini_key = key
            # Persist to .env so it stays active when the teacher runs the project
            with open(self.env_path, "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={key}\n")
            return True, "Activated & saved to .env!"
        except Exception as e:
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "400" in err_msg:
                return False, "Invalid API Key. Check your key at aistudio.google.com"
            return False, f"Activation error: {err_msg[:80]}"

    def get_brain_mode(self) -> str:
        if self.gemini_client:
            return "Hybrid (Local NLP + Gemini 2.5 Flash)"
        return "Hybrid (Local NLP + Dictionary/Wiki/Web QA)"

    def _load_intents(self):
        with open(self.intents_path, "r", encoding="utf-8") as f:
            return json.load(f)["intents"]

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"user_name": None, "todos": []}

    def _save_state(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\+\-\*/\^\(\)\.:]", "", text)
        return text

    def _train_models(self):
        for intent in self.intents_data:
            for pattern in intent["patterns"]:
                self.patterns.append(self._clean(pattern))
                self.tags.append(intent["tag"])
        self.X_tfidf = self.vectorizer.fit_transform(self.patterns)
        self.classifier.fit(self.X_tfidf, self.tags)

    def predict_intent(self, query: str, threshold: float = 0.45):
        """
        Tier-1 Router: routes deterministic utilities locally when confidence >= 0.45.
        Passes all open-ended general questions, definitions, and conversations to Tier-2.
        """
        cleaned = self._clean(query)

        if re.fullmatch(r"[\d\s\+\-\*/\(\)\.\^]+", cleaned) and any(c.isdigit() for c in cleaned):
            return "calculate", 1.0, "Rule-Math"
        if re.search(r"\bconvert\b.*\bto\b", cleaned):
            return "unit_convert", 0.98, "Rule-Convert"
        if cleaned.startswith("summarize:") or cleaned.startswith("summarize "):
            return "summarize", 0.96, "Rule-NLP"

        vec = self.vectorizer.transform([cleaned])

        probs = self.classifier.predict_proba(vec)[0]
        lr_idx = int(np.argmax(probs))
        lr_tag = self.classifier.classes_[lr_idx]
        lr_conf = float(probs[lr_idx])

        sims = cosine_similarity(vec, self.X_tfidf).flatten()
        cos_idx = int(np.argmax(sims))
        cos_tag = self.tags[cos_idx]
        cos_conf = float(sims[cos_idx])

        if lr_tag == cos_tag:
            final_tag = lr_tag
            final_conf = min(1.0, 0.5 * lr_conf + 0.5 * cos_conf + 0.15)
        elif cos_conf > 0.60:
            final_tag, final_conf = cos_tag, cos_conf
        else:
            final_tag, final_conf = lr_tag, lr_conf

        if final_conf < threshold:
            return "general_qa", final_conf, "Tier2-OpenDomain"
        return final_tag, final_conf, "Tier1-EnsembleNLP"

    # -------------------- TIER 1: LOCAL ACTION SKILLS --------------------

    def _get_static_response(self, tag: str) -> str:
        for intent in self.intents_data:
            if intent["tag"] == tag and intent["responses"]:
                return random.choice(intent["responses"])
        return "Done."

    def _handle_weather(self, query: str) -> str:
        match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", query, re.IGNORECASE)
        city = match.group(1).strip() if match else "Kota"
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            current = data["current_condition"][0]
            return (
                f"Weather in {city.title()}: {current['weatherDesc'][0]['value']}, "
                f"{current['temp_C']}°C (Feels like {current['FeelsLikeC']}°C), "
                f"Humidity: {current['humidity']}%."
            )
        except Exception:
            return f"Could not fetch live weather for '{city}' (check internet connection)."

    def _handle_news(self) -> str:
        url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")[:5]
            headlines = [f"  {i+1}. {item.find('title').text}" for i, item in enumerate(items)]
            return "Top 5 Live Headlines:\n" + "\n".join(headlines)
        except Exception:
            return "Unable to fetch live news headlines right now."

    def _handle_calculate(self, query: str) -> str:
        expr = query.lower()
        for word in ["calculate", "solve", "compute", "what is the value of", "what is", "evaluate", "math"]:
            expr = expr.replace(word, "")
        expr = expr.strip(" ?")
        try:
            result = SafeMathEvaluator.evaluate(expr)
            if isinstance(result, float):
                result = round(result, 6)
            return f"Result: {expr} = {result}"
        except Exception:
            return "Invalid math expression. Try: 'calculate 25 * (4 + 6)' or 'sqrt(144) + sin(30)'."

    def _handle_unit_convert(self, query: str) -> str:
        match = re.search(r"([-+]?\d*\.?\d+)\s*([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)", query.lower())
        if not match:
            return "Format: 'convert 10 km to miles' or 'convert 37 celsius to fahrenheit'."
        val = float(match.group(1))
        src, dst = match.group(2), match.group(3)
        conversions = {
            ("km", "miles"): val * 0.621371,
            ("miles", "km"): val * 1.60934,
            ("m", "feet"): val * 3.28084,
            ("meters", "feet"): val * 3.28084,
            ("feet", "m"): val * 0.3048,
            ("cm", "inches"): val * 0.393701,
            ("inches", "cm"): val * 2.54,
            ("kg", "pounds"): val * 2.20462,
            ("kg", "lbs"): val * 2.20462,
            ("pounds", "kg"): val * 0.453592,
            ("lbs", "kg"): val * 0.453592,
            ("celsius", "fahrenheit"): (val * 9 / 5) + 32,
            ("c", "f"): (val * 9 / 5) + 32,
            ("fahrenheit", "celsius"): (val - 32) * 5 / 9,
            ("f", "c"): (val - 32) * 5 / 9,
        }
        if (src, dst) in conversions:
            return f"{val} {src} = {conversions[(src, dst)]:.4f} {dst}"
        return f"Conversion from '{src}' to '{dst}' is not supported yet."

    def _handle_system_info(self) -> str:
        info = [
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
            f"Processor: {platform.processor() or 'Standard x86_64'}"
        ]
        if PSUTIL_AVAILABLE:
            cpu = psutil.cpu_percent(interval=0.3)
            mem = psutil.virtual_memory()
            info.append(f"CPU Usage: {cpu}%")
            info.append(f"RAM Usage: {mem.percent}% ({mem.used // (1024**2)} MB / {mem.total // (1024**2)} MB)")
            battery = psutil.sensors_battery()
            if battery:
                status = "Charging" if battery.power_plugged else "Discharging"
                info.append(f"Battery: {battery.percent}% ({status})")
        return "System Diagnostics:\n  " + "\n  ".join(info)

    def _handle_open_web(self, query: str) -> str:
        q = query.lower()
        sites = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://www.github.com",
            "stackoverflow": "https://stackoverflow.com",
            "leetcode": "https://leetcode.com"
        }
        if "search google for" in q:
            term = q.split("search google for", 1)[1].strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(term)}")
            return f"Opened Google Search for '{term}'."
        if "play on youtube" in q:
            term = q.split("play on youtube", 1)[1].strip()
            webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(term)}")
            return f"Searching YouTube for '{term}'."
        for name, url in sites.items():
            if name in q:
                webbrowser.open(url)
                return f"Launched {name.capitalize()} in browser."
        return "Try 'open github', 'search google for <topic>', or 'play on youtube <video>'."

    def _handle_todo_add(self, query: str) -> str:
        triggers = ["add task", "new todo", "remind me to", "add to my todo list", "schedule task", "create task", "put on my list"]
        task_text = query
        for t in triggers:
            if query.lower().startswith(t):
                task_text = query[len(t):].strip(" :-")
                break
        if not task_text:
            return "Please specify a task (e.g., 'Remind me to submit AI Lab file')."
        self.state["todos"].append({
            "task": task_text,
            "done": False,
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        self._save_state()
        return f"Added Task #{len(self.state['todos'])}: '{task_text}'"

    def _handle_todo_show(self) -> str:
        todos = self.state["todos"]
        if not todos:
            return "Your To-Do list is currently empty."
        lines = []
        for idx, item in enumerate(todos, 1):
            mark = "[✓]" if item["done"] else "[ ]"
            lines.append(f"  {idx}. {mark} {item['task']} ({item['created']})")
        return "Your To-Do List:\n" + "\n".join(lines)

    def _handle_todo_done(self, query: str) -> str:
        match = re.search(r"(\d+)", query)
        if not match:
            return "Specify the task number to mark complete (e.g., 'complete task 1')."
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(self.state["todos"]):
            self.state["todos"][idx]["done"] = True
            self._save_state()
            return f"Marked Task #{idx+1} ('{self.state['todos'][idx]['task']}') as completed!"
        return f"Task #{idx+1} does not exist."

    def _handle_summarize(self, query: str) -> str:
        text = re.sub(r"^(summarize this text|summarize paragraph|summarize|tldr)[:\s]*", "", query, flags=re.IGNORECASE).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        if len(sentences) < 2:
            return "Provide at least 2–3 sentences after 'summarize:' to generate an extractive summary."
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(sentences)
        scores = np.asarray(tfidf.sum(axis=1)).ravel()
        top_n = min(2, len(sentences))
        top_indices = sorted(np.argsort(scores)[-top_n:])
        summary = " ".join([sentences[i] for i in top_indices])
        return f"Extractive Summary:\n{summary}"

    # -------------------- TIER 2: GENERAL KNOWLEDGE, DICTIONARY & LLM --------------------

    def _extract_search_keywords(self, query: str) -> str:
        """Strips conversational & dictionary prefixes cleanly without syntax bugs."""
        cleaned = re.sub(
            r"^(who is|who was|what is|what are|what was|why is|why do|why does|how does|how do|how to|tell me about|explain|define|can you explain|give me information on)\s+",
            "",
            query.strip(),
            flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"^(the\s+)?(meaning|definition|synonym|synonyms)\s+(of|for)\s+",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE
        ).strip(" ?.!,'\"")
        return cleaned if cleaned else query.strip(" ?.")

    def _lookup_dictionary(self, word: str):
        """Fetches exact English definitions & examples from Free Dictionary API (no API key needed)."""
        if not re.fullmatch(r"[a-zA-Z\-]{2,35}", word):
            return None
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word.lower())}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            entry = data[0]
            meaning = entry["meanings"][0]
            pos = meaning.get("partOfSpeech", "")
            defn = meaning["definitions"][0]["definition"]
            example = meaning["definitions"][0].get("example", "")
            out = f"[Dictionary: {entry['word']} ({pos})]\nDefinition: {defn}"
            if example:
                out += f'\nExample: "{example}"'
            return out
        except Exception:
            return None

    def _handle_general_qa(self, query: str, sentences: int = 3):
        """
        Answers ANY general, everyday, dictionary, or academic question:
        1. Gemini 2.5 Flash LLM if activated (via .env or Web UI).
        2. Free Dictionary API for word meanings/definitions (e.g., "explain meaning of rigorous").
        3. DuckDuckGo Instant Answer API for general facts.
        4. Smart Wikipedia Search (`wikipedia.search` + summary).
        """
        # 1. Generative LLM Mode (Gemini 2.5 Flash)
        if self.gemini_client:
            try:
                recent_turns = self.chat_history[-4:]
                history_str = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in recent_turns])
                user_name_note = f"The user's name is {self.state['user_name']}." if self.state["user_name"] else ""
                prompt = (
                    f"You are a helpful, friendly, and intelligent AI Virtual Assistant. {user_name_note}\n"
                    f"Answer the user's question clearly, naturally, and concisely (3-6 sentences unless more detail is requested).\n"
                    f"Recent conversation:\n{history_str}\n\n"
                    f"User: {query}\nAssistant:"
                )
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip(), "gemini_llm", 0.98
            except Exception:
                pass

        keywords = self._extract_search_keywords(query)

        # 2. Dictionary Lookup for meanings / single-word definitions
        if re.search(r"\b(meaning|define|definition)\b", query, re.IGNORECASE) or len(keywords.split()) == 1:
            dict_ans = self._lookup_dictionary(keywords)
            if dict_ans:
                self.last_topic = keywords
                return dict_ans, "dictionary_api", 0.92

        # 3. DuckDuckGo Instant Answer API
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(keywords)}&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(ddg_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("AbstractText"):
                heading = data.get("Heading", keywords.title())
                self.last_topic = heading
                return f"[{heading}]\n{data['AbstractText']}", "duckduckgo_qa", 0.85
            if data.get("Answer"):
                return str(data["Answer"]), "duckduckgo_qa", 0.85
        except Exception:
            pass

        # 4. Smart Wikipedia Search (automatically finds closest article title)
        if WIKI_AVAILABLE:
            try:
                search_results = wikipedia.search(keywords, results=3)
                if search_results:
                    best_title = search_results[0]
                    self.last_topic = best_title
                    summary = wikipedia.summary(best_title, sentences=sentences, auto_suggest=False)
                    return f"[{best_title}]\n{summary}", "smart_wikipedia", 0.82
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    try:
                        self.last_topic = e.options[0]
                        summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
                        return f"[{e.options[0]}]\n{summary}", "smart_wikipedia", 0.78
                    except Exception:
                        pass
            except Exception:
                pass

        return (
            f"I couldn't find an instant offline answer for '{query}'. "
            f"Try 'search google for {keywords}' or activate Gemini 2.5 Flash in the sidebar for full generative AI!",
            "fallback",
            0.35
        )

    # -------------------- MAIN ROUTER --------------------

    def process_query(self, query: str):
        start_time = time.perf_counter()
        if not query or not query.strip():
            return {"response": "Please enter a message.", "intent": "empty", "confidence": 0.0, "latency_ms": 0, "exit": False}

        q_lower = query.lower().strip()

        # Context follow-up ("tell me more")
        if q_lower in ("tell me more", "more details", "elaborate", "explain more") and self.last_topic:
            resp, sub_intent, conf = self._handle_general_qa(f"Explain {self.last_topic} in more detail", sentences=6)
            latency = round((time.perf_counter() - start_time) * 1000, 1)
            self.chat_history.append((query, resp))
            return {"response": resp, "intent": f"followup({sub_intent})", "confidence": conf, "latency_ms": latency, "exit": False}

        # Name memory slot
        name_match = re.search(r"(?:my name is|i am|call me)\s+([a-zA-Z]+)", query, re.IGNORECASE)
        if name_match and not any(w in q_lower for w in ("what", "who", "how", "why")):
            self.state["user_name"] = name_match.group(1).capitalize()
            self._save_state()
            resp = f"Nice to meet you, {self.state['user_name']}! I'll remember your name."
            latency = round((time.perf_counter() - start_time) * 1000, 1)
            self.chat_history.append((query, resp))
            return {"response": resp, "intent": "memory_store", "confidence": 1.0, "latency_ms": latency, "exit": False}

        intent, confidence, model_used = self.predict_intent(query)
        should_exit = False

        if intent == "greeting":
            resp = self._get_static_response("greeting")
            if self.state["user_name"]:
                resp = resp.replace("!", f", {self.state['user_name']}!")
        elif intent in ("identity", "joke", "thanks"):
            resp = self._get_static_response(intent)
        elif intent == "time":
            resp = f"Current Time: {datetime.datetime.now().strftime('%I:%M:%S %p')}"
        elif intent == "date":
            resp = f"Today's Date: {datetime.datetime.now().strftime('%A, %B %d, %Y')}"
        elif intent == "weather":
            resp = self._handle_weather(query)
        elif intent == "news":
            resp = self._handle_news()
        elif intent == "calculate":
            resp = self._handle_calculate(query)
        elif intent == "unit_convert":
            resp = self._handle_unit_convert(query)
        elif intent == "system_info":
            resp = self._handle_system_info()
        elif intent == "open_web":
            resp = self._handle_open_web(query)
        elif intent == "todo_add":
            resp = self._handle_todo_add(query)
        elif intent == "todo_show":
            resp = self._handle_todo_show()
        elif intent == "todo_done":
            resp = self._handle_todo_done(query)
        elif intent == "summarize":
            resp = self._handle_summarize(query)
        elif intent == "exit":
            resp = self._get_static_response("exit")
            should_exit = True
        else:
            resp, sub_intent, conf = self._handle_general_qa(query)
            intent = sub_intent
            confidence = conf

        latency = round((time.perf_counter() - start_time) * 1000, 1)
        self.chat_history.append((query, resp))
        return {
            "response": resp,
            "intent": intent,
            "confidence": round(confidence, 3),
            "model": model_used,
            "brain_mode": self.get_brain_mode(),
            "latency_ms": latency,
            "exit": should_exit
        }

    def speak(self, text: str):
        print(f"\n[Assistant]: {text}")
        if self.use_voice_output and self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def run_cli(self):
        print("=" * 72)
        print(f"  HYBRID VIRTUAL ASSISTANT — Mode: {self.get_brain_mode()}")
        print("=" * 72)
        self.speak("Assistant ready! Ask any general question or run a utility command. Type 'exit' to quit.")
        while True:
            query = input("\n[You]: ").strip()
            out = self.process_query(query)
            meta = f"[Route: {out['intent']} | Conf: {out['confidence']} | {out['latency_ms']}ms]"
            self.speak(f"{out['response']}\n{meta}")
            if out["exit"]:
                break


if __name__ == "__main__":
    bot = HybridVirtualAssistant(use_voice_output=False)
    bot.run_cli()
