# Self-Learning Visa AI Assistant 🧭

A backend-only **self-learning AI assistant** for visa customer support, built for the **Issa Compass Vibe Hackathon**.

This service generates human-like consultant replies, learns from real historical conversations, and **only improves itself when performance measurably increases**.

---

## 🚀 Live Demo

**Hosted API:**

```
https://customer-sup-ai-production.up.railway.app
```

**Health Check:**

```bash
curl https://customer-sup-ai-production.up.railway.app/health
```

---

## 🧠 What This System Does

### 1. Human-like Replies (Not “AI-sounding”)

* Generates responses based on **real consultant conversations**
* Uses **retrieval-augmented prompting (RAG-style)** to ground tone and structure
* Avoids legal guarantees, stays concise, friendly, and calm

### 2. Self-Learning (Safely)

* Learns from historical conversations (`conversations.json`)
* Automatically proposes prompt improvements
* **Applies updates only if holdout performance improves**
* Prevents prompt drift via gated learning

### 3. Fully Measurable

* Built-in evaluation framework
* Prompt versioning + diffs
* Reproducible improvements via cURL (no UI required)

---

## 🏗️ Architecture Overview

```
Client / Tester
   |
   |  HTTP (cURL / API)
   v
Flask API (Railway)
   |
   ├─ /generate-reply        → AI reply (with RAG grounding)
   ├─ /train-from-history    → Gated self-learning
   ├─ /evaluate              → Quantitative scoring
   ├─ /prompt-diff/latest    → Show how the AI changed
   |
   v
Groq LLMs (Generation / Training / Judge)
   |
Supabase
   ├─ prompts
   ├─ prompt_versions
   ├─ eval_runs
   └─ eval_samples
```

---

## 📂 Project Structure

```
.
├─ app.py                # Flask API (core logic)
├─ dataset_builder.py    # Build training examples from conversations
├─ prompt_builder.py     # Base system prompt
├─ eval_and_rag.py       # Evaluation + scoring logic
├─ conversations.json    # Sample real conversations
├─ requirements.txt
├─ Procfile              # Railway startup (gunicorn)
├─ .env.example
└─ README.md
```

---

## 🔑 Environment Variables

Create a `.env` file locally, or set these in Railway **Variables**:

```env
SUPABASE_URL=
SUPABASE_API_KEY=

GROQ_API_KEY=

PROMPT_NAME=visa_assistant_v1
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TRAIN_MODEL=llama-3.1-8b-instant
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

API will run at:

```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints & Examples

### 1️⃣ Health Check

```bash
curl https://customer-sup-ai-production.up.railway.app/health
```

---

### 2️⃣ Generate AI Reply (with RAG + debug)

```bash
curl -X POST "https://customer-sup-ai-production.up.railway.app/generate-reply?debug=true&ragK=3" \
  -H "Content-Type: application/json" \
  -d '{
    "clientSequence": "I am American and currently in Bali. Can I apply from Indonesia?",
    "chatHistory": []
  }'
```

**Returns:**

* `aiReply`
* retrieved historical examples (debug mode)

---

### 3️⃣ Evaluate Current Prompt (Holdout Set)

```bash
curl -X POST https://customer-sup-ai-production.up.railway.app/evaluate \
  -H "Content-Type: application/json" \
  -d '{ "n": 25, "split": "holdout", "notes": "baseline" }'
```

---

### 4️⃣ Gated Self-Learning (Safe Auto-Improve)

```bash
curl -X POST https://customer-sup-ai-production.up.railway.app/train-from-history \
  -H "Content-Type: application/json" \
  -d '{
    "maxSamples": 6,
    "maxUpdates": 1,
    "gateK": 5,
    "minDelta": 0.2,
    "maxSeconds": 25
  }'
```

✔️ Prompt is updated **only if performance improves**.

---

### 5️⃣ View Prompt Diff (Transparency)

```bash
curl https://customer-sup-ai-production.up.railway.app/prompt-diff/latest
```

Shows exactly **what changed and why**.

---

### 6️⃣ Re-Evaluate After Training

```bash
curl -X POST https://customer-sup-ai-production.up.railway.app/evaluate \
  -H "Content-Type: application/json" \
  -d '{ "n": 25, "split": "holdout", "notes": "after training" }'
```

---

## 🧪 Why This Is “Self-Learning” (Not Just Prompt Editing)

* Uses **labeled historical data**
* Proposes prompt edits via a dedicated editor prompt
* Runs **quantitative evaluation**
* Applies updates **only if metrics improve**
* Stores every version + diff for auditability

This mirrors real-world **ML system iteration**, not just prompt tweaking.

---

## 🧑‍💻 Tech Stack

* **Backend:** Python, Flask
* **LLMs:** Groq (LLaMA family)
* **Database:** Supabase (Postgres)
* **Deployment:** Railway + Gunicorn
* **Evaluation:** LLM-as-judge + holdout splits

---
