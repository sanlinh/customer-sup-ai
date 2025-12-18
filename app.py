from flask import Flask, request, jsonify
from prompt_builder import build_system_prompt
from dataset_builder import load_conversations, build_training_examples
from eval_and_rag import judge_reply, split_holdout, mean
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
import json
import random
import os
import re

load_dotenv()

app = Flask(__name__)

# ===== Prompts (define FIRST) =====

DEFAULT_PROMPT = build_system_prompt().strip() + '\n\nReturn JSON only: {"reply":"..."}'

EDITOR_PROMPT = """
You are a prompt editor.
Improve the chatbot prompt so predicted replies match real consultant replies better.

Inputs:
- existing_prompt
- client_sequence
- chat_history
- real_consultant_reply
- predicted_ai_reply

Make small surgical edits only.
Return JSON only: {"prompt":"..."}
""".strip()

MANUAL_EDITOR_PROMPT = """
You are a prompt editor.
Apply the user's instructions to the existing prompt with minimal changes.
Keep the prompt short and consistent.

Return JSON only: {"prompt":"..."}
""".strip()

# ===== Env + Clients =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
PROMPT_NAME = os.getenv("PROMPT_NAME", "visa_assistant_v1")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_API_KEY in .env")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== Groq Setup =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TRAIN_MODEL = os.getenv("GROQ_TRAIN_MODEL", GROQ_MODEL)


groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

def groq_generate(prompt_text: str, max_tokens: int = 1024, temperature: float = 0.2, model: str = None) -> str:
    resp = groq_client.chat.completions.create(
        model=model or GROQ_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

def extract_json_object(text: str):
    """
    Handles cases like:
    - pure JSON: {"reply":"..."}
    - fenced JSON: ```json {...} ```
    - extra text around JSON
    """
    if not text:
        return None

    # strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: find first {...} block
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    return None

# ===== Supabase prompt helpers =====
def get_prompt():
    res = sb.table("prompts").select("prompt").eq("name", PROMPT_NAME).limit(1).execute()
    rows = res.data or []
    if not rows:
        sb.table("prompts").insert({"name": PROMPT_NAME, "prompt": DEFAULT_PROMPT}).execute()
        sb.table("prompt_versions").insert({
            "prompt_name": PROMPT_NAME,
            "prompt": DEFAULT_PROMPT,
            "source": "boot"
        }).execute()
        return DEFAULT_PROMPT
    return rows[0]["prompt"]

def set_prompt(p, source="auto"):
    # 1) upsert live prompt
    res1 = sb.table("prompts").upsert(
        {"name": PROMPT_NAME, "prompt": p},
        on_conflict="name"
    ).execute()

    # 2) insert version history row
    res2 = sb.table("prompt_versions").insert({
        "prompt_name": PROMPT_NAME,
        "prompt": p,
        "source": source
    }).execute()

    # 3) surface errors clearly
    err1 = getattr(res1, "error", None)
    err2 = getattr(res2, "error", None)

    if err1:
        raise RuntimeError(f"Supabase prompts upsert failed: {err1}")
    if err2:
        raise RuntimeError(f"Supabase prompt_versions insert failed: {err2}")

    print("✅ prompts upsert:", res1.data)
    print("✅ prompt_versions insert:", res2.data)

    return p

def _history_to_text(chat_history):
    history_text = ""
    for msg in chat_history or []:
        role = (msg.get("role") or "client").upper()
        message = (msg.get("message") or "").strip()
        if message:
            history_text += f"{role}: {message}\n"
    return history_text

def predict_reply_with_prompt(prompt_text: str, client_sequence: str, chat_history, model: str, max_tokens: int):
    history_text = _history_to_text(chat_history)
    full_prompt = f"""
{prompt_text}

CHAT HISTORY:
{history_text}

CLIENT MESSAGE:
{client_sequence}

Return JSON only: {{"reply":"..."}}
""".strip()

    raw = groq_generate(full_prompt, max_tokens=max_tokens, temperature=0.2, model=model)
    parsed = extract_json_object(raw) or {}
    reply = (parsed.get("reply") or "").strip() if isinstance(parsed, dict) else ""
    return reply or raw.strip()

def quick_holdout_score(prompt_text: str, holdout_samples: list, model: str, pred_tokens: int):
    totals = []
    for ex in holdout_samples:
        predicted_reply = predict_reply_with_prompt(
            prompt_text, ex["clientSequence"], ex["chatHistory"], model=model, max_tokens=pred_tokens
        )
        judged = judge_reply(
            groq_generate(prompt_text, max_tokens=512, temperature=0.0, model=GROQ_TRAIN_MODEL), extract_json_object,
            ex["clientSequence"], ex["chatHistory"],
            ex["consultantReply"], predicted_reply
        )
        totals.append(judged["total"])
    return mean(totals)

RAG_CACHE = {"built": False, "examples": []}
STOPWORDS = set("a an the and or but if then so to of in on at for with from as is are was were be been being".split())

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str):
    return {t for t in _norm(s).split() if t and t not in STOPWORDS and len(t) > 2}

def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def build_rag_cache():
    conv = load_conversations("conversations.json")
    exs = build_training_examples(conv)
    packed = []
    for i, ex in enumerate(exs):
        hist = ex.get("chatHistory") or []
        hist_text = " ".join([(m.get("role","") + ": " + (m.get("message") or "")) for m in hist[-6:]])
        query_text = f"{ex.get('clientSequence','')} {hist_text}"
        packed.append({"id": i, "query_text": query_text, "reply": ex.get("consultantReply","")})
    RAG_CACHE["examples"] = packed
    RAG_CACHE["built"] = True

def retrieve_rag(client_sequence: str, chat_history, k: int = 3):
    if not RAG_CACHE["built"]:
        build_rag_cache()
    hist_text = " ".join([(m.get("role","") + ": " + (m.get("message") or "")) for m in (chat_history or [])[-6:]])
    q = f"{client_sequence} {hist_text}"

    scored = []
    for ex in RAG_CACHE["examples"]:
        s = _jaccard(q, ex["query_text"])
        if s > 0:
            scored.append((s, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    return [{"id": ex["id"], "score": float(s), "reply": ex["reply"][:400]} for s, ex in top]


@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/debug-db")
def debug_db():
    r1 = sb.table("prompts").select("name, updated_at").order("updated_at", desc=True).limit(5).execute()
    r2 = sb.table("prompt_versions").select("*").limit(5).execute()

    return jsonify({
        "SUPABASE_URL": SUPABASE_URL,
        "PROMPT_NAME": PROMPT_NAME,
        "prompts": r1.data,
        "prompts_error": str(getattr(r1, "error", None)),
        "prompt_versions": r2.data,
        "prompt_versions_error": str(getattr(r2, "error", None)),
    })

@app.route("/generate-reply", methods=["POST"])
def generate_reply():
    data = request.get_json(silent=True) or {}

    client_sequence = (data.get("clientSequence") or "").strip()
    chat_history = data.get("chatHistory") or []

    if not client_sequence:
        return jsonify({"error": "clientSequence is required"}), 400
    
    debug = (request.args.get("debug") == "true")
    rag_k = int(request.args.get("ragK", 3))

    retrieved = retrieve_rag(client_sequence, chat_history, k=rag_k)

    history_text = ""
    for msg in chat_history:
        role = (msg.get("role") or "client").upper()
        message = (msg.get("message") or "").strip()
        if message:
            history_text += f"{role}: {message}\n"

    live_prompt = get_prompt()

    examples_text = ""
    if retrieved:
        examples_text = "REFERENCE EXAMPLES (style only, do not copy verbatim):\n"
        for ex in retrieved:
            examples_text += f"- ex#{ex['id']} sim={ex['score']:.2f}: {ex['reply']}\n"

    prompt = f"""
{live_prompt}

{examples_text}

CHAT HISTORY:
{history_text}

CLIENT MESSAGE:
{client_sequence}

Return JSON only: {{"reply":"..."}}
""".strip()

    raw = groq_generate(prompt, max_tokens=1024, temperature=0.2)

    parsed = extract_json_object(raw) or {}
    reply = (parsed.get("reply") or "").strip() if isinstance(parsed, dict) else ""

    if not reply:
        # fallback to raw
        reply = raw.strip()

    if not reply:
        reply = "Got it — quick question so I can guide you properly: what’s your nationality and where are you applying from?"

    resp = {"aiReply": reply}
    if debug:
        resp["retrievedExamples"] = retrieved
    return jsonify(resp)

@app.route("/improve-ai", methods=["POST"])
def improve_ai():
    data = request.get_json(silent=True) or {}
    client_sequence = (data.get("clientSequence") or "").strip()
    chat_history = data.get("chatHistory") or []
    consultant_reply = (data.get("consultantReply") or "").strip()

    if not client_sequence or not consultant_reply:
        return jsonify({"error": "clientSequence and consultantReply are required"}), 400

    current_prompt = get_prompt()

    predicted_text = groq_generate(f"""
{current_prompt}

INPUT:
{json.dumps({"clientSequence": client_sequence, "chatHistory": chat_history}, ensure_ascii=False)}
""".strip(), max_tokens=1024, temperature=0.2)

    editor_input = {
        "existing_prompt": current_prompt,
        "client_sequence": client_sequence,
        "chat_history": chat_history,
        "real_consultant_reply": consultant_reply,
        "predicted_ai_reply": predicted_text,
    }

    edited_raw = groq_generate(f"""
{EDITOR_PROMPT}

INPUT:
{json.dumps(editor_input, ensure_ascii=False)}
""".strip(), max_tokens=1400, temperature=0.0)

    parsed = extract_json_object(edited_raw) or {}
    updated_prompt = (parsed.get("prompt") or "").strip() if isinstance(parsed, dict) else ""

    if updated_prompt:
        set_prompt(updated_prompt, source="auto")
    else:
        updated_prompt = current_prompt

    return jsonify({"predictedReply": predicted_text, "updatedPrompt": updated_prompt})

@app.route("/improve-ai-manually", methods=["POST"])
def improve_ai_manually():
    data = request.get_json(silent=True) or {}
    instructions = (data.get("instructions") or "").strip()

    if not instructions:
        return jsonify({"error": "instructions is required"}), 400

    current_prompt = get_prompt()

    editor_input = {
        "existing_prompt": current_prompt,
        "instructions": instructions
    }

    raw = groq_generate(f"""
{MANUAL_EDITOR_PROMPT}

INPUT:
{json.dumps(editor_input, ensure_ascii=False)}
""".strip(), max_tokens=1200, temperature=0.0)

    parsed = extract_json_object(raw) or {}
    updated_prompt = (parsed.get("prompt") or "").strip() if isinstance(parsed, dict) else ""

    if updated_prompt:
        set_prompt(updated_prompt, source="manual")
    else:
        updated_prompt = current_prompt

    return jsonify({"updatedPrompt": updated_prompt})

# @app.route("/train-from-history", methods=["POST"])
# def train_from_history():
#     data = request.get_json(silent=True) or {}
#     max_samples = int(data.get("maxSamples", 20))
#     seed = int(data.get("seed", 42))

#     conv_data = load_conversations("conversations.json")
#     examples = build_training_examples(conv_data)

#     if not examples:
#         return jsonify({"error": "No training examples produced from conversations.json"}), 500

#     rnd = random.Random(seed)
#     rnd.shuffle(examples)
#     chosen = examples[:max_samples]

#     updates = 0
#     failed = 0

#     for ex in chosen:
#         client_sequence = ex["clientSequence"]
#         chat_history = ex["chatHistory"]
#         consultant_reply = ex["consultantReply"]

#         current_prompt = get_prompt()

#         predicted_text = groq_generate(f"""
# {current_prompt}

# INPUT:
# {json.dumps({"clientSequence": client_sequence, "chatHistory": chat_history}, ensure_ascii=False)}
# """.strip(), max_tokens=1024, temperature=0.2)

#         editor_input = {
#             "existing_prompt": current_prompt,
#             "client_sequence": client_sequence,
#             "chat_history": chat_history,
#             "real_consultant_reply": consultant_reply,
#             "predicted_ai_reply": predicted_text,
#         }

#         edited_raw = groq_generate(f"""
# {EDITOR_PROMPT}

# INPUT:
# {json.dumps(editor_input, ensure_ascii=False)}
# """.strip(), max_tokens=1400, temperature=0.0)

#         parsed = extract_json_object(edited_raw) or {}
#         updated_prompt = (parsed.get("prompt") or "").strip() if isinstance(parsed, dict) else ""

#         if updated_prompt:
#             set_prompt(updated_prompt, source="auto")
#             updates += 1
#         else:
#             failed += 1

#     return jsonify({
#         "samplesUsed": len(chosen),
#         "promptUpdates": updates,
#         "failedUpdates": failed,
#     })

@app.route("/train-from-history", methods=["POST"])
def train_from_history():
    import time

    data = request.get_json(silent=True) or {}

    max_samples = int(data.get("maxSamples", 20))
    seed = int(data.get("seed", 42))
    return_preview = bool(data.get("returnUpdatedPromptPreview", False))
    gate_k = int(data.get("gateK", 5))
    min_delta = float(data.get("minDelta", 0.2))



    # ===== Safety + cost controls =====
    dry_run = bool(data.get("dryRun", False))              # true = no Supabase writes
    max_updates = int(data.get("maxUpdates", 1))           # stop after N accepted updates
    max_seconds = int(data.get("maxSeconds", 20))          # stop after N seconds

    # Token limits (cheap defaults)
    pred_tokens = int(data.get("predTokens", 220))
    edit_tokens = int(data.get("editTokens", 450))

    # Use cheaper model for training if you set GROQ_TRAIN_MODEL
    train_model = os.getenv("GROQ_TRAIN_MODEL", GROQ_MODEL)

    t0 = time.time()

    conv_data = load_conversations("conversations.json")
    examples = build_training_examples(conv_data)

    if not examples:
        return jsonify({"error": "No training examples produced from conversations.json"}), 500
    
    holdout = [ex for ex in examples if split_holdout(ex) == "holdout"]
    if len(holdout) < gate_k:
        holdout = examples[:]  # fallback if small dataset

    rnd_hold = random.Random(seed + 999)
    rnd_hold.shuffle(holdout)
    holdout_small = holdout[:min(gate_k, len(holdout))]

    rnd = random.Random(seed)
    rnd.shuffle(examples)
    chosen = examples[:max_samples]

    updates = 0
    rejected = 0
    failed = 0
    stopped_reason = None
    last_updated_prompt = None
    last_gate_before = None
    last_gate_after = None
    last_gate_delta = None


    for ex in chosen:
        # hard stops (prevents burning free tier)
        if updates >= max_updates:
            stopped_reason = "maxUpdates reached"
            break
        if (time.time() - t0) > max_seconds:
            stopped_reason = "maxSeconds reached"
            break

        client_sequence = ex["clientSequence"]
        chat_history = ex["chatHistory"]
        consultant_reply = ex["consultantReply"]

        current_prompt = get_prompt()

        # 1) predicted reply (short)
        predicted_text = predict_reply_with_prompt(
        current_prompt, client_sequence, chat_history, model=train_model, max_tokens=pred_tokens
        )

        editor_input = {
            "existing_prompt": current_prompt,
            "client_sequence": client_sequence,
            "chat_history": chat_history,
            "real_consultant_reply": consultant_reply,
            "predicted_ai_reply": predicted_text,
        }

        # 2) editor (short)
        edited_raw = groq_generate(
            f"""{EDITOR_PROMPT}

INPUT:
{json.dumps(editor_input, ensure_ascii=False)}
""".strip(),
            max_tokens=edit_tokens,
            temperature=0.0,
            model=train_model
        )

        parsed = extract_json_object(edited_raw) or {}
        updated_prompt = (parsed.get("prompt") or "").strip() if isinstance(parsed, dict) else ""

        if updated_prompt:
            before = quick_holdout_score(current_prompt, holdout_small, model=train_model, pred_tokens=pred_tokens)
            after  = quick_holdout_score(updated_prompt, holdout_small, model=train_model, pred_tokens=pred_tokens)

            last_gate_before = before
            last_gate_after = after
            last_gate_delta = after - before

            if after >= before + min_delta:
                last_updated_prompt = updated_prompt
                if not dry_run:
                    set_prompt(updated_prompt, source="auto")
                updates += 1
            else:
                rejected += 1
        else:
            failed += 1

    

    resp = {
    "samplesUsed": len(chosen),
    "promptUpdates": updates,
    "failedUpdates": failed,
    "dryRun": dry_run,
    "maxUpdates": max_updates,
    "maxSeconds": max_seconds,
    "predTokens": pred_tokens,
    "editTokens": edit_tokens,
    "trainModel": train_model,
    "stoppedReason": stopped_reason,
    "promptUpdatesRejected": rejected,
    "gateK": len(holdout_small),
    "minDelta": min_delta,
    "lastGateBefore": last_gate_before,
    "lastGateAfter": last_gate_after,
    "lastGateDelta": last_gate_delta
}

    if return_preview and last_updated_prompt:
        resp["updatedPromptPreview"] = last_updated_prompt[:2000]  # cap so response isn't huge

    return jsonify(resp)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 40))
    split = (data.get("split") or "holdout").strip()
    notes = (data.get("notes") or "").strip()

    conv = load_conversations("conversations.json")
    examples = build_training_examples(conv)

    # filter split
    chosen = [ex for ex in examples if split_holdout(ex) == split]
    random.shuffle(chosen)
    chosen = chosen[:max(1, min(n, len(chosen)))]

    prompt = get_prompt()

    # Create run
    run_res = sb.table("eval_runs").insert({
        "prompt_name": PROMPT_NAME,
        "prompt": prompt,
        "n_samples": len(chosen),
        "split": split,
        "notes": notes
    }).execute()
    run_id = (run_res.data or [{}])[0].get("id")

    totals = []
    for ex in chosen:
        # Call your own generate route logic directly (reuse the same prompt format you already use)
        client_sequence = ex["clientSequence"]
        chat_history = ex["chatHistory"]
        real_reply = ex["consultantReply"]

        # You already build the prompt in /generate-reply; easiest is to just call groq_generate similarly:
        prompt_text = f"""{prompt}

CLIENT SEQUENCE:
{client_sequence}

CHAT HISTORY:
{json.dumps(chat_history, ensure_ascii=False)}

Return JSON only: {{"reply":"..."}}
""".strip()

        predicted_raw = groq_generate(prompt_text, max_tokens=512, temperature=0.2)
        predicted_obj = extract_json_object(predicted_raw) or {}
        predicted_reply = (predicted_obj.get("reply") or predicted_raw).strip()

        judged = judge_reply(groq_generate, extract_json_object, client_sequence, chat_history, real_reply, predicted_reply)
        totals.append(judged["total"])

        sb.table("eval_samples").insert({
            "run_id": run_id,
            "scenario": ex.get("scenario"),
            "client_sequence": client_sequence,
            "chat_history": chat_history,
            "consultant_reply": real_reply,
            "predicted_reply": predicted_reply,
            "scores": judged["scores"],
            "total_score": judged["total"]
        }).execute()

    avg = mean(totals)
    sb.table("eval_runs").update({"avg_score": avg}).eq("id", run_id).execute()

    return jsonify({
        "runId": run_id,
        "split": split,
        "n": len(chosen),
        "avgScore": avg
    })

@app.get("/eval-runs")
def list_eval_runs():
    r = sb.table("eval_runs").select("*").order("id", desc=True).limit(20).execute()
    return jsonify({"evalRuns": r.data or []})

@app.get("/eval-runs/<int:run_id>")
def get_eval_run(run_id: int):
    run = sb.table("eval_runs").select("*").eq("id", run_id).limit(1).execute()
    samples = sb.table("eval_samples").select("*").eq("run_id", run_id).order("id", desc=False).limit(200).execute()
    return jsonify({"run": (run.data or [None])[0], "samples": samples.data or []})

@app.get("/prompt-diff/latest")
def prompt_diff_latest():
    r = sb.table("prompt_versions").select("id,prompt").eq("prompt_name", PROMPT_NAME).order("id", desc=True).limit(2).execute()
    rows = r.data or []
    if len(rows) < 2:
        return jsonify({"error": "Need at least 2 prompt versions"}), 400
    import difflib
    new = rows[0]["prompt"].splitlines()
    old = rows[1]["prompt"].splitlines()
    diff = "\n".join(difflib.unified_diff(old, new, fromfile=f"v{rows[1]['id']}", tofile=f"v{rows[0]['id']}", lineterm=""))
    return jsonify({"fromId": rows[1]["id"], "toId": rows[0]["id"], "diff": diff})



if __name__ == "__main__":
    app.run(debug=True)
