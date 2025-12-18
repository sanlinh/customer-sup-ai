import json
import hashlib
from typing import Dict, Any, List, Tuple

def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def split_holdout(example: Dict[str, Any], holdout_ratio: float = 0.2) -> str:
    key = (example.get("scenario") or "") + "|" + (example.get("clientSequence") or "")
    bucket = stable_hash(key) % 1000
    return "holdout" if bucket < int(1000 * holdout_ratio) else "train"

JUDGE_PROMPT = """
You are grading whether an AI reply matches a real immigration consultant's style and usefulness.

Return JSON only:
{
  "scores": {
    "human_tone": 0-5,
    "correctness": 0-5,
    "completeness": 0-5,
    "next_step_question_quality": 0-5,
    "not_ai_sounding": 0-5
  },
  "notes": "short reason"
}
"""

def judge_reply(groq_generate, extract_json_object, client_sequence: str, chat_history, real_reply: str, predicted_reply: str) -> Dict[str, Any]:
    payload = {
        "clientSequence": client_sequence,
        "chatHistory": chat_history,
        "realConsultantReply": real_reply,
        "predictedAiReply": predicted_reply
    }
    raw = groq_generate(
        f"{JUDGE_PROMPT}\n\nINPUT:\n{json.dumps(payload, ensure_ascii=False)}",
        max_tokens=512,
        temperature=0.0
    )
    obj = extract_json_object(raw) or {}
    scores = (obj.get("scores") or {})
    # Safe defaults
    def clamp(x):
        try:
            x = float(x)
        except Exception:
            x = 0.0
        return max(0.0, min(5.0, x))

    scores = {k: clamp(v) for k, v in scores.items()}
    total = sum(scores.values())
    return {"scores": scores, "total": total, "notes": obj.get("notes", "")}

def mean(nums: List[float]) -> float:
    return sum(nums) / max(1, len(nums))
