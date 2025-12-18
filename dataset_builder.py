import json
import random
from typing import List, Dict, Any

NON_TEXT = {"[Non-text message]", ""}

def load_conversations(path: str = "conversations.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def msg_to_role(direction: str) -> str:
    return "client" if direction == "in" else "consultant"

def build_training_examples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    for record in data:
        conv = record.get("conversation", [])
        history: List[Dict[str, str]] = []
        i = 0

        while i < len(conv):
            # 1) gather consecutive client "in"
            if conv[i].get("direction") != "in":
                # still add to history
                txt = (conv[i].get("text") or "").strip()
                if txt not in NON_TEXT:
                    history.append({"role": msg_to_role(conv[i].get("direction")), "message": txt})
                i += 1
                continue

            client_msgs = []
            while i < len(conv) and conv[i].get("direction") == "in":
                txt = (conv[i].get("text") or "").strip()
                if txt not in NON_TEXT:
                    client_msgs.append(txt)
                i += 1

            # 2) gather the following consecutive consultant "out"
            out_start = i
            consultant_msgs = []
            while i < len(conv) and conv[i].get("direction") == "out":
                txt = (conv[i].get("text") or "").strip()
                if txt not in NON_TEXT:
                    consultant_msgs.append(txt)
                i += 1

            # if there was no consultant reply, skip this pair
            if not client_msgs or not consultant_msgs:
                # still push client msgs into history so next turns have context
                for t in client_msgs:
                    history.append({"role": "client", "message": t})
                # (don’t add consultant since none)
                continue

            examples.append({
                "contact_id": record.get("contact_id"),
                "scenario": record.get("scenario"),
                "chatHistory": history.copy(),
                "clientSequence": "\n".join(client_msgs),
                "consultantReply": "\n".join(consultant_msgs),
            })

            # 3) update history with what actually happened
            for t in client_msgs:
                history.append({"role": "client", "message": t})
            for t in consultant_msgs:
                history.append({"role": "consultant", "message": t})

            # (optional) keep going from where we left off

    return examples

if __name__ == "__main__":
    data = load_conversations()
    ex = build_training_examples(data)
    print("Total examples:", len(ex))
    for sample in random.sample(ex, k=min(3, len(ex))):
        print("\n--- SAMPLE ---")
        print("Scenario:", sample.get("scenario"))
        print("ClientSequence:\n", sample["clientSequence"])
        print("ConsultantReply:\n", sample["consultantReply"])
        print("History turns:", len(sample["chatHistory"]))
