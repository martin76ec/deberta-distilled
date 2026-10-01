import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "martin76ec/deberta-infrastructure-triage-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def predict_jev_answer(state: dict, question_spec: dict) -> dict:
    premise = f"Incident State: {json.dumps(state)}"
    q_type = question_spec["type"]
    inst = question_spec["instructions"]

    if q_type == "choice":
        options = list(question_spec["criteria"].keys())
        hyps = [
            f"Question: {inst} Option: {k} ({question_spec['criteria'][k]})"
            for k in options
        ]
    elif q_type == "noul":
        options = ["true", "false"]
        hyps = [f"Proposition: {inst} Truth: {b}" for b in options]
    elif q_type == "score":
        options = question_spec["criteria"]
        hyps = [f"Score Scale: {inst} Level: {lvl}" for lvl in options]

    inputs = tokenizer(
        [premise] * len(options), hyps, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        pos_logits = model(**inputs).logits[:, 1]
        probs = torch.softmax(pos_logits, dim=0).tolist()

    top_idx = int(torch.tensor(probs).argmax().item())
    prob_dict = {str(opt): round(p, 4) for opt, p in zip(options, probs)}

    if q_type == "choice":
        return {
            "type": "choice",
            "choice": options[top_idx],
            "confidence": round(probs[top_idx], 2),
            "probabilities": prob_dict,
        }
    elif q_type == "noul":
        return {
            "type": "noul",
            "noul": options[top_idx] == "true",
            "confidence": round(probs[top_idx], 2),
            "probabilities": prob_dict,
        }
    elif q_type == "score":
        return {
            "type": "score",
            "score": options[top_idx],
            "confidence": round(probs[top_idx], 2),
            "probabilities": prob_dict,
        }


state = {
    "incident_id": "INC-DB-20000",
    "alert_name": "AuroraPostgresConnectionSaturation",
    "environment": "production",
    "database_cluster": "aurora-production-primary",
    "pool_metrics": {
        "active_connections": 3882,
        "max_connections_limit": 4000,
        "saturation_pct": 97.0,
    },
}

question_spec = {
    "type": "choice",
    "instructions": "Determine which engineering team should own the escalation.",
    "criteria": {
        "orders_application_team": "Application query deadlock or unindexed join",
        "database_reliability_engineering": "Deep database engine contention or pool failures",
        "traffic_infrastructure": "Ingress overload requiring rate limiting",
        "cloud_provider_support": "Underlying cloud hypervisor degradation",
    },
}

deberta_answer = predict_jev_answer(state, question_spec)

jev_reference_response = {
    "type": "choice",
    "choice": "orders_application_team",
    "confidence": 0.93,
    "probabilities": {
        "orders_application_team": 0.89,
        "database_reliability_engineering": 0.07,
        "traffic_infrastructure": 0.03,
        "cloud_provider_support": 0.02,
    },
}

print("-" * 65)
print("DEBERTA (STUDENT) - FORMATTED JEV RESPONSE:")
print("-" * 65)
print(json.dumps(deberta_answer, indent=2))

print("ORIGINAL JEV (TEACHER) - REFERENCE RESPONSE:")
print("-" * 65)
print(json.dumps(jev_reference_response, indent=2))

print("\n" + "-" * 65)
match = deberta_answer["choice"] == jev_reference_response["choice"]
print(
    f"Decision Match: {match} (Student: '{deberta_answer['choice']}' | Teacher: '{jev_reference_response['choice']}')"
)
print("-" * 65)
