import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft = inputs.pop("soft_probability")
        outputs = model(**inputs)
        t = 2.0
        teacher = torch.stack([1.0 - soft, soft], dim=-1)
        student = F.log_softmax(outputs.logits / t, dim=-1)
        loss = (
            0.5 * F.kl_div(student, teacher, reduction="batchmean") * (t**2)
            + 0.5 * outputs.loss
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((logits.argmax(axis=-1) == labels).mean())}


ds = load_dataset("martin76ec/infrastructure-incident-synthetic", "default")
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")


def prep(b):
    out = tok(b["premise"], b["hypothesis"], truncation=True, max_length=256)
    out["labels"] = b["label"]
    out["soft_probability"] = [float(p) for p in b["soft_probability"]]
    return out


tokenized = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)
base = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small", num_labels=2
)
model = get_peft_model(
    base,
    LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["query_proj", "value_proj"],
    ),
)

args = TrainingArguments(
    output_dir="./out",
    learning_rate=3e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
    logging_steps=50,
    report_to="none",
)

trainer = DistillTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

torch.cuda.reset_peak_memory_stats()
train_out = trainer.train()
peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

test_res = trainer.evaluate(tokenized["test"])
test_preds = trainer.predict(tokenized["test"])
student_probs = F.softmax(torch.tensor(test_preds.predictions), dim=-1)[:, 1].numpy()
teacher_probs = np.array(tokenized["test"]["soft_probability"])

train_ep, train_loss = [], []
eval_ep, eval_loss, eval_acc = [], [], []
for h in trainer.state.log_history:
    if "loss" in h and "epoch" in h:
        train_ep.append(h["epoch"])
        train_loss.append(h["loss"])
    if "eval_loss" in h and "epoch" in h:
        eval_ep.append(h["epoch"])
        eval_loss.append(h["eval_loss"])
        eval_acc.append(h["eval_accuracy"])

fig, axs = plt.subplots(2, 2, figsize=(11, 9))

axs[0, 0].plot(train_ep, train_loss, label="Train Loss", color="royalblue")
axs[0, 0].plot(eval_ep, eval_loss, label="Val Loss", color="crimson", marker="o")
axs[0, 0].set_title("Loss Curves")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(eval_ep, eval_acc, color="forestgreen", marker="o")
axs[0, 1].set_title("Validation Accuracy")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Accuracy")
axs[0, 1].grid(True)

axs[1, 0].scatter(teacher_probs, student_probs, alpha=0.15, s=8, color="darkorange")
axs[1, 0].plot([0, 1], [0, 1], "k--")
axs[1, 0].set_title("Distillation Calibration (Jev vs DeBERTa)")
axs[1, 0].set_xlabel("Teacher Soft Probability")
axs[1, 0].set_ylabel("Student Soft Probability")
axs[1, 0].grid(True)

bars = ["Peak VRAM (GB)", "Train Time (min)", "Test Acc (%)"]
vals = [
    peak_vram / 1024,
    train_out.metrics["train_runtime"] / 60,
    test_res["eval_accuracy"] * 100,
]
axs[1, 1].bar(bars, vals, color=["steelblue", "slategray", "teal"])
axs[1, 1].set_title("System & Final Test Performance")
for i, v in enumerate(vals):
    axs[1, 1].text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center")
axs[1, 1].grid(axis="y")

plt.tight_layout()
plt.savefig("metrics.png", dpi=150)
print(
    f"Peak VRAM: {peak_vram:.2f} MB | Test Acc: {test_res['eval_accuracy'] * 100:.2f}% | Saved: metrics.png"
)
