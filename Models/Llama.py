import re
import torch
from typing import List, Dict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

import pronouncing  # CMUdict-based

# ---------- Phoneme utilities (ARPAbet via pronouncing) ----------

_WORD_RE = re.compile(r"[A-Za-z']+")

def strip_stress(arpabet: str) -> str:
    # Remove 0/1/2 stress digits (e.g., AH0 -> AH)
    return re.sub(r"\d", "", arpabet)

def text_to_arpabet_words(text: str) -> List[str]:
    """
    Converts text into a list of word-level ARPAbet strings (stress-stripped).
    Falls back to the raw word if not in CMUdict to avoid losing alignment.
    """
    words = _WORD_RE.findall(text)
    arpawords = []
    for w in words:
        lw = w.lower()
        phones = pronouncing.phones_for_word(lw)
        if phones:
            # choose first pronunciation; strip stress
            arp = strip_stress(phones[0])
            arpawords.append(arp)
        else:
            # Fallback: mark as unknown token; you can also just keep the grapheme
            # arpawords.append(lw)  # alternative
            arpawords.append(f"UNK({lw})")
    return arpawords

def text_to_phoneme_line(text: str) -> str:
    """
    Formats phonemes as word-separated with ' | ' between words.
    Example: "the cat" -> "DH AH | K AE T"
    """
    arpawords = text_to_arpabet_words(text)
    return " ".join(arpawords)


# ---------- Dataset building (phonemes -> text) ----------

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

def build_example(text: str) -> Dict[str, str]:
    """
    Builds one training pair where the INPUT is phonemes and the TARGET is original text.
    We’ll create:
      - prompt: everything up to (and including) the <TEXT> line
      - target: the original text (model should generate this)
    """
    text = (text or "").strip()
    if not text:
        return None

    phon_line = text_to_phoneme_line(text)

    prompt = (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phon_line}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )
    target = text  # what we want the model to generate
    full = prompt + target
    return {"prompt": prompt, "target": target, "full": full}

def prepare_split(split: str):
    base = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    base = base.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)

    def mapper(batch):
        outs = [build_example(t) for t in batch["text"]]
        # filter Nones (empty lines etc.)
        outs = [o for o in outs if o is not None]
        if not outs:
            return {"prompt": [], "target": [], "full": []}
        return {
            "prompt": [o["prompt"] for o in outs],
            "target": [o["target"] for o in outs],
            "full":   [o["full"]   for o in outs],
        }

    ds = base.map(mapper, batched=True, remove_columns=base.column_names)
    return ds


# ---------- Tokenization with prefix-masked labels ----------

def make_tokenize_fn(tokenizer, max_length: int = 512, min_target_tokens: int = 4):
    """
    Ensures every example has at least `min_target_tokens` supervised tokens.
    We encode prompt and target separately, then truncate the prompt to leave room.
    """
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id must be set"

    def _tok(batch):
        input_ids_batch, attn_batch, labels_batch = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            # Encode WITHOUT adding extra special tokens
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t = tokenizer(target, add_special_tokens=False)["input_ids"]

            # Skip pathological cases
            if len(t) == 0:
                continue

            # Reserve space for target
            max_prompt_len = max_length - min_target_tokens
            if max_prompt_len <= 0:
                continue

            # Truncate prompt to leave room
            if len(p) > max_prompt_len:
                p = p[:max_prompt_len]

            # Fit as much target as possible, but keep at least min_target_tokens
            space = max_length - len(p)
            if space < min_target_tokens:
                # Even after truncating prompt, no room left → skip example
                continue
            t = t[:space]

            ids = p + t
            attn = [1] * len(ids)
            labs = ([-100] * len(p)) + t[:]  # supervise only target

            # Pad to max_length
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids  += [pad_id] * pad_len
                attn += [0] * pad_len
                labs += [-100] * pad_len

            input_ids_batch.append(ids)
            attn_batch.append(attn)
            labels_batch.append(labs)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels": labels_batch,
        }
    return _tok

def debug_supervision(ds, name):
    import numpy as np
    import random
    n = min(2000, len(ds))
    cnt = 0
    for i in range(n):
        labs = ds[i]["labels"]
        if any(l != -100 for l in labs):
            cnt += 1
    print(f"[{name}] examples with at least 1 supervised token: {cnt}/{n}")

# ---------- Custom collator (keep labels, just pad if needed) ----------

class CausalLMDataCollator(DataCollatorWithPadding):
    """
    Uses tokenizer padding for inputs. Expects 'labels' already provided;
    pads labels with -100 to match input length.
    """
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels")
        batch = super().__call__(features)

        max_len = batch["input_ids"].shape[1]
        padded = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded.append(lab)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


# ---------- Main training with LoRA ----------

def main():
    # 1) Build dataset
    print("Building phonemes → text dataset...")
    train_ds = prepare_split("train")
    val_ds   = prepare_split("validation")

    # 2) Tokenizer & special tokens
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add tags and ensure a pad token
    special = {"additional_special_tokens": TAGS}
    added = tokenizer.add_special_tokens(special)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA convention

    # 3) Tokenize with prompt-masked labels
    print("Tokenizing dataset with masked labels...")
    tok_fn = make_tokenize_fn(tokenizer, max_length=512)
    tokenized_train = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(tok_fn,   batched=True, remove_columns=val_ds.column_names)

    # 4) Load base model, resize embeddings, then wrap with LoRA
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # --- LoRA config (typical for LLaMA) ---
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # attention
            "gate_proj", "up_proj", "down_proj"          # MLP
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check

    # 5) Collator & args
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results_lora",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        learning_rate=2e-4,          # a bit higher is common for (Q)LoRA
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        report_to="none",
    )

    vocab = model.get_input_embeddings().num_embeddings
    print("tokenizer/model vocab:", len(tokenizer), vocab)
    debug_supervision(tokenized_train, "train")
    debug_supervision(tokenized_val, "val")

    # 6) Train
    print("Setting up Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # 7) Save adapters (default) + tokenizer
    print("Saving the LoRA adapters...")
    trainer.save_model("./llama_phonemes_to_text_lora")  # saves PEFT adapters
    tokenizer.save_pretrained("./llama_phonemes_to_text_lora")

    # --- Optional: export a merged full model (fp16, larger) ---
    print("Merging LoRA into base weights for export...")
    merged = model.merge_and_unload()
    merged.save_pretrained("./llama_phonemes_to_text_lora_merged")
    tokenizer.save_pretrained("./llama_phonemes_to_text_lora_merged")

    # 8) Evaluate
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()
