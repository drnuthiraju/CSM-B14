"""Standalone QLoRA training; it never runs when the web app starts."""
from __future__ import annotations
import argparse, importlib.util, random
from pathlib import Path
import numpy as np
import torch
from src.config import get_settings
from src.training.prepare_dataset import load_prepared

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

def hardware() -> dict:
    cuda = torch.cuda.is_available()
    return {"cuda": cuda, "name": torch.cuda.get_device_name(0) if cuda else None,
            "vram": torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda else 0,
            "bf16": bool(cuda and torch.cuda.is_bf16_supported()), "bnb": importlib.util.find_spec("bitsandbytes") is not None}

def preflight(train: Path, validation: Path, minimum_vram: float) -> dict:
    missing = [str(p) for p in (train, validation) if not p.is_file()]
    if missing: raise FileNotFoundError("Missing prepared splits: " + ", ".join(missing))
    if not get_settings().hf_token: raise RuntimeError("HF_TOKEN is missing; set it in .env without committing it.")
    report = hardware()
    if not report["bnb"]: raise RuntimeError("bitsandbytes is unavailable; use a compatible CUDA environment.")
    if not report["cuda"]: raise RuntimeError("CUDA is unavailable. QLoRA training needs a supported NVIDIA CUDA GPU.")
    if report["vram"] < minimum_vram: raise RuntimeError(f"GPU has {report['vram']:.1f} GB VRAM; this configuration requires {minimum_vram:.1f}+ GB. 4 GB is not practical for Llama 3.1 8B QLoRA; use Colab/Kaggle.")
    return report

def masked_tokenize(records, tokenizer, max_length: int):
    """Mask prompt tokens so causal-LM loss is computed only for the class label."""
    output=[]
    for row in records:
        prefix=tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        target=tokenizer(row["label"]+tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids=(prefix+target)[:max_length]
        if len(ids) <= len(prefix): raise ValueError("--max-length truncates the classification label.")
        output.append({"input_ids":ids,"attention_mask":[1]*len(ids),"labels":[-100]*len(prefix)+ids[len(prefix):]})
    return output

class Collator:
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def __call__(self, features):
        labels=[x.pop("labels") for x in features]; batch=self.tokenizer.pad(features,padding=True,return_tensors="pt"); width=batch["input_ids"].shape[1]
        batch["labels"]=torch.tensor([x+[-100]*(width-len(x)) for x in labels]); return batch

def arguments():
    p=argparse.ArgumentParser(); p.add_argument("--train-file",type=Path,default=Path("data/processed/ragtruth/train.jsonl")); p.add_argument("--validation-file",type=Path,default=Path("data/processed/ragtruth/validation.jsonl")); p.add_argument("--output-dir",type=Path,default=Path("models/checkpoints/lora-ragtruth")); p.add_argument("--epochs",type=float,default=1); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--gradient-accumulation-steps",type=int,default=8); p.add_argument("--max-length",type=int,default=1024); p.add_argument("--learning-rate",type=float,default=2e-4); p.add_argument("--r",type=int,default=16); p.add_argument("--lora-alpha",type=int,default=32); p.add_argument("--lora-dropout",type=float,default=.05); p.add_argument("--target-modules",default=",".join(DEFAULT_TARGET_MODULES)); p.add_argument("--minimum-vram-gb",type=float,default=10); p.add_argument("--dry-run","--check-only",action="store_true"); return p.parse_args()

def main():
    a=arguments(); report=hardware(); print(f"CUDA={report['cuda']}; GPU={report['name']}; VRAM_GB={report['vram']:.2f}; bitsandbytes={report['bnb']}")
    if a.dry_run:
        try: preflight(a.train_file,a.validation_file,a.minimum_vram_gb); print("Preflight passed; no model loaded and no training started.")
        except (FileNotFoundError,RuntimeError) as e: print(f"Preflight failed: {e}"); raise SystemExit(2)
        return
    r=preflight(a.train_file,a.validation_file,a.minimum_vram_gb)
    from datasets import Dataset
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments
    s=get_settings(); random.seed(s.seed); np.random.seed(s.seed); torch.manual_seed(s.seed)
    tok=AutoTokenizer.from_pretrained(s.model_name,token=s.hf_token); tok.pad_token=tok.eos_token; tok.padding_side="right"
    dtype=torch.bfloat16 if r["bf16"] else torch.float16
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(s.model_name,token=s.hf_token,quantization_config=q,device_map={"":0}); model.config.use_cache=False; model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=get_peft_model(model,LoraConfig(r=a.r,lora_alpha=a.lora_alpha,lora_dropout=a.lora_dropout,target_modules=[x.strip() for x in a.target_modules.split(",")],bias="none",task_type="CAUSAL_LM"))
    train=Dataset.from_list(masked_tokenize(load_prepared(a.train_file),tok,a.max_length)); val=Dataset.from_list(masked_tokenize(load_prepared(a.validation_file),tok,a.max_length))
    args=TrainingArguments(output_dir=str(a.output_dir),num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.batch_size,gradient_accumulation_steps=a.gradient_accumulation_steps,learning_rate=a.learning_rate,fp16=dtype==torch.float16,bf16=dtype==torch.bfloat16,gradient_checkpointing=True,optim="paged_adamw_8bit",eval_strategy="epoch",save_strategy="epoch",seed=s.seed,report_to="none",remove_unused_columns=False)
    Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=Collator(tok)).train(); a.output_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir)
if __name__ == "__main__": main()
