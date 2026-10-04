"""Independent adapter inference and robust two-class output parsing."""
from __future__ import annotations
import re
from src.training.prepare_dataset import instruction_prompt

def parse_classification(text: str) -> str:
    """Prefer the final explicit class token; reject ambiguous generations."""
    labels=re.findall(r"\b(GROUNDED|HALLUCINATED)\b", text.upper())
    if not labels: raise ValueError("Model did not return GROUNDED or HALLUCINATED.")
    return labels[-1]

def classify(context: str, response: str, adapter_path: str) -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import get_settings
    s=get_settings(); tok=AutoTokenizer.from_pretrained(adapter_path); base=AutoModelForCausalLM.from_pretrained(s.model_name,token=s.hf_token,device_map={"":0}); model=PeftModel.from_pretrained(base,adapter_path)
    prompt=instruction_prompt({"context":context,"response":response}); ids=tok(prompt,return_tensors="pt").to(model.device); output=model.generate(**ids,max_new_tokens=4,do_sample=False)
    return parse_classification(tok.decode(output[0][ids["input_ids"].shape[1]:],skip_special_tokens=True))
