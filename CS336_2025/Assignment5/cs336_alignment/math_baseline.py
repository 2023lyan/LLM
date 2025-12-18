from vllm import LLM, SamplingParams
from typing import Callable, List
import sys
import pathlib
from datasets import load_dataset
import json
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from drgrpo_grader import r1_zero_reward_fn

PROMPT_PATH = "./prompts/r1_zero.prompt"
OUT_PATH = "./output"

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
):
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    total_format = 0
    total_answer = 0
    total_reward = 0
    
    for i, output in enumerate(outputs):
        model_output = output.outputs[0].text
        gold_answer = answers[i]

        reward = reward_fn(model_output, gold_answer)

        results.append({
            "prompt": prompts[i],
            "model_output": model_output,
            "gold_answer": gold_answer,
            "reward": reward,
        })

        total_reward += reward["reward"]
        total_format += reward["format_reward"]
        total_answer += reward["answer_reward"]
        
    print(f"Accuracy = {total_reward / len(prompts):.4f}")
    print(f"Format Accuracy = {total_format / len(prompts):.4f}")
    print(f"False = {(1 - total_format / len(prompts)):.4f}")
    
    os.makedirs(OUT_PATH, exist_ok=True)
    
    with open(pathlib.Path(OUT_PATH) / "zero_shot_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
if __name__ == "__main__":
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
        )

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        r1_zero_prompt = f.read()

    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")["valid"]

    prompts = []
    answers = []

    for item in ds:
        q = item["question"]
        a = str(item["answer"])
        
        prompt = r1_zero_prompt.replace("{question}", q)
        prompts.append(prompt)
        answers.append(a)
        
    evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        answers,
        sampling_params
    )