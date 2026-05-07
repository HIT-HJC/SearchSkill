#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import string
from typing import List, Dict, Any

import requests
import torch
import transformers
from tqdm import tqdm


#########################################
# 与 infer.py 对齐的 prompt & 搜索逻辑
#########################################

# Qwen2.5 的 eos token（和 infer.py 一致）
CURR_EOS = [151645, 151643]

# infer.py 里的模板：把每一轮输出 + <information> 检索结果 拼在后面
CURR_SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"

# infer.py 的用户 prompt 模板（语义等价）
USER_PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: {question}\n"
)

# infer.py 中用于停止的目标序列
TARGET_SEQUENCES = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]


class StopOnSequence(transformers.StoppingCriteria):
    """与 infer.py 一致的 StoppingCriteria：看到 </search> 系列就停一轮"""

    def __init__(self, target_sequences, tokenizer):
        super().__init__()
        self.target_ids = [tokenizer.encode(ts, add_special_tokens=False) for ts in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(t, device=input_ids.device) for t in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def extract_last_search_query(text: str) -> str:
    """与 infer.py 的 get_query 一致：取最后一个 <search>...</search> 里的内容"""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def call_retriever(query: str, port: int) -> str:
    """
    按作者规范调用检索器：
    POST http://127.0.0.1:{port}/retrieve
    body: {"queries": [query], "topk": 3, "return_scores": true}
    把结果转成：
      Doc i(Title: title) text
    拼成一个长字符串
    """
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True,
    }
    url = f"http://127.0.0.1:{port}/retrieve"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    results = r.json()["result"]

    def passages_to_string(retrieval_result: List[Dict[str, Any]]) -> str:
        s = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            parts = content.split("\n")
            title = parts[0]
            text = "\n".join(parts[1:])
            s += f"Doc {idx+1}(Title: {title}) {text}\n"
        return s

    return passages_to_string(results[0])


def build_chat_prompt(tokenizer, question: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    user_prompt = USER_PROMPT_TEMPLATE.format(question=q)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return user_prompt


#########################################
# EM 计算（多答案 + SQuAD 风格归一化）
#########################################

def normalize_answer(s: str) -> str:
    """
    标准 SQuAD 风格归一化：
      - 小写
      - 去标点
      - 去 a/an/the
      - 合并空白
    Search-R1 论文里 EM 参照 Yu 等人的 QA 评测，这种归一化是主流做法。
    """

    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_answer_from_output(text: str) -> str:
    """
    从模型输出中抽取 <answer>...</answer>。
    如果没有标注，就用最后一行兜底。
    """
    m = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    return text.strip().splitlines()[-1].strip()


def get_gold_answers(ex: Dict[str, Any]) -> List[str]:
    """
    HotpotQA 里 answer 通常是：
      - "Crystal Dynamics"
      - ["Crystal Dynamics"]
    这里统一转成 list[str]，支持多答案匹配。
    """
    raw = ex.get("answer", "")
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip() != ""]
    else:
        s = str(raw).strip()
        return [s] if s else []


def exact_match_multi(pred: str, gold_list: List[str]) -> int:
    """
    EM：预测命中任意一个 gold（归一化后相同）就算 1。
    """
    if not gold_list:
        return 0
    norm_pred = normalize_answer(pred)
    for g in gold_list:
        if norm_pred == normalize_answer(g):
            return 1
    return 0


#########################################
# 数据加载
#########################################

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


#########################################
# 主流程
#########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--retriever_port", type=int, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)

    # 解码参数（与 infer.py 对齐，top_p 可选）
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=8, help="最大搜索轮数（防止死循环）")

    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_jsonl))
    log_dir = os.path.dirname(os.path.abspath(args.log_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # logging 配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("eval")

    #########################################
    # 加载模型：显式用 float32
    #########################################
    logger.info("Loading model & tokenizer from %s (float32)", args.model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,  # 显式 FP32，避免你之前遇到的 FP 异常
        device_map="auto",
    )

    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(TARGET_SEQUENCES, tokenizer)]
    )

    # 数据
    ds = load_jsonl(args.data_path)
    logger.info("Loaded %d examples from %s", len(ds), args.data_path)

    n_correct = 0
    acc_every = 20

    with open(args.out_jsonl, "w", encoding="utf-8") as fout, torch.no_grad():
        for idx, ex in enumerate(tqdm(ds, desc="Evaluating", total=len(ds))):
            question = ex.get("question", "").strip()
            gold_answers = get_gold_answers(ex)

            prompt = build_chat_prompt(tokenizer, question)
            step_cnt = 0
            trace_steps = []  # 记录该样本的完整过程

            # 推理 + 检索循环
            while True:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                attention_mask = torch.ones_like(input_ids)

                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=args.temperature,
                )
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p

                outputs_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

                generated_tokens = outputs_ids[0, input_ids.shape[1]:]
                out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                full_text = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
                tmp_query = extract_last_search_query(full_text)

                # 如果已经以 EOS 结束，认为这是最后一轮，不再检索；但仍然记录这轮 generated。
                if outputs_ids[0, -1].item() in CURR_EOS:
                    trace_steps.append(
                        {
                            "step": step_cnt,
                            "generated": out_text,
                            "query": tmp_query,
                            "retrieved": None,
                        }
                    )
                    break

                # 还没结束 —— 正常的“搜索轮”
                if tmp_query:
                    try:
                        search_results = call_retriever(tmp_query, args.retriever_port)
                    except Exception as e:
                        logger.warning(f"Retriever error at idx={idx}: {e}")
                        search_results = ""
                else:
                    search_results = ""

                trace_steps.append(
                    {
                        "step": step_cnt,
                        "generated": out_text,
                        "query": tmp_query,
                        "retrieved": search_results,
                    }
                )

                # 把这轮输出和检索结果塞回 prompt
                prompt += CURR_SEARCH_TEMPLATE.format(
                    output_text=out_text, search_results=search_results
                )
                step_cnt += 1
                if step_cnt >= args.max_steps:
                    # 安全停止
                    break

            # 从最后一轮输出中抽 answer
            final_output_text = trace_steps[-1]["generated"] if trace_steps else out_text
            pred_answer = extract_answer_from_output(final_output_text)
            em = exact_match_multi(pred_answer, gold_answers)
            n_correct += em

            # 写入完整记录（包含过程）
            record = {
                "idx": idx,
                "question": question,
                "gold": gold_answers,
                "prediction": pred_answer,
                "em": em,
                "steps": trace_steps,  # 每一步的 generated / query / retrieved
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 打 log：显式展示每个样本的过程
            logger.info("=" * 80)
            logger.info(f"[Example {idx}] Q: {question}")
            for step in trace_steps:
                logger.info(f"  [Step {step['step']}] generated:")
                logger.info(step["generated"])
                if step["query"] is not None:
                    logger.info(f"    [Query] {step['query']}")
                    logger.info(f"    [Retrieved]\n{step['retrieved']}")
            logger.info(f"  GOLD: {gold_answers}")
            logger.info(f"  PRED: {pred_answer} (EM={em})")

            if (idx + 1) % acc_every == 0:
                curr_acc = n_correct / (idx + 1)
                logger.info(f"[{idx + 1}] running EM accuracy = {curr_acc:.4f}")

    final_acc = n_correct / max(1, len(ds))
    logger.info("Final EM accuracy on %d examples: %.4f", len(ds), final_acc)


if __name__ == "__main__":
    main()
