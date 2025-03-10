# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple
import pandas as pd

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftConfig, PeftModel
import torch
import time
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import time
import re
from collections import Counter
import string
import re
import argparse
import json
import sys

from llama import LLaMA, ModelArgs, Tokenizer, Transformer
device = "cuda" if torch.cuda.is_available() else "cpu"
    

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    quantizer: bool=False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=True)
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu",weights_only=False)

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    model_args.adapter_layer = 30
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    print(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    # model.load_state_dict(adapter_checkpoint, strict=False)
    # Load attention gate parameters for each layer

    print("Loading per-layer attention gate parameters")
    attention_gate_dict = adapter_checkpoint["gate"]  # Dictionary { "L0.attention.gate": tensor, ... }
    # print(attention_gate_dict)

    # Manually update only the matching layers
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(name)
            if "module." + name in attention_gate_dict.keys():
                print("yes")
                # print(param)
                print(f"Updating {name}")
                param.copy_(attention_gate_dict["module." + name])
                # print(param)
            if "adapter_query.weight" == name:
                print(param)
                param.copy_(adapter_checkpoint["adapter_query.weight"])
                print(param)

    print("Attention gate parameters loaded successfully.")
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len: int = 3500,
    max_batch_size: int = 32,
    quantizer: bool = False,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size, quantizer)

    def getInt(s, options):
        res = s
        res = normalize_answer(res)
        options = normalize_answer(options)
        print(f'res{res}, options{options}')
        # best_match, score = process.extractOne(res, options)
        if options in res:
            print('yes')
            return res
        else:
            print('No')
            return 'None'

    def inference(input_prompt, generator):

        output_text = generator.generate([input_prompt], max_gen_len=100, temperature=temperature, top_p=top_p)
        file = open('llama2_English_tuned_xsquad_prompt2f.txt', 'a') 
        file.write(str(output_text[0])+"\n ******************* \n")
        file.close()
        output_texts = output_text[0][len(input_prompt)-10:]
        return output_texts
    
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        print(f'prediction_tokens:{prediction_tokens} and ground_truth_tokens:{ground_truth_tokens}')
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        # print(f'num_same {num_same}')
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def exact_match_score(prediction, ground_truth):
        return (normalize_answer(prediction) == normalize_answer(ground_truth))


    l = {}
    # instr = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction: \nThe task is to solve reading comprehension problems. You will be provided questions on a set of passages and you will need to provide the answer as it appears in the passage. The answer should be in the same language as the question and the passage. Answer as concisely as possible in the same format as the examples below:\n"
    instr = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction: \nYou will answer reading comprehension questions using information from a provided passage. Extract the exact answer from the passage without modification and present it in the following structured format:\n{'answer': <Extracted Answer>} \n"


    for lang in ['en']:
        f1 = exact_match = total = 0
        print(lang)
        a = 0
        corr = 0
        tot = 0
        ambig = 0
        t1 = time.time()
        code = 'xquad.' + lang
        dataset = load_dataset("google/xquad", code)
        dataset = dataset['validation']
        # dataset = dataset['validation'].train_test_split(test_size=0.1, seed=42)
        # dataset = dataset['test']
        print(dataset)
        for inst in dataset:
            a += 1
            prompt =  '\n### Input: \n' + 'Context:\n' + inst["context"] +  "\n"  + 'Question:\n' + inst["question"] + "\n"
            inp_prompt = instr + prompt + '\n###Response:\n{"answer":'
            outputs = inference(inp_prompt,generator)
            ground_truths = inst['answers']['text'][0]
            match = re.search(r'{"answer": (.*?)}', outputs)
            if match:
                res = match.group(1)
                print(res)
            else:
                res =""
                print("Answer not found")
            file = open('llama2_English_tuned_xsquad.txt', 'a') 
            file.write(res + '---' + ground_truths +"\n ******************* \n")
            file.close()       
            exact_match += exact_match_score(res, ground_truths)
            f1 += f1_score(res, ground_truths)
            total += 1 
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        print(exact_match, f1) 
        l[lang] = [f1, exact_match]



    

    df = pd.DataFrame(l)
    df = df.T
    df.to_csv("llama2_English_tuned_xsquad.csv")

if __name__ == "__main__":
    fire.Fire(main)
