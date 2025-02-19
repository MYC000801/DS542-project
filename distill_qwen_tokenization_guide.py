from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import torch
import random
import numpy as np


def get_message(instruction=None, response=None):

    assert instruction != None or response != None

    if response == None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction == None:
        message = [
            {"role": "assistant", "content": response}
        ]
    else:
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    return message


def main():

    # init
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_parallel_size=1,
    )

    # prompts for llm
    print('='*50)
    original_prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    print(f"original prompt: {original_prompt}\n")
    print('='*50)
    prompt = [tokenizer.apply_chat_template(get_message(original_prompt), tokenize=False, add_generation_prompt=True)]
    print(f"input prompt to vllm: {prompt}\n")
    print('='*50)

    # response
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024)
    response = llm.generate(prompt, sampling_params)
    response = list(map(lambda x: x.outputs[0].text, response))[0]
    print(f"response from vllm: {response}\n")
    print('='*50)

    # push prompt and response into a message
    message = get_message(original_prompt, response)
    print(f"message: {message}\n")
    print('='*50)

    # tokenizer.apply_chat_template error due to:
    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/tokenizer_config.json#:~:text=%22-,chat_template,-%22%3A%20%22%7B%25%20if
    print("tokenizer.apply_chat_template has an error where it automatically removes the thought in the template:")
    message_chat_template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    print(f"message_chat_template: {message_chat_template}\n")
    print('='*50)

    # solution
    print("we can solve it by tokenize prompt and response separately:")
    prompt_str = tokenizer.apply_chat_template([message[0]], add_generation_prompt=True, tokenize=False)
    response_str = message[1]['content'] + tokenizer.decode([tokenizer.eos_token_id])
    prompt_response_str = prompt_str + response_str
    print(f"prompt_response_str: {prompt_response_str}\n")


if __name__ == "__main__":
    main()
