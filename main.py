import argparse
import json
import logging
import sys

from openai import OpenAI
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _generate(model,
              prompts,
              enable_cuda,
              max_new_tokens,
              dtype,
              trust_remote_code, ):
    logging.info("Loading model.")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    # Enable CUDA GPU acceleration for the model if specified
    if enable_cuda:
        model.to("cuda")

    logging.info("Generating answers.")
    results = []
    for prompt in tqdm(prompts, desc="Generating Answers", unit=" answer", smoothing=0.06):
        # Tokenize the prompt
        prompt_tokenized = tokenizer(prompt, return_tensors="pt")
        # Enable CUDA GPU acceleration for the tokenized prompt if specified
        if enable_cuda:
            prompt_tokenized.to("cuda")
        # Generate response
        tokens = model.generate(
            **prompt_tokenized,
            do_sample=False,
            max_new_tokens=max_new_tokens
        )

        # Do not include the prompt template and the answer, just the response
        results.append(tokenizer.decode(tokens[0], skip_special_tokens=True)[len(prompt):])
    return results


def generate(model,
             file,
             template,
             enable_cuda,
             max_new_tokens,
             dtype,
             trust_remote_code, ):
    # Load benchmark questions
    logging.info("Processing benchmark questions.")
    prompt = []
    for data in tqdm(read_jsonl(file), desc="Processing Benchmark Questions", unit=" question", smoothing=0.06):
        prompt.append(template.format(prompt=data["question"]))

    # Pass the parameters down to generate responses
    return _generate(
        model,
        prompt,
        enable_cuda,
        max_new_tokens,
        dtype,
        trust_remote_code,
    )


def evaluate(model_answers, file, api):
    # Load benchmark correct answers
    logging.info("Processing benchmark correct answers.")
    correct_answers = []
    for data in tqdm(read_jsonl(file), desc="Processing Benchmark Correct Answers", unit=" answer", smoothing=0.06):
        correct_answers.append(data["answer"])

    # Ensure the number of answers are equal so `i` won't go out of bounds
    if not (len(model_answers) == len(correct_answers)):
        logging.error("Mismatch in the number of answers!")
        sys.exit(1)

    # Compute the score of answers
    logging.info("Computing score of answers.")
    client = OpenAI(
        api_key=api,
    )

    # Evaluate score with GPT-4
    acc = 0
    err = 0
    for i in tqdm(range(0, len(model_answers), 1), desc="Evaluating answers via GPT-4", unit=" answer", smoothing=0.06):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Answer 1: {model_answers[i]}\nAnswer 2: {correct_answers[i]}\nOutput \"True\" if "
                               f"both answers carry the same information. Otherwise, output \"False\".",
                },
            ],
            model="gpt-4",
            temperature=0,
        )
        if "True" in chat_completion:
            if "False" in chat_completion:
                logging.error(f"Error in GPT-4 judging, both \"True\" and \"False\" are present. "
                              f"GPT-4 response: {chat_completion}")
                err += 100 / len(model_answers)
            else:
                acc += 100 / len(model_answers)
    return [acc, err]


def read_jsonl(file_path):
    # Reads a JSONL file line-by-line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("jsonl", type=str, nargs='?', default="test.jsonl", help="JSONL dataset")
    parser.add_argument("template", type=str, help="Prompt template to use")
    parser.add_argument("api", type=str, help="OpenAI API key")
    parser.add_argument("--enablecuda", type=bool, default=False, help="Enable CUDA (True/False)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Precision mode")
    parser.add_argument("--maxnewtokens", type=int, default=1024, help="Maximum new tokens")
    parser.add_argument("--trustremotecode", type=bool, default=False, help="Trust remote code (True/False)")
    parser.add_argument("--savefile", type=str, default="", help="Path for results to be saved")
    args = parser.parse_args()

    # Convert str to torch.dtype precision
    precision_map = {
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    precision = precision_map.get(args.precision, None)
    if precision is None:
        logging.error("Invalid precision setting. Choose 'fp16' or 'fp32'.")
        sys.exit(1)

    # Passes the arguments down for response generation
    answers = generate(
        args.model,
        args.jsonl,
        args.template,
        args.enablecuda,
        precision,
        args.maxnewtokens,
        args.trustremotecode
    )

    # Passes the arguments down for score evaluation
    score = evaluate(answers, args.jsonl, args.api)

    # Prints the score
    print(f"Score: {score[0]}\nError: {score[1]}")

    # Save information
    logging.info(f"Saving information to {args.savefile}.")
    save = {
        "model": args.model,
        "template": args.template,
        "precision": args.precision,
        "maxnewtokens": args.maxnewtokens,
        "score": score[0],
        "error": score[1],
    }
    with open(args.savefile, "w") as savefile_json:
        json.dump(save, savefile_json)
