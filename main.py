import argparse
import json
import logging
import sys
from pathlib import Path

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
    device = "cuda" if enable_cuda else "cpu"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    # Enable CUDA GPU acceleration for the model if specified
    model.to(device)

    logging.info("Generating answers.")
    results = []
    for prompt in tqdm(prompts, desc="Generating Answers", unit=" answer", smoothing=0.06):
        # Tokenize the prompt
        prompt_tokenized = tokenizer(prompt, return_tensors="pt")
        # Enable CUDA GPU acceleration for the tokenized prompt if specified
        prompt_tokenized.to(device)
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
        prompt.append(template.format(
            prompt=f"{data['question']}\nFeel free to reason out loud before concluding. "
                   f"Once you've reached your final answer, present it as a single, concise sentence on a new line, "
                   f"starting with \"[Answer]:\" for clear emphasis."))

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

    # Declaration of commonly used value
    length = len(model_answers)

    # Ensure the number of answers are equal so `i` won't go out of bounds
    if not (length == len(correct_answers)):
        logging.error("Mismatch in the number of answers!")
        sys.exit(1)

    # Compute the score of answers
    logging.info("Computing score of answers.")
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api,
    )
    # Main logic
    acc = inst = acc_err = 0
    for i in tqdm(range(length), desc="Evaluating answers with GPT-4", unit=" answer", smoothing=0.06):
        # Declaration of commonly used value
        answer = model_answers[i]

        # Logic for inst
        if "\n[Answer]:" in answer:
            answer = answer.split("\n[Answer]:")[1]
            inst += 2
        elif "[Answer]:" in answer:
            answer = answer.split("[Answer]:")[1]
            inst += 1
        else:
            acc_err += 1
            continue

        # Logic for acc
        # Message to query GPT-4 with
        message = f"Answer 1: {answer}\nAnswer 2: {correct_answers[i]}\nOutput "
        f"\"True\" if both answers carry the same information. Otherwise, output \"False\"."
        # Analyze with GPT-4
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                },
            ],
            model="gpt-4",
            temperature=0,
        )

        # Process GPT-4 response
        if "True" in chat_completion:
            if "False" in chat_completion:
                logging.warning("Error in GPT-4 judging, both \"True\" and \"False\" are present. "
                                f"GPT-4 response: \"{chat_completion}\"\nRetrying with 2 attempts.")
                acc_reevaluate = reevaluate(message, client)
                acc += acc_reevaluate
                acc_err += 1 - acc_reevaluate
            else:
                acc += 1
        elif "False" not in chat_completion:
            logging.warning("Error in GPT-4 judging, both \"True\" and \"False\" are not present. "
                            f"GPT-4 response: \"{chat_completion}\"\nRetrying with 2 attempts.")
            acc_reevaluate = reevaluate(message, client)
            acc += acc_reevaluate
            acc_err += 1 - acc_reevaluate

    # Convert raw scores into percentiles and return them
    acc_percent = (acc / length) * 100
    inst_percent = (inst / length) * 50
    acc_err_percent = (acc_err / length) * 100
    return acc_percent, inst_percent, acc_err_percent


def reevaluate(prompt, client):
    # Two retries
    for _ in tqdm(range(2), desc="Retrying GPT-4 evaluation", unit=" answer", smoothing=0.06):
        # Analyze with GPT-4
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="gpt-4",
            temperature=0,
        )
        # Check whether GPT-4 makes an error in retries, if so, continue retrying until _ = 1
        if "True" in chat_completion and "False" not in chat_completion:
            return 1
    # GPT-4 keeps making errors
    return 0


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

    # Check for valid precision
    if precision is None:
        logging.error(f"Invalid precision setting \"{args.precision}\". Choose \"fp16\" or \"fp32\". ")
        sys.exit(1)

    # Check for valid template
    if "{prompt}" not in args.template:
        logging.error(f"Invalid template \"{args.template}\", replace where the prompt goes with " + "\"{prompt}\".")
        sys.exit(1)

    # Check for valid dataset and save file
    if not Path(args.jsonl).exists():
        logging.error(f"Invalid dataset \"{args.jsonl}\".")
        sys.exit(1)
    if not (args.savefile == "" or Path(args.savefile).is_file()):
        logging.error(f"Invalid save file \"{args.savefile}\".")
        sys.exit(1)

    # Check for valid CUDA configuration
    if args.enablecuda and not torch.cuda.is_available():
        logging.warning("CUDA is enabled, but CUDA is not available with PyTorch. "
                        "Make sure you have CUDA installed and PyTorch compiled with CUDA. "
                        "Automatically disabling CUDA.")
        args.enablecuda = False

    # Warn user when trust remote code was enabled
    if args.trustremotecode:
        logging.warning("Trust remote code is enabled, this is dangerous.")

    # Passes the arguments down for response generation
    answers = generate(
        args.model,
        args.jsonl,
        args.template,
        args.enablecuda,
        args.maxnewtokens,
        precision,
        args.trustremotecode
    )

    # Passes the arguments down for score evaluation
    score = evaluate(answers, args.jsonl, args.api)

    # Prints the score
    print(f"Acc: {score[0]}%\nInst: {score[1]}%\nAcc_err: +-{score[2]}%")

    # Save result
    if not args.savefile == "":
        logging.info(f"Saving result to {args.savefile}.")
        save = {
            "model": args.model,
            "template": args.template,
            "precision": args.precision,
            "maxnewtokens": args.maxnewtokens,
            "acc": score[0],
            "inst": score[1],
            "acc_err": score[2],
        }
        with open(args.savefile, "x") as savefile_json:
            json.dump(save, savefile_json)
