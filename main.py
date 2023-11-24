import argparse
import json
import logging
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer


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


def evaluate(model_answers, file, enable_cuda):
    # Load benchmark correct answers
    logging.info("Processing benchmark correct answers.")
    correct_answers = []
    for data in tqdm(read_jsonl(file), desc="Processing Benchmark Correct Answers", unit=" answer", smoothing=0.06):
        correct_answers.append(data["answer"])

    # Enable CUDA GPU acceleration if specified
    device = "cuda" if enable_cuda else "cpu"

    # Load embedding model
    logging.info("Loading embedding model.")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device)

    # Ensure the number of answers are equal so `i` won't go out of bounds
    if not (len(model_answers) == len(correct_answers)):
        print("Mismatch in the number of answers!")
        sys.exit(1)

    # Compute the score of answers
    logging.info("Computing score of answers.")
    similarity = 0
    for i in tqdm(range(0, len(model_answers), 1), desc="Computing Score of Answers", unit="answer", smoothing=0.06):
        model_answer_embed = generate_embedding(model_answers[i], tokenizer, model, device, 512)
        correct_answer_embed = generate_embedding(correct_answers[i], tokenizer, model, device, 512)
        similarity += compute_similarity(0.5, model_answer_embed, correct_answer_embed)
    return similarity


def generate_embedding(text, tokenizer, model, device, max_seq_len):
    # Generate embedding for `text` with embedding model `model`
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    # Ensure that the embeddings are on CPU, required for NumPy
    return embeddings.cpu().numpy()[0]


def compute_similarity(alpha, model_ans_embed, correct_ans_embed):
    # Computes cosine similarity and Euclidean similarity
    norm_model_ans = model_ans_embed / np.linalg.norm(model_ans_embed)
    norm_correct_ans = correct_ans_embed / np.linalg.norm(correct_ans_embed)
    cosine_similarity = np.dot(norm_model_ans, norm_correct_ans)
    euclidean_dist = np.linalg.norm(model_ans_embed - correct_ans_embed)
    normalized_euclidean_similarity = 1 / (1 + euclidean_dist)

    # Multiplies cosine similarity by a factor of `alpha` and Euclidean similarity by a factor of `1 - alpha`
    similarity_score = alpha * cosine_similarity + (1 - alpha) * normalized_euclidean_similarity

    return similarity_score


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
        print("Invalid precision setting. Choose 'fp16' or 'fp32'.")
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
    score = evaluate(answers, args.jsonl, args.enablecuda)

    # Prints the score
    print(f"Score: {evaluate}")

    # Save information
    logging.info(f"Saving information to {args.savefile}")
    save = {
        "model": args.model,
        "template": args.template,
        "precision": args.precision,
        "maxnewtokens": args.maxnewtokens,
        "score": score
    }
    with open(args.savefile, "w") as savefile_json:
        json.dump(save, savefile_json)
