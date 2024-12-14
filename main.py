import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model,
			 sysprompt,
			 file,
			 enable_cuda,
			 max_new_tokens,
			 dtype,
			 trust_remote_code, ):
	# Load benchmark questions
	prompts = []
	for data in tqdm(read_jsonl(file), desc="Processing Benchmark Questions", unit="question", smoothing=0.06):
		prompts.append(f"{data['question']}\nFeel free to reason out loud before concluding. "
					   f"Once you've reached your final answer, present it as a single, concise sentence on a new line, "
					   f"starting with \"[Answer]:\" for clear emphasis.")

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

	# Generate answers
	answers = []
	for prompt in tqdm(prompts, desc="Generating Answers", unit="answer", smoothing=0.06):

		# Format prompt
		prompt = tokenizer.apply_chat_template(
			[
				{"role": "system", "content": sysprompt},
				{"role": "user", "content": prompt}
			],
			tokenize=False,
			add_generation_prompt=True,
		)


		# Tokenize the prompt
		prompt_tokenized = tokenizer([prompt], return_tensors="pt")
		# Enable CUDA GPU acceleration for the tokenized prompt if specified
		prompt_tokenized.to(device)
		# Generate response
		tokens = model.generate(
			**prompt_tokenized,
			do_sample=False,
			max_new_tokens=max_new_tokens
		)
		tokens = [
			output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_tokenized.input_ids, tokens)
		]

		# Do not include the prompt template and the answer, just the response
		answers.append(tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])

	print(answers)
	return answers


def evaluate(answers,
			 sysprompt,
			 file,
			 evaluator,
			 enable_cuda,
			 max_new_tokens,
			 dtype,
			 trust_remote_code, ):
	# Load benchmark correct answers
	correct_answers = []
	for data in tqdm(read_jsonl(file), desc="Processing Benchmark Correct Answers", unit="answer", smoothing=0.06):
		correct_answers.append(data["answer"])

	# Length of answers
	length = len(answers)

	# Ensure the number of answers are equal
	if not (length == len(correct_answers)):
		logging.error("Mismatch in the number of questions/answers!")
		sys.exit(1)


	# Initialize evaluator
	device = "cuda" if enable_cuda else "cpu"

	# Load tokenizer and format prompt
	tokenizer = AutoTokenizer.from_pretrained(evaluator)

	# Load model
	logging.info("Loading model")
	model = AutoModelForCausalLM.from_pretrained(
		evaluator,
		trust_remote_code=trust_remote_code,
		torch_dtype=dtype,
	)

	# Enable CUDA GPU acceleration for the model if specified
	model.to(device)

	# Main logic
	acc = inst = acc_err = 0
	for i in tqdm(range(length), desc="Evaluating answers with local evaluator", unit="answer", smoothing=0.06):
		# Answer and prompt
		answer = answers[i]
		prompt = f"Answer 1: {answer}\nAnswer 2: {correct_answers[i]}\nOutput \"True\" if both answers carry the same information. Otherwise, output \"False\"."

		# Format prompt
		prompt = tokenizer.apply_chat_template(
			[
				{"role": "system", "content": sysprompt},
				{"role": "user", "content": prompt}
			],
			tokenize=False,
			add_generation_prompt=True,
		)

		# Seperate the actual answer from response
		if "\n[Answer]:" in answer:
			answer = answer.split("\n[Answer]:")[1]
			inst += 2
		elif "[Answer]:" in answer:
			answer = answer.split("[Answer]:")[1]
			inst += 1
		else:
			acc_err += 1
			continue

		# Tokenize the prompt
		prompt_tokenized = tokenizer([prompt], return_tensors="pt")
		# Enable CUDA GPU acceleration for the tokenized prompt if specified
		prompt_tokenized.to(device)
		# Generate response
		tokens = model.generate(
			**prompt_tokenized,
			do_sample=False,
			max_new_tokens=max_new_tokens
		)
		tokens = [
			output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_tokenized.input_ids, tokens)
		]

		# Do not include the prompt template and the answer, just the respons
		evaluation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

		# Process evaluator response and look for errors
		if "True" in evaluation:
			if "False" in evaluation:
				logging.warning("Error in evaluator judging, both \"True\" and \"False\" are present. "
								f"Evaluator response: \"{evaluation}\"\nRetrying with 2 attempts.")
				acc_reevaluate = re_evaluate(model, tokenizer, prompt_tokenized, max_new_tokens)
				acc += acc_reevaluate
				acc_err += 1 - acc_reevaluate
			else:
				acc += 1

		elif "False" not in evaluation:
			logging.warning("Error in evaluator judging, both \"True\" and \"False\" are not present. "
							f"Evaluator response: \"{evaluation}\"\nRetrying with 2 attempts.")
			acc_reevaluate = re_evaluate(model, tokenizer, prompt_tokenized, max_new_tokens)
			acc += acc_reevaluate
			acc_err += 1 - acc_reevaluate

	# Convert raw scores into percentiles and return them
	acc_percent = (acc / length) * 100
	inst_percent = (inst / length) * 50
	acc_err_percent = (acc_err / length) * 100
	return acc_percent, inst_percent, acc_err_percent


def re_evaluate(model,
				tokenizer,
				prompt,
				max_new_tokens, ):
	# Two retries
	for _ in tqdm(range(2), desc="Retrying  evaluation", unit="answer", smoothing=0.06):
		# Generate response
		tokens = model.generate(
			**prompt,
			do_sample=False,
			max_new_tokens=max_new_tokens
		)
		tokens = [
			output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenizer([prompt], return_tensors="pt").input_ids, tokens)
		]

		# Do not include the prompt template and the answer, just the respons
		evaluation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

		# Check whether the evaluator makes an error in retries
		if "True" in evaluation and "False" not in evaluation:
			return 1

	# Evaluator keeps making errors
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
	parser.add_argument("sysprompt", type=str, help="System prompt for model")
	parser.add_argument("jsonl", type=str, nargs='?', default="test.jsonl", help="JSONL testset")
	parser.add_argument("evaluator", type=str, help="Evaluator model to use")
	parser.add_argument("evalsysprompt", type=str, help="System prompt for evaluator")
	parser.add_argument("--enablecuda", type=bool, default=False, help="Enable CUDA (True/False)")
	parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"],
						help="Precision (Default fp16)")
	parser.add_argument("--maxnewtokens", type=int, default=1024, help="Maximum new tokens (Default 1024)")
	parser.add_argument("--trustremotecode", type=bool, default=False, help="Trust remote code (True/False)")
	parser.add_argument("--savefile", type=str, default="",
						help="Path for results to be saved (Default project folder)")
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
	results = generate(
		args.model,
		args.sysprompt,
		args.jsonl,
		args.enablecuda,
		args.maxnewtokens,
		precision,
		args.trustremotecode
	)

	# Passes the arguments down for score evaluation
	score = evaluate(results,
					 args.evalsysprompt,
					 args.jsonl,
					 args.evaluator,
					 args.enablecuda,
					 args.maxnewtokens,
					 precision,
					 args.trustremotecode
					 )

	# Prints the score
	print(f"Acc: {score[0]}%\nInst: {score[1]}%\nAcc_err: +-{score[2]}%")

	# Save result
	if not args.savefile == "":
		logging.info(f"Saving result to {args.savefile}.")
		save = {
			"model": args.model,
			"evaluator": args.evaluator,
			"sysprompt": args.sysprompt,
			"evalsysprompt": args.evalsysprompt,
			"acc": score[0],
			"inst": score[1],
			"acc_err": score[2],
		}
		with open(args.savefile, "x") as savefile_json:
			json.dump(save, savefile_json)
