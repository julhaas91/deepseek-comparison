# AI Model Evaluation Script

This repository contains an script to compare the results of two locally distilled versions of DeepSeek R1 (7B and 32B paramater versions).

## Overview

The script consists of the following steps:

1. **Local Inference**: Execute inference on two AI models using predefined questions from `examples.json`. Output stored in separate response files.

2. **Response Consolidation**: Merge responses from both models into `responses_merged.txt`, aligning answers to corresponding questions.

3. **Evaluation**: Utilize `gemini-1.5-pro` (or a local model via `evaluate_local()`) to assess merged responses, comparing model performance and assigning quality scores.

4. **Analysis**: Process evaluation results to generate aggregated metrics and performance insights for each model.

## Run script locally

1. Install [uv](https://github.com/astral-sh/uv)
2. Run the distilled models locally on your machine (I recommend using [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/)). Ensure the model is configured to accept requests at `http://localhost:1234/v1/chat/completions`.
3. Add your Gemini key to `.env`
4. Run `uv run python main.py` in the terminal

## Data and Results
- [`questions.json`](questions.json): Questions used for model evaluation 

- [`responses_deepseek-r1-distill-qwen-7b.jsonl`](responses_deepseek-r1-distill-qwen-7b.jsonl): Answers from DeepSeek R1 7B model

- [`responses_deepseek-r1-32b.jsonl`](responses_deepseek-r1-32b.jsonl): Answers from DeepSeek R1 32B model

- [`responses_merged.txt`](responses_merged.txt): Combined responses from both models

- [`evaluation_results_gemini_pro.txt`](evaluation_results_gemini_pro.txt): Evaluation results by Google's Gemini Pro 1.5

## Disclaimer
This script is intended for experimental purposes only. The results obtained should not be considered representative or conclusive. Please note: The evaluation questions used have not been optimized or validated. Results may contain errors or inaccuracies. Findings may not accurately reflect the quality of answers in other use cases or real-world scenarios. Users should exercise caution when interpreting or applying these results.

Use this script and its outputs at your own discretion, understanding its experimental nature and potential limitations.