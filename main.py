"""
Module for running local inference with language models, merging and evaluating responses.

This module provides functionality to run inference with local language models,
merge responses from different models, and evaluate their performance using
external evaluation models via Gemini API or local evaluation systems.
"""

import subprocess
import json
import time
import os
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd


def run_local_inference(
    model_name: str, questions_file_path: str, output_file_path: str, delay: int = 20
) -> None:
    """
    Run inference locally for a specific model and save responses to a JSONL file.

    Args:
        model_name: Name of the model to use for inference
        questions_file_path: Path to the JSON file containing questions
        output_file_path: Path to save the responses
        delay: Delay between requests in seconds (default: 20)
    """
    print(f"Starting inference with model: {model_name}")

    system_prompt = """
    You are a highly knowledgeable AI assistant with expertise across multiple domains. 
    Your primary function is to provide accurate, comprehensive, and insightful answers 
    to questions on a wide range of topics. Please adhere to the following guidelines:

    1. Provide detailed, well-structured responses that are informative and easy to understand.
    2. Use credible information and cite sources when appropriate.
    3. If a question is unclear or lacks context, ask for clarification before answering.
    4. When dealing with complex topics, break down your explanation into digestible parts.
    5. If you're unsure about an answer, acknowledge the limitations of your knowledge.
    6. Offer practical advice or actionable steps when relevant to the question.
    7. Maintain a professional and impartial tone while being engaging and approachable.
    8. Respect ethical considerations and avoid harmful or inappropriate content.
    9. Tailor your language and complexity level to suit the apparent knowledge level of the user.
    10. Be prepared to explain technical terms or concepts if needed.

    Your goal is to provide the most helpful and accurate information possible for each query you receive.
    """

    with open(questions_file_path, "r") as f:
        questions_data: Dict[str, Any] = json.load(f)

    with open(output_file_path, "w") as f:
        for category, questions in questions_data["categories"].items():
            for question in questions:
                print(
                    f"Processing question ID {question['id']}: {question['question']}"
                )

                # Prepare request data
                data: Dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Hello! How can I help you today?",
                        },
                        {"role": "user", "content": question["question"]},
                    ],
                    "temperature": 0.7,
                    "max_tokens": -1,
                    "stream": False,
                }

                # Send request and measure time
                start_time: float = time.time()
                response: str = send_request(
                    user_message=question["question"],
                    system_prompt=system_prompt,
                    data=data,
                )
                end_time: float = time.time()
                response_time: float = end_time - start_time

                # Create and save response entry
                response_entry: Dict[str, Any] = {
                    "category": category,
                    "id": question["id"],
                    "question": question["question"],
                    "response": response,
                    "response_time": response_time,
                }

                f.write(json.dumps(response_entry) + "\n")
                print(f"Response for ID {question['id']} saved.\n")

                # Wait before sending the next request
                if delay > 0:
                    time.sleep(delay)

    print(f"All responses have been processed and saved to {output_file_path}")


def send_request(
    user_message: str, system_prompt: str, data: Dict[str, Any] = None
) -> str:
    """
    Send a request to the local inference server with system prompt context.

    Args:
        user_message: The user message to process
        system_prompt: The system prompt to provide context
        data: Request data as a dictionary (default: None)

    Returns:
        Server response as a string
    """
    print("Sending request to local model.")
    
    if data is None:
        data = {}

    data["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    command: List[str] = [
        "curl",
        "http://localhost:1234/v1/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(data),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    print("Received response from local model.")
    return result.stdout


def merge_responses(
    file1_path: str, file2_path: str, output_file_path: str
) -> List[Dict[str, Any]]:
    """
    Merge responses from two JSONL files based on matching question IDs.

    Args:
        file1_path: Path to the first JSONL file
        file2_path: Path to the second JSONL file
        output_file_path: Path to save the merged responses (as TXT)

    Returns:
        List of merged response entries
    """
    print(f"Merging responses from {file1_path} and {file2_path}")

    # Load JSONL files into dictionaries based on question ID
    data1: Dict[Union[str, int], Dict[str, Any]] = load_jsonl(file1_path)
    data2: Dict[Union[str, int], Dict[str, Any]] = load_jsonl(file2_path)

    # Extract model names from filenames for better labeling
    model1_name: str = (
        os.path.basename(file1_path).replace("responses_", "").replace(".jsonl", "")
    )
    model2_name: str = (
        os.path.basename(file2_path).replace("responses_", "").replace(".jsonl", "")
    )

    # Merge responses based on matching IDs
    merged_data: List[Dict[str, Any]] = []
    for id_key in data1:
        if id_key in data2:  # Ensure both models answered the same questions
            merged_entry: Dict[str, Any] = {
                "id": id_key,
                "category": data1[id_key]["category"],
                "question": data1[id_key]["question"],
                f"{model1_name}_response": data1[id_key]["response"],
                f"{model2_name}_response": data2[id_key]["response"],
            }
            merged_data.append(merged_entry)

    # Write merged output to a new TXT file (one JSON per line)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for entry in merged_data:
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Merged data saved to {output_file_path}")
    return merged_data


def load_jsonl(file_path: str) -> Dict[Union[str, int], Dict[str, Any]]:
    """
    Load a JSONL file into a dictionary keyed by question ID.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Dictionary with ID keys and entry values
    """
    data_dict: Dict[Union[str, int], Dict[str, Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry: Dict[str, Any] = json.loads(line)
            data_dict[entry["id"]] = entry
    return data_dict


def setup_gemini_client() -> genai.Client:
    """
    Setup and return the Gemini API client.

    Returns:
        Configured Gemini API client
    """
    load_dotenv()
    gemini_key: Optional[str] = os.getenv("GEMINI_KEY")
    return genai.Client(api_key=gemini_key)


def evaluate_with_gemini(
    client: genai.Client,
    input_file_path: str,
    output_file_path: str,
    model: str = "gemini-1.5-pro",
    temperature: float = 0.3,
) -> None:
    """
    Evaluate AI responses by comparing two models' responses using Gemini.

    Args:
        client: The Gemini API client
        input_file_path: Path to the merged TXT file
        output_file_path: Path to save the evaluation results
        model: The Gemini model to use for evaluation (default: "gemini-1.5-pro")
        temperature: The temperature setting for generation (default: 0.3)
    """
    print(f"Evaluating responses with Gemini model {model}")

    # System instruction for the evaluation
    system_prompt: str = """
    # AI Response Evaluation System

    You are a highly skilled evaluator of AI-generated content with expertise in natural 
    language understanding, reasoning, and factual accuracy assessment. Your task is to 
    objectively compare pairs of AI responses to the same set of questions.

    ## Input Format
    You will receive a TXT file containing merged AI responses:
    - The file contains responses from two different models to the same questions.
    - Each line follows this structure: {"id": number, "category": "difficulty_level", 
      "question": "text", "model_a_response": JSON_string, "model_b_response": JSON_string}
    - The "model_a_response" and "model_b_response" fields contain a nested JSON with model 
      information and the actual response content in choices[0].message.content
    - The actual AI response is within the "content" field, sometimes including a <think> section

    ## Evaluation Instructions
    For each question-response pair:
    1. **Extract and read both actual responses carefully**, focusing on the content after any <think> sections
    2. **Consider these key quality factors**:
       - Factual accuracy and correctness of the answer
       - Comprehensiveness and depth of explanation
       - Clarity and coherence of reasoning
       - Relevance to the question asked
       - Presentation (formatting, mathematical notation, organization)

    3. **Assign a quality score** between 0 and 1 to each response where:
       - 0.0-0.2: Poor (contains significant errors, misses the question's point)
       - 0.3-0.5: Fair (partially addresses the question with some issues)
       - 0.6-0.7: Good (adequately addresses the question with minor issues)
       - 0.8-0.9: Excellent (comprehensive, accurate, well-reasoned)
       - 1.0: Outstanding (exceptional in all aspects)

    Compare and score the two model responses. Ensure a clear distinction between their scores,
    avoiding ties. If initial evaluation results in equal scores, conduct a more nuanced 
    reassessment to identify subtle differences and assign slightly different scores. The goal
    is to always have one model score higher than the other, even if the difference is minimal.

    4. **Write a concise justification** (1 sentence) explaining your scoring decision, 
       highlighting key strengths or weaknesses

    ## Output Format
    Generate a single JSONL file where each line contains:
    ```
    {
      "question_id": <int>,
      "question": <string>,
      "{name of first model}_score": <int>,
      "{name of first model}_justification": <string>,
      "{name of second model}_score": <int>,
      "{name of second model}_justification": <string>
    }
    ```

    ## Guidelines
    - Be impartial and objective in your assessment
    - Disregard the <think> sections when evaluating responses
    - Consider the correctness of mathematical solutions and explanations
    - Evaluate based on the actual answer and explanation provided
    - Look for clear, step-by-step reasoning in responses to math problems
    - Maintain consistent evaluation criteria across all question pairs
    - Consider both the correctness of the final answer and the quality of explanation
    """

    # Upload the merged response file
    file_upload = client.files.upload(
        file=input_file_path, config={"mime_type": "text/plain"}
    )

    # Generate content for evaluation
    response = client.models.generate_content(
        model=model,
        contents=[file_upload],
        config=types.GenerateContentConfig(
            SYSTEM_PROMPT=system_prompt,
            temperature=temperature,
        ),
    )

    # Save the evaluation results
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(str(response.text))
        print(f"Evaluation results saved to {output_file_path}")


def evaluate_local(
    input_file_path: str,
    output_file_path: str,
    model1_name: str,
    model2_name: str,
    delay: int = 3,
    temperature: float = 0.3,
) -> None:
    """
    Evaluate AI responses with a local model.

    Args:
        input_file_path: Path to the merged responses file
        output_file_path: Path to save the evaluation results
        model1_name: Name of the first model being compared
        model2_name: Name of the second model being compared
        delay: Delay between requests in seconds (default: 3)
        temperature: The temperature setting for generation (default: 0.3)
    """
    print("Evaluating responses with local model.")

    system_prompt: str = f"""
    # AI Response Evaluator

    You are an expert evaluator comparing two AI-generated responses to a single question.

    ## Input Format
    You will receive a JSON object with this structure:
    {{
    "id": "<int>",
    "category": "<string>",
    "question": "<string>",
    "model1_response": "<JSON_string>",
    "model2_response": "<JSON_string>"
    }}

    The actual responses are in the "content" field of choices[0].message within each model's response.

    ## Evaluation Instructions
    1. Extract and analyze both responses, ignoring any <think> sections.
    2. Evaluate based on: accuracy, comprehensiveness, clarity, relevance, and presentation.
    3. Score each response from 0 to 1:
    0.0-0.2: Poor | 0.3-0.5: Fair | 0.6-0.7: Good | 0.8-0.9: Excellent | 1.0: Outstanding
    Ensure scores are distinct, even if only slightly.
    4. Provide a brief justification (1 sentence) for each score.

    ## Output Format
    Respond with a JSON object:
    {{
    "question_id": "<int>",
    "question": "<string>",
    "{model1_name}_score": "<float>",
    "{model1_name}_justification": "<string>",
    "{model2_name}_score": "<float>",
    "{model2_name}_justification": "<string>"
    }}

    ## Guidelines
    - Be objective and consistent in your evaluation.
    - Focus on the quality of the answer and explanation.
    - For math problems, consider both the final answer and the reasoning process.
    - Maintain the same evaluation criteria for all questions.
    """
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                try:
                    response: str = send_request(
                        user_message=line.strip(),
                        system_prompt=system_prompt,
                        data={"temperature": temperature},
                    )
                    response_json = json.loads(response)
                    parsed_response = response_json["choices"][0]["message"]["content"]

                    output_file.write(json.dumps(parsed_response) + "\n")
                    print("Model response saved.\n")

                    if delay > 0:
                        time.sleep(delay)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    continue


def analyze_model_comparison(
    input_file_path: str, model1_name: str, model2_name: str
) -> None:
    """
    Analyzes comparison results of two models, providing aggregated performance metrics.

    Args:
        input_file_path: Path to the JSONL file containing the comparison data
        model1_name: Name of the first model being compared
        model2_name: Name of the second model being compared
    """
    # Load data from JSONL file
    data = []
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                # print(f"Error decoding JSON: {e}")
                continue  # Skip to the next line

    # Create a Pandas DataFrame
    df = pd.DataFrame(data)

    # Calculate average scores
    avg_1_score = df[f"{model1_name}_score"].mean()
    avg_2_score = df[f"{model2_name}_score"].mean()

    # Determine which model performed better on each question
    df["better_model"] = df.apply(
        lambda row: f"{model1_name}"
        if row[f"{model1_name}_score"] > row[f"{model2_name}_score"]
        else (
            f"{model2_name}"
            if row[f"{model2_name}_score"] > row[f"{model1_name}_score"]
            else "Tie"
        ),
        axis=1,
    )

    # Count the number of questions where each model performed better
    count_a_better = (df["better_model"] == f"{model1_name}").sum()
    count_b_better = (df["better_model"] == f"{model2_name}").sum()
    count_ties = (df["better_model"] == "Tie").sum()

    print("## Model Comparison Analysis:")
    print(f"Average Model '{model1_name}' Score: {avg_1_score:.3f}")
    print(f"Average Model '{model2_name}' Score: {avg_2_score:.3f}")
    print(
        f"Number of questions where Model '{model1_name}' performed better: {count_a_better}"
    )
    print(
        f"Number of questions where Model '{model2_name}' performed better: {count_b_better}"
    )
    print(f"Number of questions where the models tied: {count_ties}")

    # Additional descriptive metrics (example: score difference)
    df["score_difference"] = df[f"{model1_name}_score"] - df[f"{model2_name}_score"]
    avg_score_difference = df["score_difference"].mean()
    std_score_difference = df["score_difference"].std()

    print("\n## Additional Metrics:")
    print(
        f"Average Score Difference ('{model1_name}' - '{model2_name}'): {avg_score_difference:.3f}"
    )
    print(f"Standard Deviation of Score Difference: {std_score_difference:.3f}")

    # Determine which model performed better overall based on average scores
    if avg_1_score > avg_2_score:
        better_model = f"Model '{model1_name}'"
    elif avg_2_score > avg_1_score:
        better_model = f"Model '{model2_name}'"
    else:
        better_model = "Both models performed equally well"

    print("\n## Final Statement:")
    print(f"Based on the average scores, {better_model} gave better results in total.")


def main() -> None:
    """
    Main function to orchestrate the complete evaluation workflow.
    """
    # Configuration
    questions_file_path: str = "questions.json"
    model1_name: str = "deepseek-r1-distill-qwen-7b"
    model2_name: str = "deepseek-r1-32b"
    responses1_file: str = "responses_deepseek-r1-distill-qwen-7b.jsonl"
    responses2_file: str = "responses_deepseek-r1-32b.jsonl"
    merged_file: str = "responses_merged.txt"
    evaluation_file: str = "evaluation_results_gemini_pro.txt"

    # Uncomment these sections as needed
    # # Step 1: Run inference for both models
    # run_local_inference(model1_name, questions_file_path, responses1_file)
    # run_local_inference(model2_name, questions_file_path, responses2_file)

    # # Step 2: Merge the responses
    # merge_responses(responses1_file, responses2_file, merged_file)

    # # Step 3: Evaluate with Gemini
    # client: genai.Client = setup_gemini_client()
    # evaluate_with_gemini(client, merged_file, evaluation_file)
    
    # # Step 3: Evaluate locally
    # evaluate_local(
    #     input_file_path=merged_file,
    #     output_file_path=evaluation_file,
    #     model1_name=model1_name,
    #     model2_name=model2_name,
    # )

    # Step 4: Analyze the results
    analyze_model_comparison(evaluation_file, model1_name, model2_name)


if __name__ == "__main__":
    main()
