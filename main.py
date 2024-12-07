import os
import json
import time
import uuid
from typing import List, Dict, Any
import faiss
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# Set your OpenAI API key (or do so via env variable)
# openai.api_key = "YOUR_API_KEY_HERE"

# Configurations
MODEL_NAME = "gpt-4o-mini"  # Changed from "gpt-4o-mini" to "gpt-4" or use "gpt-3.5-turbo" if you don't have GPT-4 access
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 20
SIMILARITY_THRESHOLD = 0.9
DATASET_OUTPUT_DIR = "./dataset_output"
os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

#############################################
# Utility Functions
#############################################

def call_llm(messages: List[Dict[str, str]], model: str = MODEL_NAME, max_tokens: int = 1024, temperature=0.7) -> str:
    """Call the LLM and return the response text."""
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def create_embedding(text: str) -> np.ndarray:
    """Create an embedding for the given text using OpenAI embeddings."""
    # In production, consider batch embedding for efficiency.
    embedding_response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    embedding = np.array(embedding_response.data[0].embedding, dtype='float32')
    return embedding

def is_duplicate(embedding: np.ndarray, index: faiss.Index) -> bool:
    """Check if embedding is too similar to any in the index."""
    if index.ntotal == 0:
        return False
    # Perform a similarity search
    k = 1
    D, I = index.search(embedding.reshape(1, -1), k)
    # If the top match is closer than our threshold, consider it a duplicate
    # Note: FAISS gives a distance, for cosine similarity you might store normalized vectors.
    # By default embeddings from OpenAI are not normalized. We can consider Euclidean distances.
    # For simplicity, assume lower distance = more similar. Adjust threshold as needed.
    # You might want to normalize embeddings or use inner product to represent similarity properly.
    # Here, we treat low distance as high similarity. 
    # This threshold logic might need refining.
    distance = D[0][0]
    # This is a rough heuristic. Usually you'd define a threshold after experimentation.
    # For embeddings, you might want to rely on vector norms. Let's assume <0.5 is "too similar".
    return distance < (1 - SIMILARITY_THRESHOLD)

def add_to_index(embedding: np.ndarray, index: faiss.Index):
    """Add embedding to FAISS index."""
    index.add(embedding.reshape(1, -1))

def save_batch_to_disk(batch: List[Dict[str, Any]], category_name: str):
    """Save the generated batch to disk as a JSON lines file."""
    filename = os.path.join(DATASET_OUTPUT_DIR, f"{category_name.replace(' ', '_')}_{uuid.uuid4()}.jsonl")
    with open(filename, 'w', encoding='utf-8') as f:
        for item in batch:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

#############################################
# Entropy Evaluation
#############################################

def evaluate_entropy_with_llm(examples: List[str]) -> float:
    """Use LLM to evaluate entropy/diversity of the generated batch."""
    # Prompt the LLM to assess diversity
    # You might define a schema: 
    # "Rate the diversity of the following examples on a scale from 0.0 (no diversity) to 1.0 (max diversity)"
    # Then parse the LLM's numeric output.
    examples_str = "\n".join(examples)
    messages = [
        {"role": "system", "content": "You are an expert in evaluating textual diversity and entropy."},
        {"role": "user", "content": f"Consider the following {len(examples)} examples:\n\n{examples_str}\n\nRate their diversity on a scale from 0.0 (no diversity) to 1.0 (very high diversity). Only provide a numeric answer."}
    ]
    response = call_llm(messages)
    # Try to parse a float from the response
    try:
        val = float(response)
        return max(0.0, min(val, 1.0))
    except:
        # If parsing fails, return a default value
        return 0.5

#############################################
# Refinement
#############################################

def refine_example(example: str, reason: str) -> str:
    """Ask LLM to refine an example to improve uniqueness or complexity."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant improving data quality."},
        {"role": "user", "content": f"The following example is too similar or low entropy:\n\n{example}\n\nReason: {reason}\nPlease modify it to be more unique and address the mentioned issue."}
    ]
    revised_example = call_llm(messages)
    return revised_example

#############################################
# Roadmap Generation
#############################################

def generate_roadmap(objective: str) -> Dict[str, Any]:
    """Use the Controller LLM to generate a hierarchical roadmap."""
    messages = [
        {"role": "system", "content": "You are a planner that generates hierarchical structures for dataset generation."},
        {"role": "user", "content": f"""Given the objective '{objective}', generate a hierarchical roadmap as JSON.
        Format must be exactly:
        {{
            "top_level_categories": [
                {{
                    "name": "category name",
                    "subcategories": ["subcat1", "subcat2", ...]
                }},
                ...
            ]
        }}"""}
    ]
    roadmap_response = call_llm(messages)
    try:
        roadmap = json.loads(roadmap_response)
        return roadmap
    except json.JSONDecodeError:
        # Fallback structure if JSON parsing fails
        return {
            "top_level_categories": [
                {
                    "name": "PyTorch Model Architecture",
                    "subcategories": ["Model Definition", "Training Loop", "Loss Functions"]
                },
                {
                    "name": "Data Processing",
                    "subcategories": ["Data Loading", "Preprocessing", "Augmentation"]
                }
            ]
        }

#############################################
# Generation Logic
#############################################

def generate_examples_for_subcategory(subcategory_name: str, 
                                    index: faiss.Index, 
                                    prompt_modifier: str = "") -> List[Dict[str, Any]]:
    """Generate a batch of examples for a given subcategory."""
    base_prompt = f"""Generate {BATCH_SIZE} distinct Python code examples for '{subcategory_name}'.
    Each example should be complete and self-contained.
    Format each example with a clear separator like '---' between examples.
    Make examples diverse and production-quality."""
    
    if prompt_modifier:
        base_prompt += f"\nAdditional requirements: {prompt_modifier}"

    messages = [
        {"role": "system", "content": "You are an expert PyTorch developer creating training examples."},
        {"role": "user", "content": base_prompt}
    ]
    
    raw_text = call_llm(messages, max_tokens=2048)  # Increased token limit for code
    batch_examples = [ex.strip() for ex in raw_text.split('---') if ex.strip()][:BATCH_SIZE]

    final_examples = []
    for ex in batch_examples:
        ex_embedding = create_embedding(ex)
        if is_duplicate(ex_embedding, index):
            # If duplicate, try to refine it
            messages = [
                {"role": "system", "content": "You are an expert at modifying code to be more unique."},
                {"role": "user", "content": f"Make this code example more unique while preserving functionality:\n\n{ex}"}
            ]
            ex = call_llm(messages, max_tokens=1024)
            ex_embedding = create_embedding(ex)

        final_examples.append({
            "example": ex,
            "subcategory": subcategory_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        add_to_index(ex_embedding, index)
    
    return final_examples

#############################################
# Main Orchestration
#############################################

def main():
    objective = "Create a dataset of PyTorch AI model code examples for training a language model"
    print("Generating roadmap...")
    roadmap = generate_roadmap(objective)
    
    # Initialize FAISS index
    dimension = 1536  # dimension of text-embedding-ada-002 embeddings
    index = faiss.IndexFlatL2(dimension)
    
    total_examples = 0
    
    for category in roadmap.get("top_level_categories", []):
        category_name = category["name"]
        print(f"\nProcessing category: {category_name}")
        
        for subcat in category["subcategories"]:
            print(f"  Generating examples for subcategory: {subcat}")
            try:
                batch_data = generate_examples_for_subcategory(subcat, index)
                total_examples += len(batch_data)
                save_batch_to_disk(batch_data, category_name)
                print(f"    Generated {len(batch_data)} examples")
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"    Error generating examples for {subcat}: {str(e)}")
                continue
    
    print(f"\nDataset generation complete. Total examples generated: {total_examples}")

if __name__ == "__main__":
    main()
