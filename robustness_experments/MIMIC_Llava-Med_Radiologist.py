#!/usr/bin/env python3
import json
import os
import hashlib
from datetime import datetime
import re
import gc
import time
import yaml
import requests
from PIL import Image
import base64
import torch
from typing import Optional
from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.engine import create_engine
import pandas as pd

# Import conversation objects as used by the Gradio interface.
from llava.conversation import default_conversation, conv_templates

source_folder='/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files'

# Configuration handling
def get_from_cnfg(key_path, file_path="/home/bsada1/config.yaml"):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            
        keys = key_path.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value
            
    except FileNotFoundError:
        print(f"File {file_path} not found")
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
    except KeyError:
        print(f"Key path {key_path} not found")
    except Exception as e:
        print(f"Error: {e}")
    return None

# Database connection setup
DB_URL = get_from_cnfg("cd_url")
engine = create_engine(DB_URL)

# Memory management utilities
def get_gpu_memory_usage():
    """
    Get current GPU memory usage in MB
    Returns: Memory allocated and memory cached
    """
    # Get memory in bytes and convert to MB
    memory_allocated = torch.cuda.memory_allocated() / 1024**2
    memory_cached = torch.cuda.memory_reserved() / 1024**2
    return memory_allocated, memory_cached

def log_memory_usage(step: str):
    """
    Log current GPU memory usage with step information
    Args:
        step: Description of current step
    """
    allocated, cached = get_gpu_memory_usage()
    print(f"Memory Usage {step}:")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Cached: {cached:.2f} MB")
    print("-" * 50)

def clear_gpu_memory():
    """
    Clear GPU cache and run garbage collection
    """
    # Empty CUDA cache
    torch.cuda.empty_cache()
    # Run Python garbage collection
    gc.collect()

# Data fetching
def fetch_generation_data(engine):
    import pandas as pd
    import re
    from sqlalchemy import text
    from sqlalchemy.dialects.postgresql.base import PGDialect
    def fake_get_server_version_info(self, connection):
        version_str = connection.execute(text("SELECT version()")).scalar()
        match = re.search(r'v(\d+)\.(\d+)\.(\d+)', version_str)
        if match:
            return tuple(map(int, match.groups()))
        return (13, 0, 0)
    PGDialect._get_server_version_info = fake_get_server_version_info
    query = f"SELECT id,question_id,condition as question_type, text as question,answer as ground_truth,image from mimic_all_qns; "
    return pd.read_sql(query, con=engine)

# Duplicate checking
def check_duplicate(engine, uid, question_id, question, question_category, model_name, image_link):
    query = text("""
        SELECT 1 FROM model_responses_r2
        WHERE 
        uid = :uid
        AND question_id = :question_id and 
        question = :question
          AND question_category = :question_category
          AND model_name = :model_name
          AND image_link = :image_link
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {
            "uid": uid,
            "question_id": question_id,
            "question": question,
            "question_category": question_category,
            "model_name": model_name,
            "image_link": image_link
        }).fetchone()
    return result is not None

# Response storage
def insert_model_response(engine, uid, question_id, question, question_category, actual_answer, model_name, model_answer, image_link):
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text("""
                INSERT INTO model_responses_r2
                (uid,question_id,question, question_category, actual_answer, model_name, model_answer, image_link)
                VALUES (:uid,:question_id,:question, :question_category, :actual_answer, :model_name, :model_answer, :image_link)
            """), {
                "uid": uid,
                "question_id": question_id,
                "question": question,
                "question_category": question_category,
                "actual_answer": actual_answer,
                "model_name": model_name,
                "model_answer": model_answer,
                "image_link": image_link
            })
            trans.commit()  # Commit the transaction
        except Exception as e:
            trans.rollback()
            raise e

@dataclass
class ControllerConfig:
    # Adjust the controller URL/port if needed.
    controller_url: str = "http://localhost:10000"
    model_name: str = "llava-med-v1.5-mistral-7b"
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 1536  # Increase token limit for more detailed output.
    serve_dir: str = "serve_images"
    source_dir: str = "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files/"

class LLaVAController:
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        print(f"Initialized controller with URL: {self.config.controller_url}")
        print(f"Using model: {self.config.model_name}")
        os.makedirs(self.config.serve_dir, exist_ok=True)

    def _get_worker_address(self) -> str:
        print("\nGetting worker address...")
        response = requests.post(
            f"{self.config.controller_url}/get_worker_address",
            json={"model": self.config.model_name}
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get worker address: {response.text}")
        worker_addr = response.json()["address"]
        print(f"Got worker address: {worker_addr}")
        return worker_addr

    def _prepare_image(self, image_path: str) -> tuple[str, str]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        print(f"\nLoading image from: {image_path}")
        image = Image.open(image_path)
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        current_time = datetime.now()
        date_dir = os.path.join(
            self.config.serve_dir,
            f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d}"
        )
        os.makedirs(date_dir, exist_ok=True)
        serve_path = os.path.join(date_dir, f"{image_hash}.jpg")
        if not os.path.isfile(serve_path):
            print(f"Processing image from {image_path}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            image.save(serve_path, 'JPEG', quality=95)
            print(f"Saved processed image to: {serve_path}")
        else:
            print(f"Using existing processed image: {serve_path}")
        return image_hash, serve_path

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_llava(self, text: str, image_path: str) -> str:
        """
        Process a query by building a prompt using the Gradio conversation template,
        appending the user's query, and sending it along with the image to the worker.
        """
        try:
            print("\nProcessing query:", text)
            text = text[:1200]  # Enforce a cut-off if needed.
            image_hash, serve_path = self._prepare_image(image_path)
            encoded_image = self._encode_image_to_base64(serve_path)
            worker_addr = self._get_worker_address()

            # Ensure the query contains an <image> token.
            if '<image>' not in text:
                text = text + '\n<image>'

            # Build the conversation state following the Gradio method.
            state = default_conversation.copy()
            state.append_message(state.roles[0], text)
            state.append_message(state.roles[1], None)
            if len(state.messages) == state.offset + 2:
                template_name = "mistral_instruct"
                new_state = conv_templates[template_name].copy()
                new_state.append_message(new_state.roles[0], state.messages[-2][1])
                new_state.append_message(new_state.roles[1], None)
                state = new_state
            prompt = state.get_prompt()

            print("\nFinal prompt (from conversation template):", prompt)
            print(f"Number of <image> tokens in prompt: {prompt.count('<image>')}")

            # Use a simple stop token.
            stop_token = "\n"
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_tokens,
                "stop": stop_token,
                "images": [encoded_image],
                "text": prompt  # Extra key in case the generate() method checks for it.
            }
            print("\nSending request to worker with image data...")
            response = requests.post(
                f"{worker_addr}/worker_generate_stream",
                json=payload,
                headers={"User-Agent": "LLaVA Client"},
                stream=True
            )
            if response.status_code != 200:
                raise RuntimeError(f"Worker error: {response.text}")

            full_response = ""
            for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if line:
                    data = json.loads(line.decode())
                    if data["error_code"] == 0:
                        new_text = data["text"][len(prompt):].strip()
                        full_response = new_text
                    else:
                        raise RuntimeError(f"Generation error {data['error_code']}: {data['text']}")
            
            # Clean up memory after processing
            clear_gpu_memory()
            
            print("\nFinal response from model:", full_response)
            return full_response

        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise

def main():
    # Initialize the controller
    controller = LLaVAController()
    model_id = "llava-med-v1.5-mistral-7b"
    error_list = set()
    
    # Fetch all data from the database
    data_rows = fetch_generation_data(engine)
    
    # Process each row
    for index, row in data_rows.iterrows():
        uid = row["id"]
        question_id = row["question_id"]
        question_category = row["question_type"]
        question = row["question"]
        actual_answer = row["ground_truth"]
        image_link = source_folder + os.path.sep + row["image"]  # This should be the full path to the image
        
        # Check if this combination has already been processed
        if check_duplicate(engine, uid, str(question_id), question, question_category, model_id, image_link):
            print(f"Duplicate record found for question: {question}. Skipping generation.")
            continue
        
        try:
            # Generate answer
            generated_answer = controller.generate_llava(question, image_link)
            
            # Add some delay between requests
            time.sleep(5)
            
            # Print results
            print(f"{model_id} : {generated_answer}")
            print(f"GT: {actual_answer}")
            
            # Insert into database
            insert_model_response(engine, uid, question_id, question, question_category, 
                                  actual_answer, model_id, generated_answer, image_link)
            print('--------------------------------')
            
        except Exception as e:
            error_list.add(f"{str(uid)}-{str(question_id)}-{str(question_category)}")
            print(f"Error processing {question_id}: {str(e)}")
            continue

    # Print any errors that occurred
    if error_list:
        print(f"Errors occurred for {len(error_list)} items:")
        for err in error_list:
            print(f"  - {err}")

if __name__ == "__main__":
    main()