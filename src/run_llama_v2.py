#!/usr/bin/env python3
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from dataclasses import dataclass
from typing import List, Dict, Set
import torch.cuda
from queue import Queue
from threading import Lock
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class LlamaVisionConfig:
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    max_new_tokens: int = 1600
    source_dir: str = "/NAS/bsada1/coderepo/CARES/data/Harvard-FairVLMed/images"
    batch_size: int = 4  # Process multiple images in parallel
    num_workers: int = 2  # Number of worker threads for image loading

class LlamaVisionController:
    def __init__(self, config: LlamaVisionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading Llama Vision model: {self.config.model_id}")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # Use nested float 4 for better accuracy
        )

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            device_map="auto",
            quantization_config=quantization_config
        )

        # Set to eval mode
        self.model.eval()

        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
            # Note: We're not using DataParallel as it doesn't support generate()
            # Instead, we'll let device_map="auto" handle multi-GPU distribution
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        # Create thread-safe queues and locks
        self.image_queue = Queue(maxsize=config.batch_size * 2)
        self.output_lock = Lock()

        logging.info("Model and processor loaded successfully.")

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image in a separate thread"""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {e}")
            return None

    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def process_batch(self, batch_data: List[Dict]) -> List[str]:
        """Process a batch of images and questions"""
        try:
            images = []
            texts = []

            for item in batch_data:
                image = item['image']
                text = item['text']

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

                images.append(image)
                texts.append(input_text)

            # Process inputs in smaller sub-batches if needed
            sub_batch_size = min(len(images), self.config.batch_size)
            all_outputs = []

            for i in range(0, len(images), sub_batch_size):
                sub_images = images[i:i + sub_batch_size]
                sub_texts = texts[i:i + sub_batch_size]

                # Batch process inputs
                inputs = self.processor(
                    sub_images,
                    sub_texts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                # Free up memory
                torch.cuda.empty_cache()

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        num_beams=2,  # Reduced from 4 to improve memory usage
                        early_stopping=True
                    )
                    all_outputs.extend(outputs)

            return [self.processor.decode(output) for output in all_outputs]
        except Exception as e:
            logging.error(f"Error in process_batch: {str(e)}")
            raise

    def process_records(self, records: List[Dict], processed_ids: Set[str], output_file: str):
        """Process multiple records efficiently"""
        batch = []
        batch_records = []

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_image = {}

            for record in records:
                question_id = record.get("question_id")
                if question_id in processed_ids:
                    continue

                image_path = os.path.join(self.config.source_dir, record.get("image"))
                future = executor.submit(self.preprocess_image, image_path)
                future_to_image[future] = record

                if len(future_to_image) >= self.config.batch_size:
                    self._process_futures_batch(future_to_image, batch, batch_records)

                    if len(batch) >= self.config.batch_size:
                        self._process_and_save_batch(batch, batch_records, output_file)
                        batch = []
                        batch_records = []

            # Process remaining items
            if future_to_image:
                self._process_futures_batch(future_to_image, batch, batch_records)
            if batch:
                self._process_and_save_batch(batch, batch_records, output_file)

    def _process_futures_batch(self, future_to_image, batch, batch_records):
        """Process a batch of futures from the thread pool"""
        for future in future_to_image:
            record = future_to_image[future]
            try:
                image = future.result()
                if image:
                    batch.append({
                        'image': image,
                        'text': record.get("text")
                    })
                    batch_records.append(record)
            except Exception as e:
                logging.error(f"Error processing image for question_id {record.get('question_id')}: {e}")

    def _process_and_save_batch(self, batch, batch_records, output_file):
        """Process a batch and save results"""
        try:
            answers = self.process_batch(batch)

            with self.output_lock:
                with open(output_file, "a", encoding="utf-8") as fout:
                    for record, answer in zip(batch_records, answers):
                        output_record = {
                            "question_id": record.get("question_id"),
                            "image": record.get("image"),
                            "question": record.get("text"),
                            "answer": answer
                        }
                        fout.write(json.dumps(output_record) + "\n")
                        fout.flush()
        except Exception as e:
            logging.error(f"Error processing batch: {e}")

def main():
    input_file = "/NAS/bsada1/coderepo/CARES/data/Harvard-FairVLMed/fundus_factuality_proc_jb-1.jsonl"
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_llama-ans{ext}"

    # Read processed IDs
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    record = json.loads(line)
                    processed_ids.add(record.get("question_id"))
                except Exception:
                    continue
    logging.info(f"Found {len(processed_ids)} already processed records.")

    # Initialize controller
    config = LlamaVisionConfig()
    controller = LlamaVisionController(config)

    # Read all records first
    records = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except Exception as e:
                    logging.error(f"Error parsing record: {e}")

    # Process records in batches
    controller.process_records(records, processed_ids, output_file)
    logging.info(f"All done. Results have been written to {output_file}")

if __name__ == "__main__":
    main()