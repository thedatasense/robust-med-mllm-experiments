#!/usr/bin/env python3
import argparse
import json
import os
import time
from time import sleep
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from dataclasses import dataclass

@dataclass
class LlamaVisionConfig:
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    max_new_tokens: int = 1600  # Adjust this value as needed
    source_dir: str = "/NAS/bsada1/coderepo/CARES/data/Harvard-FairVLMed/images"  # Default source dir

class LlamaVisionController:
    def __init__(self, config: LlamaVisionConfig):
        self.config = config
        print(f"Loading Llama Vision model: {self.config.model_id}")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Tie weights for potential performance benefits
        try:
            self.model.tie_weights()
        except Exception as e:
            print("Warning: Could not tie weights:", e)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        # Print one parameter's device for verification (optional)
        for name, param in self.model.named_parameters():
            print(f"Parameter '{name}' is on device: {param.device}")
            break

        print("Model and processor loaded successfully.")

    def process_query(self, text: str, image_name: str) -> str:
        # Build the full image path
        image_path = os.path.join(self.config.source_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")

        # Construct conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            }
        ]
        # Prepare the input text using the processor's chat template
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        print("Prepared input text for the model.")

        # Process the image and text into model inputs
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        print("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        # Decode the model output
        response = self.processor.decode(output[0])
        print("Inference completed. Response:", response)
        return response

def main():
    parser = argparse.ArgumentParser(
        description="Run Llama Vision inference on a JSONL file and save responses."
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSONL file')
    parser.add_argument('--source_dir', type=str, default="/NAS/bsada1/coderepo/CARES/data/Harvard-FairVLMed/images",
                        help='Directory where images are located')
    args = parser.parse_args()

    input_file = args.input_file
    # Create output file name by appending "_llama-ans" before the extension.
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_llama-ans{ext}"

    # Read already processed question_ids from the output file (if it exists)
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    record = json.loads(line)
                    processed_ids.add(record.get("question_id"))
                except Exception:
                    continue
    print(f"Found {len(processed_ids)} already processed records.")

    # Initialize the Llama Vision controller
    config = LlamaVisionConfig(source_dir=args.source_dir)
    controller = LlamaVisionController(config)

    line_number = 0
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "a", encoding="utf-8") as fout:
        for line in fin:
            line_number += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question_id = record.get("question_id")
                # Skip already processed records.
                if question_id in processed_ids:
                    print(f"Skipping question_id {question_id} (line {line_number}) already processed.")
                    continue

                image_file = record.get("image")
                question_text = record.get("text")
                print(f"\nProcessing question_id {question_id} (line {line_number}) with image: {image_file}")

                answer = controller.process_query(question_text, image_file)

                output_record = {
                    "question_id": question_id,
                    "image": image_file,
                    "question": question_text,
                    "answer": answer
                }
                fout.write(json.dumps(output_record) + "\n")
                fout.flush()  # Flush after each record so progress is saved.
                print(f"Finished processing question_id {question_id} (line {line_number}).")
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
                error_record = {
                    "question_id": record.get("question_id") if 'record' in locals() else None,
                    "image": record.get("image") if 'record' in locals() else None,
                    "question": record.get("text") if 'record' in locals() else None,
                    "answer": f"Error: {e}"
                }
                fout.write(json.dumps(error_record) + "\n")
                fout.flush()
            sleep(0.5)  # Optional delay between processing records

    print(f"\nAll done. Results have been written to {output_file}")

if __name__ == "__main__":
    main()
