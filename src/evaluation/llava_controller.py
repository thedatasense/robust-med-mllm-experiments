import json
import os
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import requests
from PIL import Image
import base64

# Import the conversation objects used by the Gradio interface.
from llava.conversation import default_conversation, conv_templates

@dataclass
class ControllerConfig:
    # Make sure the controller URL/port match your running instance.
    controller_url: str = "http://localhost:10000"  # Adjust if necessary
    model_name: str = "llava-med-v1.5-mistral-7b"
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 1536  # Increase token limit for more details.
    serve_dir: str = "serve_images"
    source_dir: str = "/NAS/bsada1/coderepo/CARES/data/Harvard-FairVLMed/images"

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

    def _prepare_image(self, image_name: str) -> tuple[str, str]:
        source_path = os.path.join(self.config.source_dir, image_name)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Image not found: {source_path}")
        print(f"\nLoading image from: {source_path}")
        image = Image.open(source_path)
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        current_time = datetime.now()
        date_dir = os.path.join(
            self.config.serve_dir,
            f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d}"
        )
        os.makedirs(date_dir, exist_ok=True)
        serve_path = os.path.join(date_dir, f"{image_hash}.jpg")
        if not os.path.isfile(serve_path):
            print(f"Processing image {image_name}")
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

    def process_query(self, text: str, image_name: str) -> str:
        """
        Process a query by building a prompt using the Gradio conversation template,
        appending the user's query, and sending it along with the image to the worker.
        """
        try:
            print("\nProcessing query:", text)
            text = text[:1200]  # Enforce a cut-off if needed.
            image_hash, serve_path = self._prepare_image(image_name)
            encoded_image = self._encode_image_to_base64(serve_path)
            worker_addr = self._get_worker_address()

            # Ensure the query contains an <image> token.
            if '<image>' not in text:
                text = text + '\n<image>'

            # ----- Build conversation state as done in Gradio -----
            # Start with the default conversation state.
            state = default_conversation.copy()
            # Append the user message.
            state.append_message(state.roles[0], text)
            # Append a placeholder for the AI response.
            state.append_message(state.roles[1], None)
            # If this is the first round, use the template.
            if len(state.messages) == state.offset + 2:
                template_name = "mistral_instruct"  # Gradio overwrites to use this template.
                new_state = conv_templates[template_name].copy()
                # Use the last human message from the default conversation.
                new_state.append_message(new_state.roles[0], state.messages[-2][1])
                new_state.append_message(new_state.roles[1], None)
                state = new_state
            # Build the final prompt.
            prompt = state.get_prompt()
            # --------------------------------------------------------

            print("\nFinal prompt (from conversation template):", prompt)
            print(f"Number of <image> tokens in prompt: {prompt.count('<image>')}")

            # Use a simple stop token to avoid tensor mismatches.
            stop_token = "\n"
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_tokens,
                "stop": stop_token,
                "images": [encoded_image],
                "text": prompt  # Extra key to help satisfy generate() requirements.
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
                        # Strip the prompt from the beginning of the response.
                        new_text = data["text"][len(prompt):].strip()
                        full_response = new_text
                    else:
                        raise RuntimeError(f"Generation error {data['error_code']}: {data['text']}")
            print("\nFinal response from model:", full_response)
            return full_response

        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise