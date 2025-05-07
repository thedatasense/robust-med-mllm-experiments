#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/thedatasense/llm-healthcare/blob/main/MIMIC_GPT_Evaluation_C.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().system('pip install -q  sqlalchemy  pandas psycopg2-binary matplotlib')


# In[15]:


import pandas as pd
import sys
from IPython.display import clear_output
from sqlalchemy.engine import create_engine
#from datasets import load_dataset
from openai import OpenAI
import io
import base64
import random
import requests
import torch
from PIL import Image
#from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
#from qwen_vl_utils import process_vision_info
import os
import pandas as pd
from sqlalchemy.engine import create_engine
#from transformers import AutoProcessor, BitsAndBytesConfig
import json
import yaml
import platform
from sqlalchemy import text
from IPython.display import display,clear_output
import time
import json
import os
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
import traceback
from sqlalchemy import create_engine, text
import yaml


# In[17]:


cnfig_file="/Users/bineshkumar/Documents/config.yaml"
def get_from_cnfg(key_path,file_path=cnfig_file):
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
os_name = platform.system()


# In[23]:


if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    from google.colab import userdata
    engine = create_engine(userdata.get('DB_URL'))
    gem_key=userdata.get('DB_URL')
    oai_key=userdata.get('DB_URL')
    source_folder='/content/drive/MyDrive/Health_Data/MIMIC_JPG_AVL/mimic-cxr-jpg/2.1.0/files/'
elif os_name == "Darwin":
    cnfig_file="/Users/bineshkumar/Documents/config.yaml"
    DB_URL = get_from_cnfg("gcp_db_url",cnfig_file)
    gem_key=get_from_cnfg("gem_token",cnfig_file)
    oai_key=get_from_cnfg("oai_token",cnfig_file)
    source_folder='/Users/bineshkumar/Documents/mimic-cxr-jpg/2.1.0/files/'
elif os_name == "Linux":
    DB_URL = get_from_cnfg("gcp_db_url",cnfig_file)
    gem_key=get_from_cnfg("gem_token",cnfig_file)
    oai_key=get_from_cnfg("oai_token",cnfig_file)
    source_folder=""
engine = create_engine(DB_URL)


# In[19]:


def check_duplicate(full_file_path):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM mimicxp.svu_all_qns WHERE full_file_path = :path"
            ), {"path": full_file_path})
            count = result.scalar()
            return count > 0
    except Exception as e:
        error_message = f"Error checking if image is processed: {str(e)}"
        print(error_message)
        error_logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": full_file_path,
            "error_type": "DB_QUERY_ERROR",
            "error_message": error_message,
            "traceback": traceback.format_exc()
        })
        save_error_logs_safe()
        return False


# In[20]:


def insert_record(record):
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(text("""
                    INSERT INTO mimicxp.svu_all_qns
                    (case_id, frame_file, full_file_path, ground_truth, gpt_response, original_response)
                    VALUES (:case_id, :frame_file, :full_file_path, :ground_truth, :gpt_response, :original_response)
                """), {
                    "case_id": record.get("case_id"),
                    "frame_file": record.get("frame_file"),
                    "full_file_path": record.get("full_file_path"),
                    "ground_truth": record.get("ground_truth"),
                    "gpt_response": record.get("gpt_response"),
                    "original_response": record.get("original_response")
                })
                trans.commit()
                print(f"Record inserted for image: {record.get('frame_file')}")
                return True
            except Exception as e:
                trans.rollback()
                raise e
    except Exception as e:
        error_message = f"Error inserting record: {str(e)}"
        print(error_message)
        error_logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": record.get("full_file_path"),
            "error_type": "DB_INSERT_ERROR",
            "error_message": error_message,
            "traceback": traceback.format_exc()
        })
        save_error_logs_safe()
        return False


# In[8]:


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


# In[21]:


def encode_image_stream(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    else:
        return None

def generate_gpt_response(prompt_text, image_link):
    base64_image = encode_image_stream(image_link)

    if base64_image is None:
        return None

    client = OpenAI(api_key=oai_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "When shown an image containing medical or surgical instruments, respond ONLY with a JSON object containing an array of instrument names you can identify in the image. Do not include any explanations, descriptions, or additional commentary."
                    "\n\nThe JSON response should follow this exact format:"
                    "\n{"
                    "\n  \"instruments\": ["
                    "\n    \"Instrument Name 1\","
                    "\n    \"Instrument Name 2\","
                    "\n    \"Instrument Name 3\""
                    "\n  ]"
                    "\n}"
                    "\n\nIf no instruments are visible or identifiable in the image, respond with:"
                    "\n{"
                    "\n  \"instruments\": []"
                    "\n}"
                    "\n\nDo not include any text before or after the JSON object. The response must be valid, parseable JSON containing only the instrument names."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content

def save_failed_images(failed_list, filename="failed_images_gpt4o.txt"):
    with open(filename, "w") as f:
        for img in failed_list:
            f.write(f"{img}\n")


# In[9]:


root_path='/Volumes/TVault2/datasets/sugvu24/extracted_frames'


# In[6]:


error_log_csv = "error_log.csv"
error_logs = []
import time
# Helper function to save error logs safely
def save_error_logs_safe():
    try:
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            error_df.to_csv(error_log_csv, index=False)
            print(f"Error logs saved to {error_log_csv}")
    except Exception as e:
        print(f"Error saving error logs: {e}")


# In[25]:


# Extract case_id from full_file_path
def extract_case_id(file_path):
    try:
        # Split the path by directory separator
        path_parts = file_path.split(os.sep)

        # Look for a part that starts with "case_"
        for part in path_parts:
            if part.startswith("case_"):
                # Extract the number part (remove "case_" prefix)
                case_num = part[5:]
                # Convert to integer if it's all digits
                if case_num.isdigit():
                    return int(case_num)
                else:
                    return case_num  # Return as string if not all digits

        # If no case part is found, return None
        return None
    except Exception as e:
        print(f"Error extracting case_id: {e}")
        return None


# In[ ]:


processed_count = 0
skipped_count = 0
error_count = 0

try:
    for root, dirs, files in os.walk(root_path):
        print(f"Currently in: {root}")
        for file in files:
            if file.endswith(".csv") and not file.startswith("."):
                file_path = os.path.join(root, file)
                print(f"  Found file: {file_path}")

                try:
                    df = pd.read_csv(os.path.join(root, file))
                    print(df.columns)
                except Exception as e:
                    error_message = f"Error reading CSV file {file_path}: {str(e)}"
                    print(error_message)
                    error_logs.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "file_path": file_path,
                        "error_type": "CSV_READ_ERROR",
                        "error_message": error_message,
                        "traceback": traceback.format_exc()
                    })
                    save_error_logs_safe()
                    error_count += 1
                    continue

                for index, row in df.iterrows():
                    try:
                        frame_file = row['frame_filename']
                        gt_tool_name = row['groundtruth_toolname']

                        # Get full image path
                        img_path = os.path.join(root, frame_file)
                        case_id = extract_case_id(img_path)

                        # Skip if we've already processed this image
                        if check_duplicate(img_path):
                            print(f"Skipping already processed image: {img_path}")
                            skipped_count += 1
                            continue

                        # Display the image
                        try:
                            img = mpimg.imread(img_path)
                            plt.figure(figsize=(10, 8))
                            plt.imshow(img)
                            plt.title(f"Case ID: {case_id}\nFrame: {frame_file}")
                            plt.axis('off')
                            plt.show()
                        except Exception as e:
                            error_message = f"Error displaying image {img_path}: {str(e)}"
                            print(error_message)
                            error_logs.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "file_path": img_path,
                                "error_type": "IMAGE_DISPLAY_ERROR",
                                "error_message": error_message,
                                "traceback": traceback.format_exc()
                            })
                            save_error_logs_safe()
                            error_count += 1

                        # Generate and display the GPT response
                        try:
                            gpt_response = generate_gpt_response("Identify the tool name", img_path)
                            print(f"Original response: {gpt_response}")

                            # Parse the JSON response
                            try:
                                # Strip any leading/trailing whitespace
                                cleaned_response = gpt_response.strip()

                                # Handle potential markdown code block formatting
                                if cleaned_response.startswith("```json"):
                                    cleaned_response = cleaned_response.replace("```json", "", 1)
                                    if cleaned_response.endswith("```"):
                                        cleaned_response = cleaned_response[:-3]
                                    cleaned_response = cleaned_response.strip()

                                gpt_tools_json = json.loads(cleaned_response)
                                gpt_tools_list = gpt_tools_json.get("instruments", [])
                                gpt_tools_str = ", ".join(gpt_tools_list)
                            except Exception as e:
                                gpt_tools_str = "Error parsing JSON response"
                                error_message = f"JSON parsing error for {img_path}: {str(e)}"
                                print(error_message)
                                error_logs.append({
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "file_path": img_path,
                                    "error_type": "JSON_PARSE_ERROR",
                                    "error_message": error_message,
                                    "original_response": gpt_response,
                                    "traceback": traceback.format_exc()
                                })
                                save_error_logs_safe()
                                error_count += 1

                            print(f"GPT Response: {gpt_tools_str}")
                            print(f"Ground Truth: {gt_tool_name}")

                            # Add record to database
                            record = {
                                "case_id": case_id,
                                "frame_file": frame_file,
                                "full_file_path": img_path,
                                "ground_truth": gt_tool_name,
                                "gpt_response": gpt_tools_str,
                                "original_response": gpt_response
                            }

                            if insert_record(record):
                                processed_count += 1

                        except Exception as e:
                            error_message = f"GPT processing error for {img_path}: {str(e)}"
                            print(error_message)
                            error_logs.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "file_path": img_path,
                                "error_type": "GPT_PROCESSING_ERROR",
                                "error_message": error_message,
                                "traceback": traceback.format_exc()
                            })
                            save_error_logs_safe()
                            error_count += 1

                            # Still add to database with error note
                            error_record = {
                                "case_id": case_id,
                                "frame_file": frame_file,
                                "full_file_path": img_path,
                                "ground_truth": gt_tool_name,
                                "gpt_response": "ERROR: GPT processing failed",
                                "original_response": str(e)
                            }
                            insert_record(error_record)

                        # Wait for 5 seconds
                        #time.sleep(5)

                        # Clear the output
                        clear_output(wait=True)

                    except Exception as e:
                        error_message = f"Error processing row {index} in file {file_path}: {str(e)}"
                        print(error_message)
                        error_logs.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "file_path": file_path,
                            "row_index": index,
                            "error_type": "ROW_PROCESSING_ERROR",
                            "error_message": error_message,
                            "traceback": traceback.format_exc()
                        })
                        save_error_logs_safe()
                        error_count += 1

                # We've finished processing this CSV file
                break
except Exception as e:
    error_message = f"Critical error in main processing loop: {str(e)}"
    print(error_message)
    error_logs.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_type": "CRITICAL_ERROR",
        "error_message": error_message,
        "traceback": traceback.format_exc()
    })
    error_count += 1
finally:
    # Save error logs
    save_error_logs_safe()

    # Print statistics
    print(f"Processing complete.")
    print(f"Total new images processed: {processed_count}")
    print(f"Total images skipped (already in database): {skipped_count}")
    print(f"Total errors encountered: {error_count}")
    if error_count > 0:
        print(f"See {error_log_csv} for error details.")


# In[ ]:




