#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
# from IPython.display import clear_output # Removed - IPython specific
from sqlalchemy.engine import create_engine
from openai import OpenAI
import io
import base64
import random
import requests
import torch # Note: torch is imported but not used in the provided script snippet.
from PIL import Image
import os
import json
import yaml
import platform
from sqlalchemy import text
# from IPython.display import display # Removed - IPython specific
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import traceback
# from sqlalchemy import create_engine, text # Already imported via sqlalchemy

# --- Configuration ---
cnfig_file="/Users/bineshkumar/Documents/config.yaml" # Make sure this path is correct or adjust as needed
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
        print(f"Error: Configuration file {file_path} not found")
    except yaml.YAMLError as e:
        print(f"Error: YAML parsing error: {e}")
    except KeyError:
        print(f"Error: Key path {key_path} not found in configuration")
    except Exception as e:
        print(f"Error reading configuration: {e}")
    # Exit if configuration fails, as API keys/DB URL are essential
    sys.exit(f"Exiting due to configuration error.")

os_name = platform.system()

# Initialize configuration variables
DB_URL = None
gem_key = None # Note: gem_key is loaded but not used in the script snippet
oai_key = None
source_folder = None

if 'google.colab' in sys.modules:
    print("Warning: Running outside Colab. Ensure config.yaml is set up correctly.")
    # Attempt to load from config file as fallback for non-Colab environments
    # (Original Colab logic using userdata is removed)
    DB_URL = get_from_cnfg("gcp_db_url",cnfig_file)
    gem_key=get_from_cnfg("gem_token",cnfig_file)
    oai_key=get_from_cnfg("oai_token",cnfig_file)
    source_folder='' # Adjust if needed outside Colab
elif os_name == "Darwin":
    # cnfig_file="/Users/bineshkumar/Documents/config.yaml" # Already defined above
    DB_URL = get_from_cnfg("gcp_db_url",cnfig_file)
    gem_key=get_from_cnfg("gem_token",cnfig_file)
    oai_key=get_from_cnfg("oai_token",cnfig_file)
    source_folder='/Users/bineshkumar/Documents/datasets/surgical-vu/extracted_frames'
elif os_name == "Linux":
    # Assuming config file path is appropriate for Linux if not default
    # cnfig_file="/path/to/your/config.yaml" # Adjust if necessary
    DB_URL = get_from_cnfg("gcp_db_url",cnfig_file)
    gem_key=get_from_cnfg("gem_token",cnfig_file)
    oai_key=get_from_cnfg("oai_token",cnfig_file)
    source_folder="" # Set appropriate default source folder for Linux if needed
else:
    sys.exit(f"Unsupported OS: {os_name}")

# Validate essential configuration
if not all([DB_URL, oai_key, source_folder is not None]): # gem_key not checked as it's unused
    sys.exit("Essential configuration (DB_URL, oai_token, source_folder) missing. Check config.yaml.")

try:
    engine = create_engine(DB_URL)
    # Test connection
    with engine.connect() as connection:
        print("Database connection successful.")
except Exception as e:
    sys.exit(f"Failed to create database engine or connect: {e}")

# --- Global Variables and Setup ---
error_log_csv = "error_log.csv"
error_logs = []

# --- Helper Functions ---
def save_error_logs_safe():
    """Saves the current error logs to a CSV file."""
    try:
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            # Use mode='a' and header=not os.path.exists(error_log_csv) to append
            file_exists = os.path.exists(error_log_csv)
            error_df.to_csv(error_log_csv, mode='a', header=not file_exists, index=False)
            print(f"Error logs saved/appended to {error_log_csv}")
            error_logs.clear() # Clear logs after saving
    except Exception as e:
        print(f"CRITICAL ERROR saving error logs: {e}")

def check_duplicate(full_file_path):
    """Checks if a record with the given file path already exists in the database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM mimicxp.svu_all_qns WHERE full_file_path = :path"
            ), {"path": full_file_path})
            count = result.scalar()
            return count > 0
    except Exception as e:
        error_message = f"Error checking duplicate for {full_file_path}: {str(e)}"
        print(error_message)
        error_logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": full_file_path,
            "error_type": "DB_QUERY_ERROR",
            "error_message": error_message,
            "original_response": "",
            "traceback": traceback.format_exc(),
            "row_index": None
        })
        # Don't save logs here, let the main loop handle it or the final block
        return False # Treat DB error as 'not duplicate found' to allow processing attempt

def insert_record(record):
    """Inserts a record into the database."""
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
                raise e # Re-raise to be caught by the outer try/except
    except Exception as e:
        error_message = f"Error inserting record for {record.get('full_file_path')}: {str(e)}"
        print(error_message)
        error_logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": record.get("full_file_path"),
            "error_type": "DB_INSERT_ERROR",
            "error_message": error_message,
            "original_response": record.get("original_response",""),
            "traceback": traceback.format_exc(),
            "row_index": None # Row index might not be available here
        })
        # Don't save logs here
        return False

def encode_image_stream(image_path):
    """Encodes an image file into a base64 string."""
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading or encoding image {image_path}: {e}")
            return None
    else:
        print(f"Image file not found: {image_path}")
        return None

def generate_gpt_response(prompt_text, image_link):
    """Generates a response from GPT-4o based on a prompt and an image path."""
    base64_image = encode_image_stream(image_link)

    if base64_image is None:
        print(f"Skipping GPT request due to image encoding error for: {image_link}")
        return None # Return None if image encoding failed

    try:
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
            max_tokens=150 # Added max_tokens for safety
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API for {image_link}: {e}")
        # Log this specific error type
        error_logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": image_link,
            "error_type": "API_CALL_ERROR",
            "error_message": str(e),
            "original_response": "",
            "traceback": traceback.format_exc(),
            "row_index": None # Row index might not be available here
        })
        return None # Indicate failure

# Function to save failed images (if needed, original function kept)
def save_failed_images(failed_list, filename="failed_images_gpt4o.txt"):
    try:
        with open(filename, "w") as f:
            for img in failed_list:
                f.write(f"{img}\n")
        print(f"List of failed images saved to {filename}")
    except Exception as e:
        print(f"Error saving failed images list: {e}")


def extract_case_id(file_path):
    """Extracts the case ID (e.g., 'case_001') from a file path."""
    try:
        # Split the path by directory separator
        path_parts = file_path.split(os.sep)

        # Look for a part that starts with "case_"
        for part in path_parts:
            if part.startswith("case_"):
                # Extract the number part (remove "case_" prefix)
                case_num_str = part[5:]
                # Try converting to integer if it's all digits, otherwise return as string
                try:
                    return int(case_num_str)
                except ValueError:
                    return case_num_str # Return as string if not purely numeric

        # If no case part is found, return None
        print(f"Warning: Could not extract case_id from path: {file_path}")
        return None
    except Exception as e:
        print(f"Error extracting case_id from {file_path}: {e}")
        return None

# --- Main Processing Logic ---
def main():
    root_path = source_folder # Use configured source folder
    if not os.path.isdir(root_path):
        sys.exit(f"Error: Source folder not found or is not a directory: {root_path}")

    processed_count = 0
    skipped_count = 0
    error_count = 0
    csv_files_processed = 0

    print(f"Starting processing in root directory: {root_path}")

    try:
        for root, dirs, files in os.walk(root_path):
            # Filter out hidden directories (optional, good practice)
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            csv_found_in_dir = False
            for file in files:
                if file.lower().endswith(".csv") and not file.startswith("."):
                    csv_files_processed += 1
                    csv_found_in_dir = True
                    file_path = os.path.join(root, file)
                    print(f"\n--- Processing CSV file: {file_path} ---")

                    try:
                        # Specify dtype={'frame_filename': str} to prevent issues if it looks numeric
                        df = pd.read_csv(file_path, dtype={'frame_filename': str})
                        print(f"  Columns found: {list(df.columns)}")
                        if 'frame_filename' not in df.columns or 'groundtruth_toolname' not in df.columns:
                            print(f"  Error: CSV missing required columns ('frame_filename', 'groundtruth_toolname'). Skipping file.")
                            error_logs.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "file_path": file_path,
                                "error_type": "CSV_MISSING_COLUMNS",
                                "error_message": "CSV missing required columns ('frame_filename', 'groundtruth_toolname')",
                                "original_response": "",
                                "traceback": "",
                                "row_index": None
                            })
                            error_count += 1
                            continue # Skip this CSV file
                    except Exception as e:
                        error_message = f"Error reading CSV file {file_path}: {str(e)}"
                        print(f"  {error_message}")
                        error_logs.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "file_path": file_path,
                            "error_type": "CSV_READ_ERROR",
                            "error_message": error_message,
                            "original_response": "",
                            "traceback": traceback.format_exc(),
                            "row_index": None
                        })
                        save_error_logs_safe() # Save immediately on critical CSV read error
                        error_count += 1
                        continue # Skip this file

                    # Process rows within the current CSV
                    for index, row in df.iterrows():
                        frame_file = None # Initialize in case of errors reading row
                        img_path = None
                        case_id = None
                        gt_tool_name = None

                        try:
                            frame_file = row['frame_filename']
                            # Handle potential NaN or missing frame filenames explicitly
                            if pd.isna(frame_file) or not isinstance(frame_file, str) or not frame_file.strip():
                                print(f"  Skipping row {index} due to invalid/missing frame_filename.")
                                error_logs.append({
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "file_path": file_path, # CSV file path
                                    "row_index": index,
                                    "error_type": "INVALID_ROW_DATA",
                                    "error_message": f"Invalid or missing frame_filename: {frame_file}",
                                    "original_response": "",
                                    "traceback": "",
                                })
                                error_count += 1
                                continue # Skip this row

                            frame_file = frame_file.strip() # Remove leading/trailing whitespace
                            gt_tool_name = row['groundtruth_toolname']

                            # Construct full image path - use root from os.walk
                            img_path = os.path.join(root, frame_file)
                            case_id = extract_case_id(img_path)

                            print(f"\n  Processing row {index}: Image: {frame_file} (Case: {case_id})")

                            # Check if already processed
                            if check_duplicate(img_path):
                                print(f"  Skipping already processed image: {img_path}")
                                skipped_count += 1
                                continue

                            # --- Optional Image Display ---
                            # Uncomment the block below if you want to see each image.
                            # Note: This will pause the script for each image.
                            # try:
                            #     if os.path.exists(img_path):
                            #         img = mpimg.imread(img_path)
                            #         plt.figure(figsize=(10, 8))
                            #         plt.imshow(img)
                            #         plt.title(f"Case ID: {case_id}\nFrame: {frame_file}\nGround Truth: {gt_tool_name}")
                            #         plt.axis('off')
                            #         plt.show() # This pauses execution
                            #     else:
                            #          print(f"  Warning: Image file not found for display: {img_path}")
                            # except Exception as e:
                            #     error_message = f"Error displaying image {img_path}: {str(e)}"
                            #     print(f"  {error_message}")
                            #     # Log display error but continue processing
                            #     error_logs.append({
                            #         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            #         "file_path": img_path,
                            #         "error_type": "IMAGE_DISPLAY_ERROR",
                            #         "error_message": error_message,
                            #         "original_response": "",
                            #         "traceback": traceback.format_exc(),
                            #         "row_index": index
                            #     })
                            #     error_count += 1 # Count display errors if desired
                            # -----------------------------

                            # Check if image file exists before calling API
                            if not os.path.exists(img_path):
                                print(f"  Error: Image file not found: {img_path}. Skipping GPT processing and DB insert.")
                                error_logs.append({
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "file_path": img_path,
                                    "error_type": "IMAGE_NOT_FOUND",
                                    "error_message": "Image file specified in CSV not found at path.",
                                    "original_response": "",
                                    "traceback": "",
                                    "row_index": index
                                })
                                error_count += 1
                                continue # Skip to the next row


                            # Generate GPT response
                            original_response = generate_gpt_response("Identify the tool name", img_path)
                            gpt_tools_str = "ERROR: No response from API" # Default if response is None

                            if original_response is not None:
                                print(f"  Original response received: {original_response[:100]}...") # Print snippet

                                # Parse the JSON response
                                try:
                                    # Strip any leading/trailing whitespace or markdown code blocks
                                    cleaned_response = original_response.strip()
                                    if cleaned_response.startswith("```json"):
                                        cleaned_response = cleaned_response.replace("```json", "", 1).strip()
                                    if cleaned_response.endswith("```"):
                                        cleaned_response = cleaned_response[:-3].strip()

                                    # Attempt to load JSON
                                    gpt_tools_json = json.loads(cleaned_response)
                                    # Check if it's a dictionary with the expected key
                                    if isinstance(gpt_tools_json, dict) and "instruments" in gpt_tools_json:
                                        gpt_tools_list = gpt_tools_json.get("instruments", [])
                                        # Ensure list items are strings
                                        gpt_tools_list = [str(item) for item in gpt_tools_list if isinstance(item, (str, int, float))]
                                        gpt_tools_str = ", ".join(gpt_tools_list) if gpt_tools_list else "[]" # Represent empty list clearly
                                    else:
                                        # Handle cases where the JSON is valid but not the expected format
                                        gpt_tools_str = "ERROR: Unexpected JSON format"
                                        print(f"  Warning: Unexpected JSON format received: {cleaned_response}")
                                        error_logs.append({
                                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "file_path": img_path,
                                            "error_type": "JSON_FORMAT_ERROR",
                                            "error_message": "Received valid JSON but not the expected {'instruments': [...]} format.",
                                            "original_response": original_response,
                                            "traceback": "",
                                            "row_index": index
                                        })
                                        error_count += 1

                                except json.JSONDecodeError as e:
                                    gpt_tools_str = "ERROR: Invalid JSON response"
                                    error_message = f"JSON parsing error for {img_path}: {str(e)}"
                                    print(f"  {error_message}")
                                    error_logs.append({
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "file_path": img_path,
                                        "error_type": "JSON_PARSE_ERROR",
                                        "error_message": error_message,
                                        "original_response": original_response,
                                        "traceback": traceback.format_exc(),
                                        "row_index": index
                                    })
                                    error_count += 1
                                except Exception as e: # Catch other potential errors during parsing/joining
                                    gpt_tools_str = "ERROR: Processing response failed"
                                    error_message = f"Unexpected error processing response for {img_path}: {str(e)}"
                                    print(f"  {error_message}")
                                    error_logs.append({
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "file_path": img_path,
                                        "error_type": "RESPONSE_PROCESSING_ERROR",
                                        "error_message": error_message,
                                        "original_response": original_response,
                                        "traceback": traceback.format_exc(),
                                        "row_index": index
                                    })
                                    error_count += 1
                            else:
                                # API call itself failed (error already logged in generate_gpt_response)
                                gpt_tools_str = "ERROR: API call failed"
                                error_count += 1 # Increment error count as processing failed


                            print(f"  Parsed GPT Response: {gpt_tools_str}")
                            print(f"  Ground Truth: {gt_tool_name}")

                            # Prepare and insert record
                            record = {
                                "case_id": case_id,
                                "frame_file": frame_file,
                                "full_file_path": img_path,
                                "ground_truth": gt_tool_name,
                                "gpt_response": gpt_tools_str, # Store parsed or error string
                                "original_response": original_response if original_response is not None else "N/A (API Error)"
                            }

                            if insert_record(record):
                                processed_count += 1
                            else:
                                # Insertion failed, error already logged by insert_record
                                error_count += 1 # Ensure error count is incremented


                        except KeyError as e:
                            error_message = f"Missing expected column '{str(e)}' in row {index} of file {file_path}"
                            print(f"  {error_message}")
                            error_logs.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "file_path": file_path,
                                "row_index": index,
                                "error_type": "MISSING_COLUMN_IN_ROW",
                                "error_message": error_message,
                                "original_response": "",
                                "traceback": traceback.format_exc()
                            })
                            error_count += 1
                        except Exception as e:
                            # Catch-all for unexpected errors processing a single row
                            error_message = f"Unexpected error processing row {index} (Image: {frame_file}) in file {file_path}: {str(e)}"
                            print(f"  {error_message}")
                            error_logs.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "file_path": img_path if img_path else file_path, # Log image path if available
                                "row_index": index,
                                "error_type": "ROW_PROCESSING_ERROR",
                                "error_message": error_message,
                                "original_response": "",
                                "traceback": traceback.format_exc()
                            })
                            error_count += 1

                        # Optional: add a small delay between API calls to avoid rate limits
                        # time.sleep(1)

                        # Save errors periodically (e.g., every 50 rows) to avoid memory issues
                        if len(error_logs) >= 50:
                            save_error_logs_safe()

                    # Finished processing rows in this CSV
                    print(f"--- Finished processing CSV: {file_path} ---")
                    if not csv_found_in_dir:
                        print(f"No CSV files found in directory: {root}")


    except Exception as e:
        # Catch critical errors in the os.walk loop itself
        error_message = f"CRITICAL error during directory traversal or processing: {str(e)}"
        print(error_message)
        error_logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": None, # Error might occur before a file path is known
            "row_index": None,
            "error_type": "CRITICAL_PROCESSING_ERROR",
            "error_message": error_message,
            "original_response": "",
            "traceback": traceback.format_exc()
        })
        error_count += 1
    finally:
        # Final save of any remaining error logs
        save_error_logs_safe()

        # Print final statistics
        print("\n========================================")
        print(f"Processing complete.")
        print(f"Total CSV files found and attempted: {csv_files_processed}")
        print(f"Total new image records processed and inserted: {processed_count}")
        print(f"Total images skipped (already in database): {skipped_count}")
        print(f"Total errors encountered: {error_count}")
        if error_count > 0 or os.path.exists(error_log_csv):
            print(f"See {error_log_csv} for error details.")
        print("========================================")

# --- Script Execution ---
if __name__ == "__main__":
    main()