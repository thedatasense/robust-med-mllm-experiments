#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U -q bitsandbytes')


# In[ ]:


get_ipython().system('pip install -q sqlalchemy cockroachdb pandas psycopg2-binary matplotlib')


# In[ ]:


# I am storing all my results in cockroach db
get_ipython().system("curl --create-dirs -o $HOME/.postgresql/root.crt 'https://cockroachlabs.cloud/clusters/5bbbe91d-b65e-410e-a783-597c93f501f6/cert'")


# In[1]:


import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import json
import re
import os, time
import yaml


# In[2]:


cnfig_file="/home/bsada1/config.yaml"
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


# In[3]:


DB_URL=get_from_cnfg("cd_url")
import pandas as pd
from sqlalchemy.engine import create_engine
engine = create_engine(DB_URL)


# In[4]:


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



# In[5]:


fetch_generation_data(engine)


# In[6]:


import gc
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
        batch_idx: Optional batch index for more detailed logging
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


# In[7]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=1')


# In[8]:


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # Use nested float 4 for better accuracy
        )


model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
   quantization_config=quantization_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)


# In[9]:


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))


# In[10]:


def clean_output(text):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
decoded_text = processor.decode(output[0])
clean_message = clean_output(decoded_text)
print(clean_message)


# In[11]:


def generate_llama(
        prompt,
        image_path
):

    image = Image.open(image_path)

    messages = [
        {
            "role": "system",
            "content": (
                    "You are an expert medical professional. "
                    "When responding, provide a concise explanation of the image findings. "
                    "For example, if asked about abnormalities, answer briefly with terms like 'atelectasis, lung opacity'."
                )
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=400)
    clear_gpu_memory()

    return (clean_output(processor.decode(output[0])))


# In[14]:


def check_duplicate(engine,uid,question_id,question, question_category, model_name,image_link):
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


# In[15]:


def insert_model_response(engine, uid,question_id,question, question_category, actual_answer, model_name, model_answer, image_link):
    from sqlalchemy import text
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



# In[16]:


source_folder='/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/files/'


# In[ ]:


from sqlalchemy import text
model_id = "Llama_3.2_11B"
import time
error_list = set()
import time
for index, row in fetch_generation_data(engine).iterrows():
    uid=row["id"]
    question_id=row["question_id"]
    question_category=row["question_type"]
    question=row["question"]
    actual_answer=row["ground_truth"]
    image_link = source_folder + row["image"] 
    if check_duplicate(engine,uid,str(question_id), question, question_category, model_id,image_link):
        print(f"Duplicate record found for question: {question}. Skipping generation.")
        continue
    try:    
        generated_answer = generate_llama( row["question"], image_link)

    except:
        error_list.add(str(uid)+ "-" + str(question_id) + "-"+str(question_category) )
        continue 
    
    time.sleep(5)
    print(f"{model_id} : {generated_answer}")
    print(f"GT: {actual_answer}")
    #insert_model_response(engine, uid,question_id,question, question_category, actual_answer, model_name, model_answer, image_link):
    insert_model_response(engine, uid,question_id,question, question_category, actual_answer,model_id , generated_answer,image_link)
    print('--------------------------------')


# In[ ]:




