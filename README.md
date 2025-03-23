# LLM for Healthcare (llm-healthcare)

This repository contains code and resources related to evaluating and using Large Language Models (LLMs) in the healthcare domain. It includes examples ranging from basic PyTorch implementations to advanced transformer models, robustness experiments with medical images, and guidelines for constructing your own LLMs.

## Repository Structure

The repository is organized as follows:

* **`/appendix-notebooks`**: Contains educational notebooks that demonstrate LLM concepts:
  * `App-A-Pytorch-Basic.ipynb`: Basic concepts of LLMs using PyTorch
  * `App-B-Transformer-Basic.ipynb`: Introduction to transformer models
  * `App-C - Building LLM.ipynb`: Guide to building your own LLM
  * `App-C.1-Instruct.ipynb`: Instructions for fine-tuning models
  * `App-D - Llama3 Pretrained.ipynb`: Working with Llama 3 models
  * `App-E - LLM Robustness NL_Augmenter.ipynb`: Testing LLM robustness
  
* **`/datasets`**: Sample text datasets for training and evaluation
* **`/notes`**: Documentation and experimental notes
* **`/robustness_experments`**: Main experimental notebooks evaluating:
  * Medical vision-language models (Llama 3, Gemini, GPT-4V) on:
    * MIMIC-CXR (chest X-rays)
    * Fundus images (eye examination)
  * Robustness experiments with various perturbations

* **`/monitor_runs_stats`**: Statistics and logs from model evaluation runs
* **`monitor_gradio.py`**: Gradio interface for monitoring model performance

## Key Features

* **Multimodal Medical Evaluation**: Testing of vision-language models on medical images
* **Robustness Analysis**: Assessment of model performance under various perturbations
* **Performance Monitoring**: Tools to track and analyze model performance metrics
* **Educational Content**: Notebooks explaining LLM fundamentals and implementation

## Getting Started

1. **Prerequisites:**
   * Python 3.x
   * Jupyter Notebook or Jupyter Lab
   * PyTorch, Transformers, and other libraries (see notebook imports)

2. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/llm-healthcare.git
   cd llm-healthcare
   ```

3. **Explore the content:**
   * Start with the appendix notebooks to understand the concepts
   * Review the robustness experiments for practical evaluations
   * Use the monitoring tools to track performance metrics

## License

This project is licensed under the terms of the license included in the `LICENSE` file.

## Acknowledgments

* The MIMIC-CXR dataset (Johnson et al.)
* Harvard-FairVLMed benchmark
* Contributors to the open-source LLM ecosystem
