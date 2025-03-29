# LLM for Healthcare (llm-healthcare)

This repository contains code and resources related to evaluating and using Large Language Models (LLMs) in the healthcare domain. It includes examples ranging from basic PyTorch implementations to advanced transformer models, robustness experiments with medical images, and guidelines for constructing your own LLMs.

## Repository Structure

The repository is organized as follows:

* **`/src/`**: Core source code including model runners and processors
* **`/data/`**: Datasets, image data, and evaluation metadata
* **`/models/`**: Model implementations organized by model type:
  * **`/models/llama/`**: Llama 3 model implementations
  * **`/models/gpt/`**: GPT model implementations
  * **`/models/gemini/`**: Google Gemini model implementations
  * **`/models/gemma/`**: Google Gemma model implementations
  * **`/models/CheXagent/`**: CheXagent model implementations
* **`/notebooks/`**: Educational notebooks demonstrating LLM concepts:
  * **`01-PyTorch-Basics.ipynb`**: Basic concepts using PyTorch
  * **`02-Transformer-Basics.ipynb`**: Introduction to transformer models
  * **`03-Building-LLM.ipynb`**: Guide to building your own LLM
  * **`04-Instruction-Tuning.ipynb`**: Instructions for fine-tuning models
  * **`05-Llama3-Pretrained.ipynb`**: Working with Llama 3 models
  * **`06-LLM-Robustness.ipynb`**: Testing LLM robustness
* **`/utils/`**: Shared utilities and helper functions
* **`/tests/`**: Test files and sample data
* **`/results/`**: Evaluation results and monitoring statistics
* **`/docs/`**: Documentation and research notes

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
   * Start with the notebooks to understand the concepts
   * Review the model experiments for practical evaluations
   * Use the monitoring tools to track performance metrics

## License

This project is licensed under the terms of the license included in the `LICENSE` file.

## Acknowledgments

* The MIMIC-CXR dataset (Johnson et al.)
* Harvard-FairVLMed benchmark
* Contributors to the open-source LLM ecosystem