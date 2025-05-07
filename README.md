# LLM for Healthcare (llm-healthcare)

This repository contains code and resources related to evaluating and using Large Language Models (LLMs) in the healthcare domain. It includes examples ranging from basic PyTorch implementations to advanced transformer models, robustness experiments with medical images, and guidelines for constructing your own LLMs.

## Repository Structure

The repository is organized as follows:

* **`/src/`**: Core source code and utilities
  * **`/src/runners/`**: Model inference runners for different models (LLama, LLaVA)
  * **`/src/data_processing/`**: Data preparation and modification utilities
  * **`/src/evaluation/`**: Evaluation metrics and tools
  * **`/src/utils/`**: Common utilities and helpers
  
* **`/experiments/`**: Experiment notebooks organized by task
  * **`/experiments/radiologist/`**: Chest X-ray interpretation experiments (includes robustness tests)
  * **`/experiments/surgical_tools/`**: Surgical tool identification experiments

* **`/tutorials/`**: Educational notebooks demonstrating LLM concepts
  * **`01-PyTorch-Basics.ipynb`**: Basic concepts using PyTorch
  * **`02-Transformer-Basics.ipynb`**: Introduction to transformer models
  * **`03-Building-LLM.ipynb`**: Guide to building your own LLM
  * **`04-Instruction-Tuning.ipynb`**: Instructions for fine-tuning models
  * **`05-Llama3-Pretrained.ipynb`**: Working with Llama 3 models
  * **`06-LLM-Robustness.ipynb`**: Testing LLM robustness
  * **`07_GRPO_Qwen_0_5_Instruct.ipynb`**: GRPO fine-tuning with Qwen
  * **`08_Tiny_VLM_Training.ipynb`**: Training tiny vision-language models

* **`/results/`**: Results and performance analysis
  * **`/results/analysis/`**: Model performance analysis
  * **`/results/monitoring/`**: Runtime monitoring statistics
  * **`/results/radiologist/`**: Radiologist task results
  * **`/results/surgical_tools/`**: Surgical tools detection results

* **`/data/`**: Datasets, image data, and evaluation metadata
* **`/docs/`**: Documentation and research notes
* **`/tests/`**: Test files and sample data
* **`/archived_files/`**: Legacy code and deprecated experiments

## Models Evaluated

This repository contains experiments with various large language and vision-language models:

1. **LLMs (Text-only models)**
   * Llama 3 (various sizes)
   * GPT models (via API)

2. **Vision-Language Models (VLMs)**
   * LLaVA-Med (Medical domain specialized)
   * Gemini (Google's multimodal model)
   * Gemma Vision (Google's open VLM)
   * CheXagent (Chest X-ray specialized model)

## Experiment Types

### Radiologist Task (X-ray Analysis)
The radiologist experiments evaluate how different models perform at interpreting chest X-rays:
* Base performance tests (standard images)
* Robustness tests using perturbed images (noise, artifacts)
* Adversarial sample testing

### Surgical Tools Detection
These experiments test model performance at identifying surgical instruments:
* Visual recognition of tools in surgical scenes
* Evaluation across different surgical procedures
* Performance comparison across model types

## Key Features

* **Multimodal Medical Evaluation**: Testing of vision-language models on medical images
* **Robustness Analysis**: Assessment of model performance under various perturbations
* **Performance Monitoring**: Tools to track and analyze model performance metrics
* **Educational Content**: Tutorials explaining LLM fundamentals and implementation

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
   * Start with the tutorials to understand the concepts
   * Review the experiment notebooks for practical evaluations
   * Use the monitoring tools to track performance metrics

## License

This project is licensed under the terms of the license included in the `LICENSE` file.

## Acknowledgments

* The MIMIC-CXR dataset (Johnson et al.)
* Harvard-FairVLMed benchmark
* Contributors to the open-source LLM ecosystem