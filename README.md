# Benchmarking Large Language Models toward reasoning fairness and unanticipated bias

This repository contains the artifacts used towards my Bachelor's Degree Thesis, presented in June 2025 here: -tbd-

## Main components

### Python Risk Identification Tool for generative AI (PyRIT)

  * **Docs** · https://azure.github.io/PyRIT 
  * **Repo** · https://github.com/Azure/PyRIT
  * **Paper** · https://arxiv.org/abs/2410.02828


### A hand-built Bias Benchmark for Question Answering (BBQ)
  * **Repo** · https://github.com/nyu-mll/BBQ
  * **Paper** · https://arxiv.org/abs/2110.08193


### Plain Python files

  * **[bbq_dataset.py](bbq_dataset.py)** · PyRIT valid dataset class representing BBQ data. This class imports the dataset from the [original](data/bbq) BBQ data files.

  * **[pyrit_tuning.py](pyrit_tuning.py)** · Inherited PyRIT class to modify how PyRIT `QuestionAnswerScorer` generates question prompts.


### Notebooks

  * **[run-aifoundry-bbqdataset-qascorer.ipynb](run-aifoundry-bbqdataset-qascorer.ipynb)** · Notebook for runing questions in the cloud against models in [Azure AI foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/what-is-azure-ai-foundry).
  
  * **[run-huggingface-bbqdataset-qascorer.ipynb](run-huggingface-bbqdataset-qascorer.ipynb)** · Notebook for runing questions agains local models through [Hugging Face Inference API](https://huggingface.co/docs/huggingface_hub/v0.13.2/en/guides/inference).
  
  * **[run-local-result-file-post-processing.ipynb](run-local-result-file-post-processing.ipynb)** · This notebook includes a few pieces of code to process the answers into the CSV files used for data analysis. This is an optional component and only includes plain Python code using Pandas.


## Data

  * **[data/bbq](data/bbq)** · A copy of the original BBQ files utilized for this work.

  * **[data/final](data/final)** · These are the files I produced during this work.


## Models

These are the models used for this experiment:

| Model |	Developer |	Size (params)	| Training data (tokens)	| Release date | License |
|-|-|-|-|-|-|
| [Phi 3 mini 4k instruct](https://ai.azure.com/explore/models/Phi-3-mini-4k-instruct/version/4/registry/azureml) | [Microsoft](https://www.microsoft.com/en-us/research/publication/phi-3-technical-report-a-highly-capable-language-model-locally-on-your-phone/) | 3.8B | 3.3T | Dec. 2024 | [MIT](https://github.com/marketplace/models/azureml/Phi-3-mini-4k-instruct?tab=license) |
| [GPT-4o mini](https://ai.azure.com/explore/models/o4-mini/version/2025-04-16/registry/azure-openai) | [Open AI](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) | ~8B | *N/A* | July 2024 | *N/A* |
| [Gemma 3 4b it](https://huggingface.co/google/gemma-3-4b-it) | [Google](https://ai.google.dev/gemma/docs/core) | 675M | 4T | Mar. 2025 | [Gemma](https://ai.google.dev/gemma/terms) |
| [SmolLM 360M instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct) | [Hugging Face](https://huggingface.co/blog/smollm) | 360M | 600B | July 2024 | [Apache](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

