# Augmented TPSA Prediction with RAG and MIPRO

This repository contains tools and code to replicate the research presented in the study **"Augmented and Programmatically Optimized LLM Prompts Reduce Chemical Hallucinations."** The project demonstrates the enhancement of molecular property prediction accuracy by combining Retrieval-Augmented Generation (RAG) and Multiprompt Instruction Proposal Optimizer (MIPRO) with large language models (LLMs). Specifically, this repository focuses on predicting **topological polar surface area (TPSA)** for molecules.

## Features
1. **Outlier Visualization**:  
   `plot_outliers.py` generates supporting information figures for analyzing outliers in TPSA prediction.
   
2. **Randomized Dataset Creation**:  
   `tpsa_random_pubchem.py` creates or overwrites `balanced_tpsa_data.csv` in the `tpsa_saved_data/` directory. This script pulls molecular data from PubChem and ensures a balanced dataset for TPSA predictions.

3. **Typed TPSA Prediction**:  
   `typed_tpsa_prediction.py`:
   - Builds datasets for training and validation.
   - Optimizes prompts for accurate TPSA prediction.
   - Plots results and saves models for reuse.
   - Requires an OpenAI API key specified in a `.env` file.

---

## Installation

### Requirements
- Python 3.9 or higher
- [RDKit](https://www.rdkit.org/) for molecular data handling.
- [DSPy](https://github.com/stanfordnlp/dspy) DSPy 2.5.18 # higher versions lack typed predictors
- OpenAI API access for LLM predictions.
- deepchem 2.8 # PyTorch not required
- additional requirements in requirements.txt

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/augmented-tpsa-prediction.git
   cd augmented-tpsa-prediction
   ```

2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a .env file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

### Re-creating manuscript data and figures
To re-create SI figure S3, run comparison\
To re-create all other SI figures, run plot_outliers\
To re-create manuscript figures, run typed_tpsa_prediction\
To re-train models, delete all .json files from tpsa_saved_data and run typed_tpsa_prediction, then recreate no_demos and no_sig json files manually\
To re-run analyis, delete all csv files from tpsa_saved_data and run typed_tpsa_prediction\
To create a new molecular dataset run tpsa_random_pubchem\
To recreate from scratch with new data, delete all csv and json files, run tpsa_random_pubchem, typed_tpsa_prediction, plot_outliers, and comparison
