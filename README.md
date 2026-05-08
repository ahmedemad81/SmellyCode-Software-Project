# SmellyCode-Software-Project

## Code Smell Detection (MLCQ + Local Models)

This project focuses on **multi-class code smell classification** using the **MLCQ dataset** and **locally deployable language models**.

The system is designed to support **multiple models** (e.g., Qwen and others) for comparison and experimentation.

## 📌 Task

Classify Java code into one of the following code smells:

* `data_class`
* `feature_envy`
* `god_class`
* `long_method`

## 🧠 Approach

* **Models:** Multiple local LLMs (e.g., Qwen, future additions)
* **Fine-tuning:** LoRA (PEFT)
* **Task type:** Sequence Classification
* **Input:** Structured prompt with definitions

## ⚙️ Setup

pip install -r requirements.txt

## 📂 Dataset

### Download MLCQ Dataset

[https://zenodo.org/records/3666840](https://zenodo.org/records/3666840)

## 🛠 Preprocessing

python -m scripts.preprocess_mlcq

* Cleans dataset
* Normalizes labels
* Extracts code snippets
* Splits into train / validation / test

## 🚀 Training

python -m scripts.train

* Uses structured input (task + definitions)
* LoRA fine-tuning
* Early stopping enabled
* Best model selected based on macro F1

## 📈 Evaluation

python -m scripts.evaluate

### Outputs

* Accuracy
* Precision / Recall / F1 (macro & weighted)
* Confusion matrix
* Classification report

Results saved to: results/mlcq_multiclass/

## 🔍 Prediction

python -m scripts.predict

Predicts code smell for a given Java snippet.

## 📁 Outputs

### Models

outputs/

### Evaluation results

results/mlcq_multiclass/

## 📊 Results

| Model                         | F1 (Macro) | Precision (Macro) | Recall (Macro) | Accuracy |
| :---------------------------- | :--------- | :---------------- | :------------- | :------- |
| Qwen3-0.6B (Long Context)     | 0.451      | 0.460             | 0.461          | 0.460    |
| Qwen3-0.6B (Short Context)    | 0.364      | 0.444             | 0.491          | 0.492    |
| Qwen3-0.6B (No Context given) | 0.427      | 0.505             | 0.498          | 0.499    |
| CodeBERTa-small-v1 (long context) | 0.409      | 0.444             | 0.4669         | 0.4661   |
| codegemma-2b (long context)   | 0.140      | 0.138             | 0.223          | 0.219    |
| DeepSeek-R1-Distill-Qwen-1.5B (Long Context) | 0.465 | 0.4718  | 0.474          | 0.474    |
| Smollm2 (v1) | 0.1406 | 0.10845 |  0.2478 |  0.2434 |
| Smollm2 (v2 -after edits) | 0.4666 | 0.47705 |  0.4804 |  0.48106 |
| llama3.2:1b (v1) | 0.3908 | 0.4096 | 0.4846 | 0.48335 | 
| llama3.2:1b (v2 -after edits) | 0.3671 |  0.3611 |  0.45773 | 0.45695 |




## 📊 Notes

* Strong performance on structural smells (`long_method`, `data_class`)
* Harder classes (`feature_envy`, `god_class`) require deeper semantic understanding
* Main confusion occurs between `feature_envy` and `long_method`
* Performance is sensitive to learning rate, class weighting, and input representation
* Structured prompts improve stability and overall performance
