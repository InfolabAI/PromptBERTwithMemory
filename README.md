# PromptBERT-based Algoritm Using Memory Bank for Detecting AI Generated Text
## About This Code
- This is my implementation of PromptBERT using memory bank for detection AI generated text
## Getting Started
### Download the datasets
- Training dataset - DAIGT V2 Train Dataset [link](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
- Test dataset - DAIGT: Extended data [link](https://www.kaggle.com/datasets/batprem/daigt-extended-data)
### Run the Code
- PromptBERT-based method using memory bank
	- Training and test - `run_train_bert.py <TRAINING_DATASET_PATH> <TEST_DATASET_PATH>`
	- Test with saved model and memory bank - `test_trained_bert_anomaly.py <SAVED_MODEL_PATH> <SAVED_FAISS_PATH> <TRAINING_DATASET_PATH> <TEST_DATASET_PATH>`
		- This code uses voting classifier learning TFIDF features
- TFIDF-based method
	- `run_idftoken_catboost.py <TRAINING_DATASET_PATH> <TEST_DATASET_PATH>`
### Code for Reference
- HPO with WanDB
	- Training - `detection-with-tfidf.py`
	- Test - `detection_with_tfidf_with_bestconfig.py`
	- Config - `config.py`  and `config.yaml`
- Only memory bank based method
	- `run_anomaly.py`
	- `test_anomaly.py`
- Text generation
	- `text_generation.py`
## The Results of Evaluation
### PromptBERT-based method using memory bank (at epoch 14)
- This result shows that PromptBERT performs better on normal test samples than anomaly test samples
	- PromptBERT is a binary classifier
- Anomaly scores of test samples are sorted first, then they are used to split test samples into four groups according to the ratio (0.01, 0.05,  0.1, 0.5) from top (anomaly samples) and bottom (normal samples)
- AUROC for anomaly test samples

| Metric      | 0.01  | 0.05  |  0.1  |  0.5  |
| ----------- |:-----:|:-----:|:-----:|:-----:|
| auroc       | 0.866 | 0.903 | 0.918 | 0.952 |
| label ratio | 0.749 | 0.713 | 0.698 | 0.726 |
| num         |  175  |  878  | 1757  | 8789  |
| score min   | 4.039 | 2.809 | 2.503 | 1.907 |

- AUROC for normal test samples

| Metric      | 0.01  | 0.05  |  0.1  |  0.5  |
| ----------- |:-----:|:-----:|:-----:|:-----:|
| auroc       | 0.999 | 0.995 | 0.990 | 0.977 |
| label ratio | 0.909 | 0.905 | 0.892 | 0.840 |
| num         |  175  |  878  | 1757  | 8789  |
| score max   | 0.000 | 1.255 | 1.464 | 1.907 |
