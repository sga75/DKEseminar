German Company Website Classification

End-to-end NLP pipeline classifying 5,702 German company websites using SBERT embeddings and semantic similarity models, benchmarked against an SVM + TF-IDF baseline (74% accuracy).
Stack: Python, SBERT, scikit-learn
Files: preprocess.py (data cleaning), semantic_models.py (SBERT-based classification), baseline_models.py (SVM/TF-IDF baseline), train.py, main.py

Prerequisite: Data Path Configuration

Before running the project, please ensure the DATA_PATH variable in scripts/config.py is updated to the absolute path of your website-data directory. 

For example:

DATA_PATH = "C:\\path\\to\\your\\website-data"

Replace "C:\path\to\your\website-data" with the actual full path where your website-data folder is located on your system.
