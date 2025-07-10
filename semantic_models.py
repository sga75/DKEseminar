from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import config
import torch

def get_sbert_embeddings(texts, model_name=config.SBERT_MODEL_NAME):

    print(f"Loading SBERT model: {model_name}...")
    device = "cuda" if config.USE_GPU_FOR_SBERT and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU for SBERT embeddings: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for SBERT embeddings.")
    
    model = SentenceTransformer(model_name, device=device)
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True) 
    return embeddings

def train_svm_sbert(X_train, y_train):

    print("\nTraining SVM (SBERT) model with hyperparameter tuning...")
    svm = SVC(random_state=42)

    if config.USE_RANDOMIZED_SEARCH:
        param_searcher = RandomizedSearchCV(
            svm,
            config.SVM_PARAM_DISTRIBUTIONS_RANDOMIZEDSEARCH,
            n_iter=config.N_ITER_RANDOMIZED_SEARCH,
            cv=3, 
            scoring='f1_weighted',
            n_jobs=config.N_JOBS,
            verbose=1,
            random_state=42
        )
    else:
        param_searcher = GridSearchCV(
            svm,
            config.SVM_PARAM_GRID_GRIDSEARCH,
            cv=3, 
            scoring='f1_weighted',
            n_jobs=config.N_JOBS,
            verbose=1
        )
    
    param_searcher.fit(X_train, y_train)

    print(f"Best parameters for SVM (SBERT): {param_searcher.best_params_}")
    print(f"Best cross-validation F1-score for SVM (SBERT): {param_searcher.best_score_:.4f}")
    return param_searcher.best_estimator_

def train_mlp_classifier(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=10)
    print("\nTraining MLPClassifier model...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)

    print(f"\n--- {model_name} Classification Report ---")
    
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    return accuracy, precision, recall, f1, y_pred 

