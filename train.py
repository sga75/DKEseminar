import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import numpy as np
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns 


import config


from data_loader import load_company_data
from preprocess import preprocess_text
from baseline_models import get_tfidf, train_multinomial_nb, train_svm_tfidf, evaluate_model as evaluate_baseline_model
from semantic_models import get_sbert_embeddings, train_svm_sbert, train_mlp_classifier, evaluate_model as evaluate_semantic_model

def _get_processed_texts_cached(raw_texts):
    """
    Loads processed texts from cache if available, otherwise preprocesses and caches.
    """
    if os.path.exists(config.PROCESSED_TEXT_CACHE):
        print(f"Loading processed texts from cache: {config.PROCESSED_TEXT_CACHE}")
        return joblib.load(config.PROCESSED_TEXT_CACHE)
    else:
        print("Preprocessing texts (this may take a moment due to spaCy lemmatization)...")
        processed_texts = [preprocess_text(text) for text in raw_texts]
        print("Text preprocessing complete.")
        print(f"Caching processed texts to: {config.PROCESSED_TEXT_CACHE}")
        joblib.dump(processed_texts, config.PROCESSED_TEXT_CACHE)
        return processed_texts

def _get_sbert_embeddings_cached(texts):
    """
    Loads SBERT embeddings from cache if available, otherwise generates and caches.
    """
    if os.path.exists(config.SBERT_EMBEDDINGS_CACHE):
        print(f"Loading SBERT embeddings from cache: {config.SBERT_EMBEDDINGS_CACHE}")
        return joblib.load(config.SBERT_EMBEDDINGS_CACHE)
    else:
        embeddings = get_sbert_embeddings(texts)
        print(f"Caching SBERT embeddings to: {config.SBERT_EMBEDDINGS_CACHE}")
        joblib.dump(embeddings, config.SBERT_EMBEDDINGS_CACHE)
        return embeddings

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 

def training_pipeline():
    """
    Orchestrates the entire training and evaluation pipeline using actual data
    and k-fold cross-validation, with caching for performance and confusion matrix plotting.
    """
    print("Starting company classification project...")

    data_dir = config.DATA_PATH
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: The data directory '{data_dir}' is empty or does not exist.")
        print("Please ensure your company data is placed in this directory as specified.")
        print("Expected structure: {data_dir}/company_id/company_data.json, code_desc.json, page1.html/txt")
        return

    
    print("Loading company data from provided directory...")
    companies_data, unique_categories = load_company_data(data_path=data_dir)
    if not companies_data:
        print("No data loaded. Please check your data directory and file structure. Exiting.")
        return

    print(f"Loaded {len(companies_data)} companies across {len(unique_categories)} categories.")

    raw_texts = [company['raw_text'] for company in companies_data]
    labels_original = [company['category'] for company in companies_data] 

    
    processed_texts = _get_processed_texts_cached(raw_texts)

    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_original) 

    
    class_names = label_encoder.classes_

    X = np.array(processed_texts, dtype=object)
    y = np.array(encoded_labels)

    n_splits = config.N_SPLITS_CROSS_VALIDATION
    if len(np.unique(y)) > n_splits:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"\nPerforming {n_splits}-fold Stratified Cross-Validation...")
    else:
        print(f"\nWarning: Not enough samples per class or too many classes for {n_splits}-fold stratification.")
        print("Falling back to standard KFold. Some folds might not contain all classes.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


    results = {
        "Multinomial Naive Bayes (TF-IDF)": [],
        "SVM (TF-IDF)": [],
        "SVM (SBERT)": [],
        "MLPClassifier (SBERT)": []
    }

    
    all_y_true = {model_name: [] for model_name in results.keys()}
    all_y_pred = {model_name: [] for model_name in results.keys()}

    
    print("\nGenerating/Loading SBERT embeddings for the entire dataset (with caching)...")
    all_sbert_embeddings = _get_sbert_embeddings_cached(processed_texts)
    
    fold_num = 1
    for train_index, test_index in kf.split(X, y):
        print(f"\n--- Fold {fold_num}/{n_splits} ---")
        X_train_text, X_test_text = X[train_index], X[test_index]
        y_train_encoded, y_test_encoded = y[train_index], y[test_index]

        print("\n--- Running models with TF-IDF features ---")
        tfidf_matrix_train, tfidf_vectorizer = get_tfidf(X_train_text.tolist())
        tfidf_matrix_test = get_tfidf(X_test_text.tolist(), vec=tfidf_vectorizer)

        
        mnb_model = train_multinomial_nb(tfidf_matrix_train, y_train_encoded)
        mnb_acc, mnb_prec, mnb_rec, mnb_f1, mnb_y_pred = evaluate_baseline_model(
            mnb_model, tfidf_matrix_test, y_test_encoded, model_name="Multinomial Naive Bayes"
        )
        results["Multinomial Naive Bayes (TF-IDF)"].append({"Accuracy": mnb_acc, "F1-Score": mnb_f1})
        all_y_true["Multinomial Naive Bayes (TF-IDF)"].extend(y_test_encoded)
        all_y_pred["Multinomial Naive Bayes (TF-IDF)"].extend(mnb_y_pred)

        
        svm_tfidf_model = train_svm_tfidf(tfidf_matrix_train, y_train_encoded)
        svm_tfidf_acc, svm_tfidf_prec, svm_tfidf_rec, svm_tfidf_f1, svm_tfidf_y_pred = evaluate_baseline_model(
            svm_tfidf_model, tfidf_matrix_test, y_test_encoded, model_name="SVM (TF-IDF) Tuned"
        )
        results["SVM (TF-IDF)"].append({"Accuracy": svm_tfidf_acc, "F1-Score": svm_tfidf_f1})
        all_y_true["SVM (TF-IDF)"].extend(y_test_encoded)
        all_y_pred["SVM (TF-IDF)"].extend(svm_tfidf_y_pred)

        
        print("\n--- Running models with SBERT embeddings ---")
        sbert_embeddings_train = all_sbert_embeddings[train_index]
        sbert_embeddings_test = all_sbert_embeddings[test_index]

        
        svm_sbert_model = train_svm_sbert(sbert_embeddings_train, y_train_encoded)
        svm_sbert_acc, svm_sbert_prec, svm_sbert_rec, svm_sbert_f1, svm_sbert_y_pred = evaluate_semantic_model(
            svm_sbert_model, sbert_embeddings_test, y_test_encoded, model_name="SVM (SBERT) Tuned"
        )
        results["SVM (SBERT)"].append({"Accuracy": svm_sbert_acc, "F1-Score": svm_sbert_f1})
        all_y_true["SVM (SBERT)"].extend(y_test_encoded)
        all_y_pred["SVM (SBERT)"].extend(svm_sbert_y_pred)

       
        mlp_sbert_model = train_mlp_classifier(sbert_embeddings_train, y_train_encoded)
        mlp_sbert_acc, mlp_sbert_prec, mlp_sbert_rec, mlp_sbert_f1, mlp_sbert_y_pred = evaluate_semantic_model(
            mlp_sbert_model, sbert_embeddings_test, y_test_encoded, model_name="MLPClassifier (SBERT)"
        )
        results["MLPClassifier (SBERT)"].append({"Accuracy": mlp_sbert_acc, "F1-Score": mlp_sbert_f1})
        all_y_true["MLPClassifier (SBERT)"].extend(y_test_encoded)
        all_y_pred["MLPClassifier (SBERT)"].extend(mlp_sbert_y_pred)

        fold_num += 1

    print("\n--- Aggregated Cross-Validation Results ---")
    print(f"{'Model':<30} | {'Avg Accuracy':<15} | {'Avg F1-Score':<15}")
    print("-" * 65)
    for model_name, fold_results in results.items():
        if fold_results:
            avg_accuracy = np.mean([res["Accuracy"] for res in fold_results])
            avg_f1_score = np.mean([res["F1-Score"] for res in fold_results])
            print(f"{model_name:<30} | {avg_accuracy:<15.4f} | {avg_f1_score:<15.4f}")
        else:
            print(f"{model_name:<30} | {'N/A':<15} | {'N/A':<15}")

    print("\n--- Generating Confusion Matrices ---")
    os.makedirs(config.PLOTS_DIR, exist_ok=True) 

    for model_name in results.keys():
        if all_y_true[model_name] and all_y_pred[model_name]:
            title = f"Confusion Matrix for {model_name} (Aggregated CV)"
            filename = os.path.join(config.PLOTS_DIR, f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_confusion_matrix.png")
            print(f"Saving {model_name} confusion matrix to {filename}")
            plot_confusion_matrix(
                y_true=all_y_true[model_name],
                y_pred=all_y_pred[model_name],
                labels=label_encoder.transform(class_names), 
                title=title,
                filename=filename
            )
        else:
            print(f"Skipping confusion matrix for {model_name} due to no data.")
