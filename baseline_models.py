from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import config

def get_tfidf(texts, vec=None):

    if vec:
        tfidf_matrix = vec.transform(texts)
        return tfidf_matrix
    else:
        vec = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.8)
        tfidf_matrix = vec.fit_transform(texts)
        return tfidf_matrix, vec

def train_multinomial_nb(X_train, y_train):
    
    mnb_model = MultinomialNB()
    print("\nTraining Multinomial Naive Bayes model...")
    mnb_model.fit(X_train, y_train)
    return mnb_model

def train_svm_tfidf(X_train, y_train):
    
    print("\nTraining SVM (TF-IDF) model with hyperparameter tuning...")
    svm_model = SVC(random_state=42)

    if config.USE_RANDOMIZED_SEARCH:
        param_searcher = RandomizedSearchCV(
            svm_model,
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
            svm_model,
            config.SVM_PARAM_GRID_GRIDSEARCH,
            cv=3,
            scoring='f1_weighted',
            n_jobs=config.N_JOBS,
            verbose=1
        )
    
    param_searcher.fit(X_train, y_train)

    print(f"Best parameters for SVM (TF-IDF): {param_searcher.best_params_}")
    print(f"Best cross-validation F1-score for SVM (TF-IDF): {param_searcher.best_score_:.4f}")
    return param_searcher.best_estimator_ 

def evaluate_model(mnb_model, X_test, y_test, model_name="Model"):
    
    print(f"Evaluating {model_name}...")
    y_pred = mnb_model.predict(X_test)

    print(f"\n--- {model_name} Classification Report ---")
    
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    return accuracy, precision, recall, f1, y_pred 

