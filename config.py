import os


DATA_PATH = "c:\\path\\to\\your\\data"

CACHE_DIR = os.path.join("cache")
PROCESSED_TEXT_CACHE = os.path.join(CACHE_DIR, "processed_texts.pkl")
SBERT_EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "sbert_embeddings.pkl")


PLOTS_DIR = os.path.join("plots")

SPACY_MODEL = "de_core_news_sm"


SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


USE_GPU_FOR_SBERT = True


N_JOBS = -1

N_SPLITS_CROSS_VALIDATION = 5


USE_RANDOMIZED_SEARCH = True

N_ITER_RANDOMIZED_SEARCH = 5 

SVM_PARAM_GRID_GRIDSEARCH = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
SVM_PARAM_DISTRIBUTIONS_RANDOMIZEDSEARCH = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf'] 
}


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
