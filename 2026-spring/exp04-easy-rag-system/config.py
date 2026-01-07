# Milvus Lite Configuration
MILVUS_LITE_DATA_PATH = "C:/Users/lh008/Desktop/exp04-easy-rag-system/milvus_lite_data.db"
COLLECTION_NAME = "medical_rag_lite"

# Data Configuration
DATA_FILE = r"C:\Users\lh008\Desktop\exp04-easy-rag-system\Datasets\Corpus\medical.json"

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
EMBEDDING_DIM = 384

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 3
INDEX_METRIC_TYPE = "L2"
INDEX_TYPE = "IVF_FLAT"
INDEX_PARAMS = {"nlist": 128}
SEARCH_PARAMS = {"nprobe": 16}

# Generation Parameters
MAX_NEW_TOKENS_GEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Global map to store document content (populated during indexing)
id_to_doc_map = {}
