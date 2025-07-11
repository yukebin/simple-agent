import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Project Constants
CHROMA_DIR = "db"  # Directory for persistent vector store
CACHE_PATH = os.path.join(CHROMA_DIR, "vitalbridge_raw_pages.json")

# FireCrawl Settings
TARGET_WEBSITE = os.getenv("TARGET_WEBSITE", "https://www.vitalbridge.com/")

# Embedding Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# LLM Settings
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.0