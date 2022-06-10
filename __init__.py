import os
from dotenv import load_dotenv

load_dotenv()
SEQ2SEQ_MODEL_PATH = os.getenv('SEQ2SEQ_MODEL_PATH')
CACHE = os.getenv('CACHE')
RUNS_DIR = os.getenv('RUNS_DIR')
