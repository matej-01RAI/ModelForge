import os
from dotenv import load_dotenv

load_dotenv()

AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY")
AZURE_AI_MODEL = os.getenv("AZURE_AI_MODEL", "claude-sonnet")
AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview")

WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspaces")
