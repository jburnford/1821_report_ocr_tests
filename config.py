"""Configuration for OCR Knowledge Graph Demo"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path("/home/jic823/emory/kg_ocr_demo")
INPUT_DIR = Path("/home/jic823/emory")
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = BASE_DIR / "figures"

# Input files
OLMOCR_FILE = INPUT_DIR / "olmocr_ocr.txt"
LEGACY_FILE = INPUT_DIR / "legacy_proquest.txt"

# Output files
LEGACY_CLEANED_FILE = OUTPUT_DIR / "legacy_cleaned.txt"

ENTITIES_OLMOCR = OUTPUT_DIR / "entities_olmocr.json"
ENTITIES_LEGACY_RAW = OUTPUT_DIR / "entities_legacy_raw.json"
ENTITIES_LEGACY_CLEANED = OUTPUT_DIR / "entities_legacy_cleaned.json"

RELATIONS_OLMOCR = OUTPUT_DIR / "relations_olmocr.json"
RELATIONS_LEGACY_RAW = OUTPUT_DIR / "relations_legacy_raw.json"
RELATIONS_LEGACY_CLEANED = OUTPUT_DIR / "relations_legacy_cleaned.json"

GRAPH_OLMOCR = OUTPUT_DIR / "graph_olmocr.graphml"
GRAPH_LEGACY_RAW = OUTPUT_DIR / "graph_legacy_raw.graphml"
GRAPH_LEGACY_CLEANED = OUTPUT_DIR / "graph_legacy_cleaned.graphml"

METRICS_FILE = OUTPUT_DIR / "metrics_comparison.json"

# Figure outputs
NETWORK_COMPARISON_FIG = FIGURES_DIR / "network_comparison_3way.png"
ENTITY_EXPLOSION_FIG = FIGURES_DIR / "entity_explosion.png"
CLEANUP_LIMITS_FIG = FIGURES_DIR / "cleanup_limits.png"
INTERACTIVE_HTML = FIGURES_DIR / "kg_interactive.html"

# API Keys - set via environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# SpaCy model
SPACY_MODEL = "en_core_web_trf"

# Gemini model - use latest preview
GEMINI_MODEL = "gemini-3-flash-preview"

# Entity types to extract
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "NORP", "LOC"]

# Relation extraction settings
MAX_TOKEN_DISTANCE = 50  # Max tokens between entities for proximity relation
CHUNK_SIZE = 1500  # Tokens per chunk for Gemini

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
