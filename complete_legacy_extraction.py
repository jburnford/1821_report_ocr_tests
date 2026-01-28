#!/usr/bin/env python3
"""
Complete the Gemini semantic relation extraction for legacy sources only.
OLMoCR is already complete.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from kg_relation_extraction import (
    load_spacy_model,
    extract_proximity_relations,
    extract_gemini_relations,
    combine_relations,
    print_relation_summary
)


def main():
    print("=" * 60)
    print("Completing Legacy Relation Extraction")
    print("=" * 60)
    print(f"Using Gemini model: {config.GEMINI_MODEL}")

    # Load spaCy
    nlp = load_spacy_model()

    results = {}

    # 1. Process Legacy Raw
    print("\n--- Processing Legacy Raw ---")
    with open(config.LEGACY_FILE, 'r', encoding='utf-8') as f:
        legacy_raw_text = f.read()

    print("Extracting proximity relations...")
    legacy_raw_proximity = extract_proximity_relations(nlp, legacy_raw_text, config.MAX_TOKEN_DISTANCE)
    print(f"Found {len(legacy_raw_proximity)} proximity relations")

    print("Extracting semantic relations with Gemini...")
    legacy_raw_semantic = extract_gemini_relations(legacy_raw_text, "Legacy Raw")

    results['legacy_raw'] = combine_relations(legacy_raw_proximity, legacy_raw_semantic)

    with open(config.RELATIONS_LEGACY_RAW, 'w', encoding='utf-8') as f:
        json.dump(results['legacy_raw'], f, indent=2, ensure_ascii=False)
    print(f"Saved to: {config.RELATIONS_LEGACY_RAW}")

    # 2. Process Legacy Cleaned
    print("\n--- Processing Legacy Cleaned ---")
    with open(config.LEGACY_CLEANED_FILE, 'r', encoding='utf-8') as f:
        legacy_cleaned_text = f.read()

    print("Extracting proximity relations...")
    legacy_cleaned_proximity = extract_proximity_relations(nlp, legacy_cleaned_text, config.MAX_TOKEN_DISTANCE)
    print(f"Found {len(legacy_cleaned_proximity)} proximity relations")

    print("Extracting semantic relations with Gemini...")
    legacy_cleaned_semantic = extract_gemini_relations(legacy_cleaned_text, "Legacy Cleaned")

    results['legacy_cleaned'] = combine_relations(legacy_cleaned_proximity, legacy_cleaned_semantic)

    with open(config.RELATIONS_LEGACY_CLEANED, 'w', encoding='utf-8') as f:
        json.dump(results['legacy_cleaned'], f, indent=2, ensure_ascii=False)
    print(f"Saved to: {config.RELATIONS_LEGACY_CLEANED}")

    # Print summary
    print_relation_summary(results)

    return results


if __name__ == "__main__":
    main()
