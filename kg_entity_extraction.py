"""
Phase 1: Entity Extraction using spaCy

Extracts named entities from all three OCR versions:
- OLMoCR (clean)
- Legacy ProQuest (raw noisy)
- Legacy ProQuest (regex cleaned)
"""

import json
import spacy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

import config


def load_spacy_model():
    """Load the spaCy transformer model."""
    print(f"Loading spaCy model: {config.SPACY_MODEL}")
    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        print(f"Model {config.SPACY_MODEL} not found. Downloading...")
        from spacy.cli import download
        download(config.SPACY_MODEL)
        nlp = spacy.load(config.SPACY_MODEL)

    # Increase max length for large documents
    nlp.max_length = 2000000
    return nlp


def extract_entities(nlp, text: str, source_name: str) -> Dict[str, Any]:
    """
    Extract named entities from text using spaCy.

    Returns a dict with:
    - entities: list of entity dicts
    - summary: counts by entity type
    - unique_entities: set of unique (text, label) pairs
    """
    print(f"\nProcessing {source_name}...")
    print(f"Text length: {len(text):,} characters")

    # Process in chunks if very large
    doc = nlp(text)

    entities = []
    entity_texts_by_type = defaultdict(list)

    for ent in doc.ents:
        if ent.label_ in config.ENTITY_TYPES:
            entity_data = {
                'text': ent.text.strip(),
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'context': text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)]
            }
            entities.append(entity_data)
            entity_texts_by_type[ent.label_].append(ent.text.strip())

    # Build summary
    summary = {}
    for label in config.ENTITY_TYPES:
        texts = entity_texts_by_type.get(label, [])
        unique_texts = set(texts)
        summary[label] = {
            'total_mentions': len(texts),
            'unique_count': len(unique_texts),
            'top_entities': Counter(texts).most_common(20)
        }

    # Calculate overall stats
    unique_entities = set((e['text'], e['label']) for e in entities)

    result = {
        'source': source_name,
        'total_entities': len(entities),
        'unique_entities': len(unique_entities),
        'entities': entities,
        'summary': summary
    }

    print(f"Found {len(entities)} entity mentions ({len(unique_entities)} unique)")

    return result


def normalize_entity(text: str) -> str:
    """Normalize an entity text for comparison."""
    # Lowercase, remove extra whitespace, strip punctuation
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def find_entity_fragments(olmocr_entities: Dict, legacy_entities: Dict) -> List[Dict]:
    """
    Find entities that are fragmented in legacy OCR but whole in OLMoCR.

    This demonstrates how OCR errors cause entity splitting.
    """
    fragments = []

    # Get unique OLMoCR entities
    olmocr_unique = set(normalize_entity(e['text']) for e in olmocr_entities['entities'])

    # For each legacy entity, check if it's a fragment of an OLMoCR entity
    legacy_texts = [e['text'] for e in legacy_entities['entities']]
    legacy_unique = set(normalize_entity(t) for t in legacy_texts)

    # Find legacy entities not in OLMoCR (potential fragments or errors)
    novel_legacy = legacy_unique - olmocr_unique

    # Find OLMoCR entities that might have fragmented versions in legacy
    for olmocr_ent in olmocr_unique:
        if len(olmocr_ent) < 3:
            continue

        # Check if parts of this entity appear as separate entities in legacy
        potential_fragments = []
        for legacy_ent in novel_legacy:
            # Check if legacy entity is part of OLMoCR entity
            if legacy_ent in olmocr_ent and len(legacy_ent) >= 3:
                potential_fragments.append(legacy_ent)
            # Check if OLMoCR is part of legacy (garbled expansion)
            elif olmocr_ent in legacy_ent:
                potential_fragments.append(legacy_ent)

        if potential_fragments:
            fragments.append({
                'original': olmocr_ent,
                'fragments': potential_fragments
            })

    return fragments[:30]  # Return top 30


def compare_entity_extraction(results: Dict[str, Dict]) -> Dict:
    """
    Compare entity extraction across all three OCR versions.
    """
    comparison = {
        'overview': {},
        'by_type': {},
        'fragmentation': {}
    }

    # Overview comparison
    for source, data in results.items():
        comparison['overview'][source] = {
            'total_mentions': data['total_entities'],
            'unique_entities': data['unique_entities'],
            'vocabulary_size': len(set(e['text'] for e in data['entities']))
        }

    # By type comparison
    for label in config.ENTITY_TYPES:
        comparison['by_type'][label] = {}
        for source, data in results.items():
            if label in data['summary']:
                comparison['by_type'][label][source] = {
                    'mentions': data['summary'][label]['total_mentions'],
                    'unique': data['summary'][label]['unique_count']
                }

    # Find fragmentation (compare OLMoCR to legacy raw)
    if 'olmocr' in results and 'legacy_raw' in results:
        comparison['fragmentation']['raw_vs_olmocr'] = find_entity_fragments(
            results['olmocr'], results['legacy_raw']
        )

    if 'olmocr' in results and 'legacy_cleaned' in results:
        comparison['fragmentation']['cleaned_vs_olmocr'] = find_entity_fragments(
            results['olmocr'], results['legacy_cleaned']
        )

    return comparison


def print_comparison_report(comparison: Dict):
    """Print a human-readable comparison report."""

    print("\n" + "=" * 70)
    print("ENTITY EXTRACTION COMPARISON REPORT")
    print("=" * 70)

    print("\n### OVERVIEW ###")
    print(f"{'Source':<20} {'Total Mentions':<18} {'Unique Entities':<18}")
    print("-" * 60)
    for source, stats in comparison['overview'].items():
        print(f"{source:<20} {stats['total_mentions']:<18} {stats['unique_entities']:<18}")

    print("\n### BY ENTITY TYPE ###")
    for label in config.ENTITY_TYPES:
        if label in comparison['by_type'] and comparison['by_type'][label]:
            print(f"\n{label}:")
            for source, stats in comparison['by_type'][label].items():
                print(f"  {source}: {stats['mentions']} mentions, {stats['unique']} unique")

    if comparison['fragmentation'].get('raw_vs_olmocr'):
        print("\n### ENTITY FRAGMENTATION (Raw Legacy vs OLMoCR) ###")
        print("Examples of entities split/corrupted in legacy OCR:")
        for frag in comparison['fragmentation']['raw_vs_olmocr'][:10]:
            print(f"  OLMoCR: '{frag['original']}'")
            print(f"    Legacy fragments: {frag['fragments'][:5]}")

    print("\n" + "=" * 70)


def main():
    """Run entity extraction on all three OCR versions."""

    print("=" * 60)
    print("Phase 1: Entity Extraction (spaCy NER)")
    print("=" * 60)

    # Load spaCy model
    nlp = load_spacy_model()

    results = {}

    # 1. Process OLMoCR (clean)
    print("\n--- Processing OLMoCR (Clean OCR) ---")
    with open(config.OLMOCR_FILE, 'r', encoding='utf-8') as f:
        olmocr_text = f.read()

    results['olmocr'] = extract_entities(nlp, olmocr_text, "OLMoCR")

    # Save OLMoCR results
    with open(config.ENTITIES_OLMOCR, 'w', encoding='utf-8') as f:
        json.dump(results['olmocr'], f, indent=2, ensure_ascii=False)
    print(f"Saved to: {config.ENTITIES_OLMOCR}")

    # 2. Process Legacy Raw (noisy)
    print("\n--- Processing Legacy ProQuest (Raw) ---")
    with open(config.LEGACY_FILE, 'r', encoding='utf-8') as f:
        legacy_raw_text = f.read()

    results['legacy_raw'] = extract_entities(nlp, legacy_raw_text, "Legacy Raw")

    # Save legacy raw results
    with open(config.ENTITIES_LEGACY_RAW, 'w', encoding='utf-8') as f:
        json.dump(results['legacy_raw'], f, indent=2, ensure_ascii=False)
    print(f"Saved to: {config.ENTITIES_LEGACY_RAW}")

    # 3. Process Legacy Cleaned (if exists)
    if config.LEGACY_CLEANED_FILE.exists():
        print("\n--- Processing Legacy ProQuest (Regex Cleaned) ---")
        with open(config.LEGACY_CLEANED_FILE, 'r', encoding='utf-8') as f:
            legacy_cleaned_text = f.read()

        results['legacy_cleaned'] = extract_entities(nlp, legacy_cleaned_text, "Legacy Cleaned")

        # Save legacy cleaned results
        with open(config.ENTITIES_LEGACY_CLEANED, 'w', encoding='utf-8') as f:
            json.dump(results['legacy_cleaned'], f, indent=2, ensure_ascii=False)
        print(f"Saved to: {config.ENTITIES_LEGACY_CLEANED}")
    else:
        print(f"\nNote: Cleaned legacy file not found at {config.LEGACY_CLEANED_FILE}")
        print("Run kg_legacy_cleanup.py first to create it.")

    # Compare results
    print("\n--- Comparing Results ---")
    comparison = compare_entity_extraction(results)
    print_comparison_report(comparison)

    # Save comparison
    comparison_file = config.OUTPUT_DIR / "entity_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to: {comparison_file}")

    return results


if __name__ == "__main__":
    main()
