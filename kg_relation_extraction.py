"""
Phase 2: Relation Extraction

Extracts relations using two methods:
A) Proximity-based relations (spaCy) - entities within N tokens
B) Semantic relation extraction (Gemini 3 Flash Preview)
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import google.generativeai as genai
import spacy

import config


# Configure Gemini
genai.configure(api_key=config.GOOGLE_API_KEY)


def load_spacy_model():
    """Load spaCy model for proximity analysis."""
    print(f"Loading spaCy model: {config.SPACY_MODEL}")
    nlp = spacy.load(config.SPACY_MODEL)
    nlp.max_length = 2000000
    return nlp


def extract_proximity_relations(nlp, text: str, max_distance: int = 50) -> List[Dict]:
    """
    Extract relations based on entity proximity.

    Entities within max_distance tokens are considered co-occurring.
    """
    doc = nlp(text)
    relations = []

    # Process each sentence
    for sent in doc.sents:
        sent_ents = [ent for ent in sent.ents if ent.label_ in config.ENTITY_TYPES]

        # Find pairs within distance threshold
        for i, ent1 in enumerate(sent_ents):
            for ent2 in sent_ents[i + 1:]:
                distance = abs(ent1.start - ent2.start)

                if distance <= max_distance:
                    relations.append({
                        'source': ent1.text.strip(),
                        'source_type': ent1.label_,
                        'target': ent2.text.strip(),
                        'target_type': ent2.label_,
                        'relation_type': 'CO_OCCURS',
                        'token_distance': distance,
                        'sentence': sent.text[:200]  # First 200 chars for context
                    })

    return relations


def chunk_text_smart(text: str, max_words: int = 800) -> List[str]:
    """
    Split text into chunks for Gemini processing.

    Uses smaller chunks (800 words ≈ 1000 tokens) to ensure we stay well
    under context limits and get complete responses.

    Tries to split on:
    1. Question numbers (Q. 123)
    2. Paragraph boundaries
    3. Sentence boundaries as fallback
    """
    chunks = []

    # First try to split by question numbers (witness testimony structure)
    question_pattern = r'(?=\n\s*(?:Q\.|[0-9]+\.))'
    sections = re.split(question_pattern, text)

    current_chunk = []
    current_words = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_words = len(section.split())

        # If section alone is too big, split by paragraphs
        if section_words > max_words:
            # Save current chunk first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_words = 0

            # Split large section by paragraphs
            paragraphs = re.split(r'\n\s*\n', section)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                para_words = len(para.split())

                if current_words + para_words > max_words:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_words = para_words
                else:
                    current_chunk.append(para)
                    current_words += para_words

        elif current_words + section_words > max_words:
            # Start new chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [section]
            current_words = section_words
        else:
            current_chunk.append(section)
            current_words += section_words

    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_gemini_relations(text: str, source_name: str) -> Dict[str, Any]:
    """
    Extract semantic relations using Gemini 3 Flash Preview.

    Processes text in chunks and extracts structured relations.
    """
    print(f"\nExtracting semantic relations from {source_name} using Gemini...")

    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for consistency
            response_mime_type="application/json",  # Force JSON output
        )
    )

    prompt_template = """You are analyzing a segment from an 1821 British parliamentary report on the timber trade between Britain, Baltic states, and North American colonies.

Extract structured relations from this text. Return valid JSON with these relation types:

{{
  "witness_testimony": [
    {{"witness": "person name", "topic": "what they testified about", "position": "their stance/opinion"}}
  ],
  "commodity_properties": [
    {{"commodity": "timber type", "property": "quality/characteristic", "comparison": "compared to what (if any)"}}
  ],
  "trade_routes": [
    {{"commodity": "what is traded", "source": "origin location", "destination": "destination", "context": "brief note"}}
  ],
  "policy_positions": [
    {{"actor": "who", "position": "supports/opposes/recommends", "policy": "what policy", "reason": "why"}}
  ],
  "economic_facts": [
    {{"fact_type": "duty/price/quantity", "value": "the value", "subject": "what it applies to", "year": "if mentioned"}}
  ]
}}

If a category has no relations, use an empty array [].
Extract ONLY relations explicitly stated in the text - do not infer or make up information.

Text segment:
{text}"""

    chunks = chunk_text_smart(text, max_words=800)
    print(f"Processing {len(chunks)} text chunks (avg {sum(len(c.split()) for c in chunks)//len(chunks)} words each)...")

    all_relations = {
        'witness_testimony': [],
        'commodity_properties': [],
        'trade_routes': [],
        'policy_positions': [],
        'economic_facts': []
    }

    errors = []
    success_count = 0

    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        if len(chunk.strip()) < 50:  # Skip very short chunks
            continue

        prompt = prompt_template.format(text=chunk)

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # With response_mime_type="application/json", response should be valid JSON
            try:
                relations = json.loads(response_text)

                # Merge relations
                for key in all_relations:
                    if key in relations and isinstance(relations[key], list):
                        all_relations[key].extend(relations[key])

                success_count += 1

            except json.JSONDecodeError as e:
                # Try to extract JSON if there's extra text
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    try:
                        relations = json.loads(json_match.group())
                        for key in all_relations:
                            if key in relations and isinstance(relations[key], list):
                                all_relations[key].extend(relations[key])
                        success_count += 1
                    except:
                        errors.append({
                            'chunk': i,
                            'error': f'JSON parse error: {str(e)}',
                            'response_preview': response_text[:500]
                        })
                else:
                    errors.append({
                        'chunk': i,
                        'error': f'JSON parse error: {str(e)}',
                        'response_preview': response_text[:500]
                    })

        except Exception as e:
            error_msg = str(e)
            errors.append({
                'chunk': i,
                'error': error_msg,
                'chunk_preview': chunk[:200]
            })
            # If it's a rate limit error, wait longer
            if 'rate' in error_msg.lower() or '429' in error_msg:
                print(f"\nRate limited, waiting 30s...")
                time.sleep(30)

        # Rate limiting - be gentle with the API
        time.sleep(0.3)

    print(f"\nSuccessfully processed {success_count}/{len(chunks)} chunks")
    if errors:
        print(f"Errors in {len(errors)} chunks")
        # Print first few errors for debugging
        for e in errors[:3]:
            print(f"  - Chunk {e.get('chunk')}: {e.get('error', 'unknown')[:100]}")

    # Deduplicate relations
    for key in all_relations:
        seen = set()
        unique_relations = []
        for rel in all_relations[key]:
            rel_str = json.dumps(rel, sort_keys=True)
            if rel_str not in seen:
                seen.add(rel_str)
                unique_relations.append(rel)
        all_relations[key] = unique_relations

    result = {
        'source': source_name,
        'relations': all_relations,
        'stats': {
            'chunks_processed': len(chunks),
            'chunks_successful': success_count,
            'errors': len(errors),
            'total_relations': sum(len(v) for v in all_relations.values())
        },
        'errors': errors  # Keep all errors for debugging
    }

    return result


def combine_relations(proximity: List[Dict], semantic: Dict) -> Dict:
    """Combine proximity-based and semantic relations."""

    combined = {
        'proximity_relations': proximity,
        'semantic_relations': semantic['relations'],
        'stats': {
            'proximity_count': len(proximity),
            'semantic_total': semantic['stats']['total_relations'],
            'semantic_by_type': {k: len(v) for k, v in semantic['relations'].items()},
            'chunks_processed': semantic['stats']['chunks_processed'],
            'chunks_successful': semantic['stats']['chunks_successful'],
            'gemini_errors': semantic['stats']['errors']
        },
        'errors': semantic.get('errors', [])
    }

    return combined


def print_relation_summary(results: Dict[str, Dict]):
    """Print summary of relation extraction results."""

    print("\n" + "=" * 70)
    print("RELATION EXTRACTION SUMMARY")
    print("=" * 70)

    for source, data in results.items():
        print(f"\n### {source.upper()} ###")

        if 'proximity_relations' in data:
            print(f"  Proximity relations: {len(data['proximity_relations'])}")

        if 'semantic_relations' in data:
            print(f"  Semantic relations:")
            for rtype, rels in data['semantic_relations'].items():
                if rels:
                    print(f"    - {rtype}: {len(rels)}")

        if 'stats' in data:
            stats = data['stats']
            print(f"  Gemini: {stats.get('chunks_successful', 0)}/{stats.get('chunks_processed', 0)} chunks successful")
            if stats.get('gemini_errors', 0) > 0:
                print(f"  Errors: {stats.get('gemini_errors', 0)}")

    print("\n" + "=" * 70)


def main():
    """Run relation extraction on all OCR versions."""

    print("=" * 60)
    print("Phase 2: Relation Extraction")
    print("=" * 60)
    print(f"Using Gemini model: {config.GEMINI_MODEL}")

    # Load spaCy
    nlp = load_spacy_model()

    results = {}

    # 1. Process OLMoCR
    print("\n--- Processing OLMoCR ---")
    with open(config.OLMOCR_FILE, 'r', encoding='utf-8') as f:
        olmocr_text = f.read()

    print("Extracting proximity relations...")
    olmocr_proximity = extract_proximity_relations(nlp, olmocr_text, config.MAX_TOKEN_DISTANCE)
    print(f"Found {len(olmocr_proximity)} proximity relations")

    print("Extracting semantic relations with Gemini...")
    olmocr_semantic = extract_gemini_relations(olmocr_text, "OLMoCR")

    results['olmocr'] = combine_relations(olmocr_proximity, olmocr_semantic)

    with open(config.RELATIONS_OLMOCR, 'w', encoding='utf-8') as f:
        json.dump(results['olmocr'], f, indent=2, ensure_ascii=False)
    print(f"Saved to: {config.RELATIONS_OLMOCR}")

    # 2. Process Legacy Raw
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

    # 3. Process Legacy Cleaned (if exists)
    if config.LEGACY_CLEANED_FILE.exists():
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
