#!/usr/bin/env python3
"""
OCR Quality Impact: Knowledge Graph Construction Demo

Main entry point that runs all phases in sequence:
- Phase 0: Legacy OCR Cleanup (regex)
- Phase 1: Entity Extraction (spaCy)
- Phase 2: Relation Extraction (spaCy + Gemini)
- Phase 3: Knowledge Graph Construction
- Phase 4: Visualization

Usage:
    python run_demo.py              # Run all phases
    python run_demo.py --phase 1    # Run specific phase
    python run_demo.py --skip-gemini # Skip Gemini (LLM) phase
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config


def run_phase_0():
    """Phase 0: Legacy OCR Cleanup"""
    print("\n" + "#" * 70)
    print("# PHASE 0: Legacy OCR Cleanup (Regex)")
    print("#" * 70)

    import kg_legacy_cleanup
    return kg_legacy_cleanup.main()


def run_phase_1():
    """Phase 1: Entity Extraction"""
    print("\n" + "#" * 70)
    print("# PHASE 1: Entity Extraction (spaCy NER)")
    print("#" * 70)

    import kg_entity_extraction
    return kg_entity_extraction.main()


def run_phase_2(skip_gemini: bool = False):
    """Phase 2: Relation Extraction"""
    print("\n" + "#" * 70)
    print("# PHASE 2: Relation Extraction")
    print("#" * 70)

    if skip_gemini:
        print("\nSkipping Gemini relation extraction (--skip-gemini flag)")
        print("Only proximity-based relations will be extracted.")
        # Run a modified version that skips Gemini
        import kg_relation_extraction as rel_module

        import spacy
        import json

        nlp = rel_module.load_spacy_model()
        results = {}

        # Process each file with proximity only
        files = [
            (config.OLMOCR_FILE, config.RELATIONS_OLMOCR, 'olmocr'),
            (config.LEGACY_FILE, config.RELATIONS_LEGACY_RAW, 'legacy_raw'),
        ]

        if config.LEGACY_CLEANED_FILE.exists():
            files.append((config.LEGACY_CLEANED_FILE, config.RELATIONS_LEGACY_CLEANED, 'legacy_cleaned'))

        for input_file, output_file, name in files:
            print(f"\nProcessing {name}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()

            proximity = rel_module.extract_proximity_relations(nlp, text, config.MAX_TOKEN_DISTANCE)

            results[name] = {
                'proximity_relations': proximity,
                'semantic_relations': {},
                'stats': {
                    'proximity_count': len(proximity),
                    'semantic_total': 0
                }
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results[name], f, indent=2, ensure_ascii=False)
            print(f"Saved to: {output_file}")

        return results

    else:
        import kg_relation_extraction
        return kg_relation_extraction.main()


def run_phase_3():
    """Phase 3: Knowledge Graph Construction"""
    print("\n" + "#" * 70)
    print("# PHASE 3: Knowledge Graph Construction & Comparison")
    print("#" * 70)

    import kg_build_compare
    return kg_build_compare.main()


def run_phase_4():
    """Phase 4: Visualization"""
    print("\n" + "#" * 70)
    print("# PHASE 4: Visualization")
    print("#" * 70)

    import kg_visualize
    return kg_visualize.main()


def print_summary():
    """Print final summary of outputs."""
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - OUTPUT SUMMARY")
    print("=" * 70)

    print("\n### Data Files ###")
    data_files = [
        config.LEGACY_CLEANED_FILE,
        config.ENTITIES_OLMOCR,
        config.ENTITIES_LEGACY_RAW,
        config.ENTITIES_LEGACY_CLEANED,
        config.RELATIONS_OLMOCR,
        config.RELATIONS_LEGACY_RAW,
        config.RELATIONS_LEGACY_CLEANED,
        config.GRAPH_OLMOCR,
        config.GRAPH_LEGACY_RAW,
        config.GRAPH_LEGACY_CLEANED,
        config.METRICS_FILE,
    ]

    for f in data_files:
        status = "OK" if f.exists() else "MISSING"
        print(f"  [{status}] {f.name}")

    print("\n### Visualization Files ###")
    viz_files = [
        config.NETWORK_COMPARISON_FIG,
        config.ENTITY_EXPLOSION_FIG,
        config.CLEANUP_LIMITS_FIG,
        config.INTERACTIVE_HTML,
    ]

    for f in viz_files:
        status = "OK" if f.exists() else "MISSING"
        print(f"  [{status}] {f.name}")

    print("\n### Key Outputs ###")
    if config.METRICS_FILE.exists():
        import json
        with open(config.METRICS_FILE, 'r') as f:
            metrics = json.load(f)

        print("\nKnowledge Graph Metrics:")
        for source in ['olmocr', 'legacy_raw', 'legacy_cleaned']:
            if source in metrics:
                m = metrics[source]
                print(f"\n  {source.replace('_', ' ').title()}:")
                print(f"    Nodes: {m.get('node_count', 'N/A')}")
                print(f"    Edges: {m.get('edge_count', 'N/A')}")
                print(f"    Components: {m.get('connected_components', 'N/A')}")
                print(f"    Largest Component: {m.get('largest_component_pct', 'N/A')}%")

        if 'fragmentation_ratio_raw' in metrics:
            print(f"\n  Entity Inflation (Legacy/OLMoCR): {metrics['fragmentation_ratio_raw']:.2f}x")

    print("\n" + "=" * 70)
    print("To view interactive visualization, open:")
    print(f"  {config.INTERACTIVE_HTML}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="OCR Quality Impact: Knowledge Graph Construction Demo"
    )
    parser.add_argument(
        '--phase', type=int, choices=[0, 1, 2, 3, 4],
        help='Run specific phase only (0-4)'
    )
    parser.add_argument(
        '--skip-gemini', action='store_true',
        help='Skip Gemini LLM relation extraction (Phase 2)'
    )
    parser.add_argument(
        '--from-phase', type=int, choices=[0, 1, 2, 3, 4],
        help='Start from specific phase (runs all subsequent phases)'
    )

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("OCR Quality Impact: Knowledge Graph Construction Demo")
    print("=" * 70)
    print(f"\nInput files:")
    print(f"  OLMoCR (clean):  {config.OLMOCR_FILE}")
    print(f"  Legacy (noisy):  {config.LEGACY_FILE}")
    print(f"\nOutput directory: {config.OUTPUT_DIR}")

    if args.phase is not None:
        # Run single phase
        phases = {
            0: run_phase_0,
            1: run_phase_1,
            2: lambda: run_phase_2(args.skip_gemini),
            3: run_phase_3,
            4: run_phase_4,
        }
        phases[args.phase]()

    else:
        # Run all phases (or from a specific phase)
        start_phase = args.from_phase if args.from_phase is not None else 0

        if start_phase <= 0:
            run_phase_0()

        if start_phase <= 1:
            run_phase_1()

        if start_phase <= 2:
            run_phase_2(skip_gemini=args.skip_gemini)

        if start_phase <= 3:
            run_phase_3()

        if start_phase <= 4:
            run_phase_4()

        print_summary()

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")


if __name__ == "__main__":
    main()
