"""
Phase 0: Regex Cleanup of Legacy OCR (2015-era Python)

Simulates what a historian could achieve with only Python regex tools in 2015.
Creates a third comparison point: Raw Legacy → Cleaned Legacy → OLMoCR.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import config


def remove_proquest_artifacts(text: str) -> str:
    """Remove ProQuest copyright notices and metadata artifacts."""

    # Remove copyright notices (multiple formats)
    patterns = [
        r'House of Commons Parliamentary Papers Online\.\s*\n?Copyright \(c\) \d{4} ProQuest Information and Learning Company\. All rights reserved\.',
        r'\[?[Hh]ouse of Commons.*?reserved\.?\]?',
        r'Copyright \(c\) \d{4} ProQuest.*?reserved\.',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Remove common artifacts like "186." that appear randomly
    text = re.sub(r'\b186\.\s*', '', text)

    # Remove document reference numbers
    text = re.sub(r'\bA \d{5}\b', '', text)

    return text


def fix_scattered_letters(text: str) -> str:
    """Fix letters scattered across multiple lines (common OCR error)."""

    # Fix "F I R S T" → "FIRST" patterns
    # This is very hard to do reliably without context

    # Fix obvious scattered headers
    text = re.sub(r'\bl\s*\n\s*FIR\s*\n\s*R\s*\n\s*ST\s*\n\s*EPORT', 'FIRST REPORT', text)
    text = re.sub(r'R\s*\n\s*E\s*\n\s*P\s*\n\s*O\s*\n\s*R\s*\n\s*T\.', 'REPORT.', text)
    text = re.sub(r'FUOK\s+Tlik', 'FROM THE', text)
    text = re.sub(r'APPENDig\s+OP\s+ACCOUNTS', 'APPENDIX OF ACCOUNTS', text)

    return text


def fix_character_substitutions(text: str) -> str:
    """Fix common OCR character substitutions (long-s, similar shapes)."""

    # Pattern: (regex, replacement)
    substitutions = [
        # 'li' misread as various characters
        (r'\btlie\b', 'the'),
        (r'\bTliE\b', 'THE'),
        (r'\bTlie\b', 'The'),
        (r'\bliave\b', 'have'),
        (r'\bwliich\b', 'which'),
        (r'\btliat\b', 'that'),
        (r'\bwitli\b', 'with'),
        (r'\btliis\b', 'this'),
        (r'\btlse\b', 'the'),
        (r'\bth\'\b', 'the'),
        (r'\bth\'e\b', 'the'),

        # 'o' and '0' confusions
        (r'\btho\b(?=\s+[a-z])', 'the'),
        (r'\bQf\b', 'of'),
        (r'\bOr\b(?=\s+the\b)', 'of'),
        (r'\bof\s*,\s*', 'of '),

        # 'a' and 'aa' confusions
        (r'\baa\b', 'as'),
        (r'\bak\b(?=\s+fur)', 'as'),

        # Common word errors
        (r'\bencowragement\b', 'encouragement'),
        (r'\bduiks\b', 'duties'),
        (r'\bBaltie\b', 'Baltic'),
        (r'\bSeppngs\b', 'Seppings'),
        (r'\bship,owners\b', 'ship-owners'),
        (r'\bsliip-owners\b', 'ship-owners'),
        (r'\bintrodece\b', 'introduce'),
        (r'\bpiblie\b', 'public'),
        (r'\baclegnate\b', 'adequate'),
        (r'\bproceed;\b', 'proceeds,'),
        (r'\binterksts\b', 'interests'),
        (r'\btunount\b', 'amount'),
        (r'\boriginelly\b', 'originally'),
        (r'\briods\b', 'periods'),
        (r'\bproviously\b', 'previously'),
        (r'\bSwill\b', 'will'),
        (r'\b0\b(?=\s+be\s+Printed)', 'to'),
        (r'\b18t1\b', '1821'),
        (r'\b1409-10\b', '1809-10'),

        # Fix ordinals with quotes
        (r'\b49"\b', '49th'),
        (r'\b50"\b', '50th'),
        (r'\b59"\b', '59th'),
        (r'\b49\'\'\b', '49th'),
        (r'\b50\'\'\b', '50th'),

        # Fix common garbled phrases
        (r'\btime CO tune\b', 'time to time'),
        (r'\bpersona\b', 'persons'),
        (r'\bfile:\b', 'the'),

        # Fix semicolon in copyright
        (r';opyright', 'Copyright'),
        (r':opyright', 'Copyright'),
    ]

    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE if pattern.islower() else 0)

    return text


def fix_broken_words(text: str) -> str:
    """Rejoin words broken across lines."""

    # Rejoin hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Fix words split by newlines without hyphen (harder - requires patterns)
    # These are specific observed cases
    patterns = [
        (r'\bwell-\s*foundOd\b', 'well-founded'),
        (r'\bpoint-?of-?view\b', 'point of view'),
        (r'\bpoint•of\s*view\b', 'point of view'),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and remove stray characters."""

    # Multiple spaces → single
    text = re.sub(r'  +', ' ', text)

    # Fix bullet as hyphen
    text = re.sub(r'•', '-', text)

    # Remove stray single characters on lines (common OCR artifact)
    text = re.sub(r'^\s*[a-zA-Z•\-]\s*$', '', text, flags=re.MULTILINE)

    # Remove lines with only punctuation or numbers
    text = re.sub(r'^\s*[\d\.\,\;\:\-\•]+\s*$', '', text, flags=re.MULTILINE)

    # Collapse multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


def fix_punctuation(text: str) -> str:
    """Fix common punctuation errors."""

    # Fix semicolons that should be colons
    # Note: Context-dependent, so we're conservative

    # Fix common patterns
    text = re.sub(r'\s+;\s*', '; ', text)  # Normalize semicolon spacing
    text = re.sub(r'\s+,\s*', ', ', text)  # Normalize comma spacing

    # Fix "it-hears" → "it bears"
    text = re.sub(r'\bit-hears\b', 'it bears', text)

    return text


def domain_specific_fixes(text: str) -> str:
    """Apply domain-specific corrections for 1821 timber trade document."""

    fixes = [
        # Known entities
        (r'\bOrTHE\b', 'OF THE'),
        (r'\b\'COMMITTEE\b', 'COMMITTEE'),
        (r'\b,APPOINTED\b', 'APPOINTED'),
        (r'\bGeo\.\s*3\b', 'Geo. III'),
        (r'\bWhale\b(?=\s+of\s+these\s+duties)', 'whole'),
        (r'\bki e\b', 'the'),
        (r'\bhy\b(?=\s+the\s+acts)', 'by'),
        (r'\b4,\b(?=\s+just)', 'a'),
        (r'\bdur\b(?=\s+North)', 'our'),
        (r'\b31\.\s*5s\b', '£1. 5s'),
        (r'\b21\.\s*ls\b', '£1. 1s'),
        (r'\b2i\.\s*1,44\.\s*lid\b', '£1. 14s. 8d'),
        (r'\b35\s*percent\b', '25 per cent'),
        (r'\bexemptio,u\b', 'exemption'),
        (r'\btendency\.,\b', 'tendency,'),

        # Section headers
        (r'\bktS\b', ''),
        (r'\bInk\b', ''),
        (r'\b2e\b', ''),
        (r'\[louse\b', 'House'),
    ]

    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)

    return text


def cleanup_legacy_ocr(text: str) -> str:
    """
    Main cleanup pipeline for legacy OCR text.

    Applies all cleanup functions in sequence.
    Returns cleaned text.
    """

    # Phase 1: Remove artifacts
    text = remove_proquest_artifacts(text)

    # Phase 2: Fix scattered letters
    text = fix_scattered_letters(text)

    # Phase 3: Fix character substitutions
    text = fix_character_substitutions(text)

    # Phase 4: Fix broken words
    text = fix_broken_words(text)

    # Phase 5: Domain-specific fixes
    text = domain_specific_fixes(text)

    # Phase 6: Fix punctuation
    text = fix_punctuation(text)

    # Phase 7: Normalize whitespace (do last)
    text = normalize_whitespace(text)

    return text


def analyze_cleanup_results(original: str, cleaned: str) -> Dict:
    """Analyze what the cleanup accomplished and what remains broken."""

    def count_words(text):
        return len(re.findall(r'\b\w+\b', text))

    def count_unique_words(text):
        return len(set(re.findall(r'\b\w+\b', text.lower())))

    original_words = count_words(original)
    cleaned_words = count_words(cleaned)
    original_unique = count_unique_words(original)
    cleaned_unique = count_unique_words(cleaned)

    # Find remaining OCR errors (words with unusual patterns)
    # Words with internal capitals, numbers mixed with letters, etc.
    remaining_errors = []
    words = set(re.findall(r'\b\w+\b', cleaned))

    for word in words:
        # Skip short words
        if len(word) < 3:
            continue
        # Check for mixed case in middle (like "OrTHE")
        if re.search(r'[a-z][A-Z]', word) and word not in ['McDougall', 'McGhie']:
            remaining_errors.append(word)
        # Check for numbers mixed with letters (except dates)
        if re.search(r'[a-zA-Z]\d|\d[a-zA-Z]', word) and not re.match(r'^\d{4}$', word):
            remaining_errors.append(word)

    return {
        'original_word_count': original_words,
        'cleaned_word_count': cleaned_words,
        'original_unique_words': original_unique,
        'cleaned_unique_words': cleaned_unique,
        'vocabulary_reduction': original_unique - cleaned_unique,
        'vocabulary_reduction_pct': round((original_unique - cleaned_unique) / original_unique * 100, 1),
        'remaining_likely_errors': sorted(set(remaining_errors))[:50],  # Top 50
    }


def main():
    """Run the legacy OCR cleanup pipeline."""

    print("=" * 60)
    print("Phase 0: Legacy OCR Cleanup (2015-era Python Regex)")
    print("=" * 60)

    # Load legacy OCR
    print(f"\nLoading legacy OCR from: {config.LEGACY_FILE}")
    with open(config.LEGACY_FILE, 'r', encoding='utf-8') as f:
        original_text = f.read()

    print(f"Original text length: {len(original_text):,} characters")

    # Run cleanup
    print("\nRunning cleanup pipeline...")
    cleaned_text = cleanup_legacy_ocr(original_text)

    print(f"Cleaned text length: {len(cleaned_text):,} characters")

    # Save cleaned text
    print(f"\nSaving cleaned text to: {config.LEGACY_CLEANED_FILE}")
    with open(config.LEGACY_CLEANED_FILE, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    # Analyze results
    print("\nAnalyzing cleanup results...")
    analysis = analyze_cleanup_results(original_text, cleaned_text)

    print("\n" + "-" * 40)
    print("CLEANUP ANALYSIS")
    print("-" * 40)
    print(f"Original word count: {analysis['original_word_count']:,}")
    print(f"Cleaned word count:  {analysis['cleaned_word_count']:,}")
    print(f"Original unique words: {analysis['original_unique_words']:,}")
    print(f"Cleaned unique words:  {analysis['cleaned_unique_words']:,}")
    print(f"Vocabulary reduction: {analysis['vocabulary_reduction']:,} ({analysis['vocabulary_reduction_pct']}%)")

    if analysis['remaining_likely_errors']:
        print(f"\nSample remaining errors ({len(analysis['remaining_likely_errors'])} found):")
        for error in analysis['remaining_likely_errors'][:20]:
            print(f"  - {error}")

    print("\n" + "=" * 60)
    print("What Regex CANNOT Fix:")
    print("=" * 60)
    print("""
- Severely garbled text like "canna ansivet the proportieti"
- Context-dependent corrections (needs dictionary/NLP)
- Entity name variations without clear patterns
- Table/column misalignment
- Mathematical notation and currency symbols
- Proper noun misspellings without known patterns
""")

    print("\nCleanup complete!")
    return cleaned_text


if __name__ == "__main__":
    main()
