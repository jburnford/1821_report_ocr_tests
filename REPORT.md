# Garbage In, Garbage Out: OCR Quality Impact on Knowledge Graph Construction in 2026

## Executive Summary

This study compares knowledge graphs constructed from the same historical document using two OCR sources: **OLMoCR** (state-of-the-art 2024 VLM-based OCR) and **Legacy ProQuest** (commercial OCR from ~2010). We used both traditional NLP (spaCy NER) and LLM-based extraction (Gemini) to build knowledge graphs from an 1821 British Parliamentary report on the timber trade (791 pages).

**Key Finding:** LLMs are surprisingly robust to OCR noise for *relation extraction*, but knowledge *graph structure* still suffers significantly from poor OCR quality. The value of clean OCR in 2026 lies not in what the LLM understands, but in what the graph can connect.

---

## Research Questions

1. **How much better does LLM-based extraction perform compared to proximity-based methods?**
2. **How much does OCR quality impact knowledge graph construction with each method?**
3. **In 2026, with powerful LLMs that can "read through" noise, does OCR quality still matter?**

---

## Methodology

### Data Sources

| Source | Description | Quality |
|--------|-------------|---------|
| **OLMoCR** | Qwen2-VL 7B fine-tuned OCR (2024) | High - ~99% character accuracy |
| **Legacy ProQuest** | Commercial OCR (~2010) | Moderate - common substitution errors |

Both sources contain the same document: *Report from the Select Committee on the Timber Trade* (1821), 791 pages of witness testimony, tables, and analysis.

### Extraction Pipeline

1. **Phase 0:** Legacy OCR cleanup (regex-based correction of common errors)
2. **Phase 1:** Entity extraction using spaCy `en_core_web_trf`
3. **Phase 2:** Relation extraction using:
   - Proximity-based (entities within 50 tokens)
   - Semantic (Gemini 3 Flash extracting structured relations)
4. **Phase 3:** Knowledge graph construction (NetworkX)
5. **Phase 4:** Comparative analysis and visualization

### Relation Types Extracted by Gemini

- `witness_testimony` - Who testified about what topic
- `trade_routes` - Commodity flows between locations
- `policy_positions` - Political stances on trade policy
- `economic_facts` - Prices, duties, quantities
- `commodity_properties` - Quality comparisons of timber types

---

## Results

### 1. LLM Robustness to OCR Noise (Surprising!)

Gemini extracted nearly identical relation counts from clean and noisy text:

| Relation Type | OLMoCR | Legacy Raw | Ratio |
|---------------|--------|------------|-------|
| witness_testimony | 373 | 375 | 1.01x |
| commodity_properties | 537 | 563 | 1.05x |
| trade_routes | 524 | 521 | 0.99x |
| policy_positions | 291 | 286 | 0.98x |
| economic_facts | 925 | 937 | 1.01x |
| **TOTAL** | **2,650** | **2,682** | **1.01x** |

**Implication:** Modern LLMs can "read through" OCR noise to extract semantic meaning. The understanding layer is noise-tolerant.

### 2. Graph Structure Degradation (Expected)

Despite similar relation counts, the graph structure differs significantly:

| Metric | OLMoCR | Legacy Raw | Impact |
|--------|--------|------------|--------|
| Total Nodes | 1,471 | 1,779 | **+21% inflation** |
| Total Edges | 5,067 | 4,670 | -8% connectivity |
| Connected Components | 169 | 305 | **+80% fragmentation** |
| Largest Component | 83.1% | 78.0% | -5.1% reach |
| Orphan Nodes | 125 (8.5%) | 253 (14.2%) | **+102% isolation** |
| Average Degree | 6.89 | 5.25 | -24% connectivity |

**Implication:** OCR errors create duplicate entities that fragment the graph into disconnected islands.

### 3. The Entity Resolution Nightmare

OCR errors create multiple representations of the same entity:

**Witness Name Variants (Legacy):**
```
'Edward Solly, Esq.' vs 'Edward Sully, Esq.' (94% similar)
'Thomas Simon' vs 'Thomas Simson' (97% similar)
'Jan Caldwell, Esq.' vs 'John Caldwell, Esq.' (92% similar)
'James Borthwick' vs 'fanscs Bortlarick, Esq.' (mangled)
'William Tindall' vs 'William Tidal' vs 'William Timid' (OCR variants)
```

**Location Variants:**
```
'America' vs 'Ainerica' vs 'Aineriea'
'Committee' vs 'Comniittee' vs 'Coinniittee'
```

**Date Variants:**
```
'1820' vs 'i82o' vs '1 820'
'1795' vs 'x795'
```

### 4. Downstream Query Impact

**Query: "Find all people who testified about timber"**

- OLMoCR: Clean traversal from PERSON → TESTIFIED_ABOUT → timber topics
- Legacy: Results fragmented across "timber" / "tirnber" / "tiniber" variants

**Query: "What trade routes connected Canada?"**

- OLMoCR: Direct paths from canonical "Canada" node
- Legacy: Must query "Canada" + "Cariada" + "Cauada" + all variants

Without entity resolution, queries return **incomplete results**.

### 5. Quantifying the Entity Resolution Burden

| Metric | Value |
|--------|-------|
| Legacy-only nodes (not in OLMoCR) | 1,063 |
| Low-degree legacy-only (likely duplicates) | 711 |
| Manual review time (30 sec each) | 5.9 hours |
| **At Canadiana scale (60M pages)** | **~54 million duplicates** |

---

## The 2026 Verdict

### What Has Changed

In 2026, large language models have fundamentally changed the OCR quality calculus:

1. **LLMs extract relations robustly from noisy text** - Gemini achieved 1.01x the relation count from legacy OCR
2. **Understanding is no longer the bottleneck** - The semantic layer is noise-tolerant
3. **Entity recognition still fails on garbage input** - NER sees "Srnith" and "Smith" as different people

### What Hasn't Changed

1. **Graph algorithms require clean entities** - Centrality, path queries, and community detection all fail on fragmented data
2. **Entity resolution remains hard** - Fuzzy matching needs context, risks false merges
3. **Duplicates compound at scale** - 711 duplicates per 791 pages → millions at collection scale

### The New Value Proposition

| Approach | Relation Extraction | Graph Structure | Ongoing Cost |
|----------|--------------------|-----------------|--------------|
| Legacy OCR + LLM | ✅ Works well | ❌ Fragmented | 🔄 Entity resolution forever |
| Clean OCR + LLM | ✅ Works well | ✅ Connected | ✅ One-time investment |

**The value of clean OCR in 2026 is not in what the LLM understands—it's in what the graph can connect.**

---

## Recommendations

### For New Digitization Projects

1. **Invest in quality OCR upfront** - OLMoCR at ~$190/million pages is cheaper than ongoing entity resolution
2. **Evaluate OCR quality on graph metrics**, not just character accuracy
3. **Test downstream tasks** - Query completeness matters more than WER

### For Existing Legacy Collections

1. **Consider selective re-OCR** of high-value documents
2. **Build entity resolution pipelines** if re-OCR is infeasible
3. **Use LLM-assisted resolution** but validate results
4. **Document known variants** for query expansion

### For Knowledge Graph Construction

1. **LLM extraction is viable** even on noisy text for relations
2. **Entity canonicalization is critical** before graph construction
3. **Graph connectivity metrics** (components, orphans) reveal OCR impact
4. **Test queries on both clean and noisy** to quantify practical impact

---

## Interactive Visualization

An interactive comparison of the knowledge graphs is available at:

**https://jburnford.github.io/1821_report_ocr_tests/**

Toggle between OLMoCR (clean) and Legacy (noisy) to see:
- Node connectivity differences
- Orphan node distribution
- OCR garbage entities (red)
- Suspicious legacy-only entities (orange)

---

## Data Files

| File | Description |
|------|-------------|
| `output/graph_olmocr.graphml` | Clean OCR knowledge graph |
| `output/graph_legacy_raw.graphml` | Legacy OCR knowledge graph |
| `output/relations_olmocr.json` | Extracted relations (clean) |
| `output/relations_legacy_raw.json` | Extracted relations (noisy) |
| `output/metrics_comparison.json` | Comparative statistics |

---

## Citation

```
@misc{ocr_kg_2026,
  title={Garbage In, Garbage Out: OCR Quality Impact on Knowledge Graph Construction},
  author={Burnford, Jim and Claude},
  year={2026},
  url={https://github.com/jburnford/1821_report_ocr_tests}
}
```

---

## Acknowledgments

- **OLMoCR** by Allen AI for state-of-the-art document OCR
- **Gemini** by Google for robust LLM-based relation extraction
- Source document: UK Parliamentary Papers, 1821

---

*Report generated January 2026*
