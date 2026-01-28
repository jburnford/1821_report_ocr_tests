"""
Phase 3: Knowledge Graph Construction and Comparison

Builds NetworkX graphs from extracted entities and relations,
then computes comparison metrics.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import networkx as nx

import config


def load_entities(filepath: Path) -> Dict:
    """Load entity extraction results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_relations(filepath: Path) -> Dict:
    """Load relation extraction results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_node_name(name: str) -> str:
    """Normalize entity name for node creation."""
    if name is None:
        return ''
    import re
    # Clean up whitespace
    name = ' '.join(name.split())
    # Remove leading/trailing punctuation
    name = name.strip('.,;:!?\'"()[]{}')
    return name


def build_knowledge_graph(
    entities_data: Dict,
    relations_data: Dict,
    source_name: str
) -> nx.Graph:
    """
    Build a NetworkX graph from entities and relations.

    Nodes = entities (with type label)
    Edges = relations (both proximity and semantic)
    """
    G = nx.Graph()
    G.graph['source'] = source_name

    # Add entity nodes
    entity_counts = Counter()
    for ent in entities_data['entities']:
        node_name = normalize_node_name(ent['text'])
        if len(node_name) < 2:  # Skip single characters
            continue

        if node_name not in G:
            G.add_node(
                node_name,
                label=ent['label'],
                type='entity',
                mentions=1
            )
        else:
            G.nodes[node_name]['mentions'] = G.nodes[node_name].get('mentions', 0) + 1

        entity_counts[ent['label']] += 1

    # Add edges from proximity relations
    proximity_edges = 0
    for rel in relations_data.get('proximity_relations', []):
        source = normalize_node_name(rel['source'])
        target = normalize_node_name(rel['target'])

        if source in G and target in G and source != target:
            if G.has_edge(source, target):
                G[source][target]['weight'] = G[source][target].get('weight', 1) + 1
            else:
                G.add_edge(
                    source, target,
                    relation_type='CO_OCCURS',
                    weight=1
                )
            proximity_edges += 1

    # Add edges from semantic relations
    semantic_edges = 0
    semantic_rels = relations_data.get('semantic_relations', {})

    # Process witness testimony
    for rel in semantic_rels.get('witness_testimony', []):
        witness = normalize_node_name(rel.get('witness', ''))
        topic = normalize_node_name(rel.get('topic', ''))

        if witness and topic and witness in G:
            # Create topic node if needed
            if topic not in G and len(topic) > 2:
                G.add_node(topic, label='TOPIC', type='topic')

            if topic in G and witness != topic:
                G.add_edge(witness, topic, relation_type='TESTIFIED_ABOUT', weight=1)
                semantic_edges += 1

    # Process trade routes
    for rel in semantic_rels.get('trade_routes', []):
        source_loc = normalize_node_name(rel.get('source', ''))
        dest_loc = normalize_node_name(rel.get('destination', ''))
        commodity = normalize_node_name(rel.get('commodity', ''))

        # Add edge between locations if both exist
        if source_loc in G and dest_loc in G and source_loc != dest_loc:
            G.add_edge(source_loc, dest_loc, relation_type='TRADE_ROUTE', weight=1)
            semantic_edges += 1

        # Connect commodity to locations
        if commodity in G:
            if source_loc in G and commodity != source_loc:
                G.add_edge(commodity, source_loc, relation_type='SOURCED_FROM', weight=1)
                semantic_edges += 1
            if dest_loc in G and commodity != dest_loc:
                G.add_edge(commodity, dest_loc, relation_type='TRADED_TO', weight=1)
                semantic_edges += 1

    # Process commodity properties
    for rel in semantic_rels.get('commodity_properties', []):
        commodity = normalize_node_name(rel.get('commodity', ''))
        property_val = normalize_node_name(rel.get('property', ''))

        if commodity in G and property_val and len(property_val) > 2:
            if property_val not in G:
                G.add_node(property_val, label='PROPERTY', type='property')
            if property_val in G and commodity != property_val:
                G.add_edge(commodity, property_val, relation_type='HAS_PROPERTY', weight=1)
                semantic_edges += 1

    # Process policy positions
    for rel in semantic_rels.get('policy_positions', []):
        actor = normalize_node_name(rel.get('actor', ''))
        policy = normalize_node_name(rel.get('policy', ''))
        position = rel.get('position', 'neutral')

        if actor in G and policy and len(policy) > 2:
            if policy not in G:
                G.add_node(policy, label='POLICY', type='policy')
            if policy in G and actor != policy:
                G.add_edge(actor, policy, relation_type=f'POSITION_{position.upper()}', weight=1)
                semantic_edges += 1

    print(f"Built graph for {source_name}:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()} (proximity: {proximity_edges}, semantic: {semantic_edges})")

    return G


def compute_graph_metrics(G: nx.Graph) -> Dict:
    """Compute various graph metrics for comparison."""

    # Basic counts
    metrics = {
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges(),
    }

    # Node type breakdown
    node_types = Counter()
    for node, attrs in G.nodes(data=True):
        node_types[attrs.get('label', 'UNKNOWN')] += 1
    metrics['nodes_by_type'] = dict(node_types)

    # Connected components
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        metrics['connected_components'] = len(components)

        # Largest component
        if components:
            largest = max(components, key=len)
            metrics['largest_component_size'] = len(largest)
            metrics['largest_component_pct'] = round(len(largest) / G.number_of_nodes() * 100, 1)
        else:
            metrics['largest_component_size'] = 0
            metrics['largest_component_pct'] = 0

        # Orphan nodes (degree 0)
        orphans = [n for n in G.nodes() if G.degree(n) == 0]
        metrics['orphan_nodes'] = len(orphans)
        metrics['orphan_pct'] = round(len(orphans) / G.number_of_nodes() * 100, 1) if G.number_of_nodes() > 0 else 0
    else:
        metrics['connected_components'] = 0
        metrics['largest_component_size'] = 0
        metrics['largest_component_pct'] = 0
        metrics['orphan_nodes'] = 0
        metrics['orphan_pct'] = 0

    # Degree statistics
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = round(sum(degrees) / len(degrees), 2) if degrees else 0
        metrics['max_degree'] = max(degrees) if degrees else 0

        # Density
        metrics['density'] = round(nx.density(G), 4)
    else:
        metrics['avg_degree'] = 0
        metrics['max_degree'] = 0
        metrics['density'] = 0

    # Edge type breakdown
    edge_types = Counter()
    for u, v, attrs in G.edges(data=True):
        edge_types[attrs.get('relation_type', 'UNKNOWN')] += 1
    metrics['edges_by_type'] = dict(edge_types)

    return metrics


def compare_graphs(graphs: Dict[str, nx.Graph]) -> Dict:
    """
    Compare metrics across multiple graphs.
    """
    comparison = {}

    for name, G in graphs.items():
        comparison[name] = compute_graph_metrics(G)

    # Calculate fragmentation ratio (legacy vs OLMoCR)
    if 'olmocr' in comparison and 'legacy_raw' in comparison:
        olmocr_nodes = comparison['olmocr']['node_count']
        legacy_nodes = comparison['legacy_raw']['node_count']
        if olmocr_nodes > 0:
            comparison['fragmentation_ratio_raw'] = round(legacy_nodes / olmocr_nodes, 2)

    if 'olmocr' in comparison and 'legacy_cleaned' in comparison:
        olmocr_nodes = comparison['olmocr']['node_count']
        legacy_nodes = comparison['legacy_cleaned']['node_count']
        if olmocr_nodes > 0:
            comparison['fragmentation_ratio_cleaned'] = round(legacy_nodes / olmocr_nodes, 2)

    return comparison


def print_comparison_table(comparison: Dict):
    """Print a formatted comparison table."""

    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH COMPARISON")
    print("=" * 80)

    sources = [k for k in comparison.keys() if not k.startswith('fragmentation')]

    # Header
    header = f"{'Metric':<30}"
    for src in sources:
        header += f"{src:<15}"
    print(header)
    print("-" * 80)

    # Metrics to display
    metrics_to_show = [
        ('node_count', 'Total Nodes'),
        ('edge_count', 'Total Edges'),
        ('connected_components', 'Connected Components'),
        ('largest_component_pct', 'Largest Component %'),
        ('orphan_nodes', 'Orphan Nodes'),
        ('orphan_pct', 'Orphan %'),
        ('avg_degree', 'Avg Degree'),
        ('density', 'Density'),
    ]

    for metric_key, metric_label in metrics_to_show:
        row = f"{metric_label:<30}"
        for src in sources:
            val = comparison[src].get(metric_key, 'N/A')
            if isinstance(val, float):
                row += f"{val:<15.2f}"
            else:
                row += f"{str(val):<15}"
        print(row)

    # Node type breakdown
    print("\n" + "-" * 80)
    print("NODES BY TYPE:")
    all_types = set()
    for src in sources:
        all_types.update(comparison[src].get('nodes_by_type', {}).keys())

    for node_type in sorted(all_types):
        row = f"  {node_type:<28}"
        for src in sources:
            val = comparison[src].get('nodes_by_type', {}).get(node_type, 0)
            row += f"{val:<15}"
        print(row)

    # Fragmentation ratios
    if 'fragmentation_ratio_raw' in comparison:
        print(f"\nFragmentation Ratio (Raw Legacy / OLMoCR): {comparison['fragmentation_ratio_raw']}")
    if 'fragmentation_ratio_cleaned' in comparison:
        print(f"Fragmentation Ratio (Cleaned Legacy / OLMoCR): {comparison['fragmentation_ratio_cleaned']}")

    print("=" * 80)


def find_node_differences(g1: nx.Graph, g2: nx.Graph, name1: str, name2: str) -> Dict:
    """Find nodes that exist in one graph but not the other."""

    nodes1 = set(g1.nodes())
    nodes2 = set(g2.nodes())

    only_in_1 = nodes1 - nodes2
    only_in_2 = nodes2 - nodes1
    common = nodes1 & nodes2

    return {
        f'only_in_{name1}': list(only_in_1)[:50],  # First 50
        f'only_in_{name2}': list(only_in_2)[:50],
        'common_count': len(common),
        f'{name1}_unique_count': len(only_in_1),
        f'{name2}_unique_count': len(only_in_2)
    }


def main():
    """Build and compare knowledge graphs."""

    print("=" * 60)
    print("Phase 3: Knowledge Graph Construction & Comparison")
    print("=" * 60)

    graphs = {}

    # Check for required files
    required_files = [
        (config.ENTITIES_OLMOCR, config.RELATIONS_OLMOCR, 'olmocr'),
        (config.ENTITIES_LEGACY_RAW, config.RELATIONS_LEGACY_RAW, 'legacy_raw'),
    ]

    optional_files = [
        (config.ENTITIES_LEGACY_CLEANED, config.RELATIONS_LEGACY_CLEANED, 'legacy_cleaned'),
    ]

    # Process required files
    for ent_file, rel_file, name in required_files:
        if not ent_file.exists():
            print(f"ERROR: Missing {ent_file}")
            print("Run kg_entity_extraction.py first.")
            return

        if not rel_file.exists():
            print(f"ERROR: Missing {rel_file}")
            print("Run kg_relation_extraction.py first.")
            return

        print(f"\nBuilding graph for {name}...")
        entities = load_entities(ent_file)
        relations = load_relations(rel_file)
        graphs[name] = build_knowledge_graph(entities, relations, name)

    # Process optional files
    for ent_file, rel_file, name in optional_files:
        if ent_file.exists() and rel_file.exists():
            print(f"\nBuilding graph for {name}...")
            entities = load_entities(ent_file)
            relations = load_relations(rel_file)
            graphs[name] = build_knowledge_graph(entities, relations, name)

    # Compare graphs
    print("\n--- Comparing Graphs ---")
    comparison = compare_graphs(graphs)
    print_comparison_table(comparison)

    # Find node differences
    if 'olmocr' in graphs and 'legacy_raw' in graphs:
        print("\n--- Node Differences (OLMoCR vs Legacy Raw) ---")
        diff = find_node_differences(graphs['olmocr'], graphs['legacy_raw'], 'olmocr', 'legacy_raw')
        print(f"Common nodes: {diff['common_count']}")
        print(f"Only in OLMoCR: {diff['olmocr_unique_count']}")
        print(f"Only in Legacy Raw: {diff['legacy_raw_unique_count']}")

        if diff['only_in_olmocr']:
            print(f"\nSample nodes only in OLMoCR: {diff['only_in_olmocr'][:10]}")
        if diff['only_in_legacy_raw']:
            print(f"Sample nodes only in Legacy Raw: {diff['only_in_legacy_raw'][:10]}")

        comparison['node_differences_raw'] = diff

    # Save graphs
    print("\n--- Saving Graphs ---")
    for name, G in graphs.items():
        if name == 'olmocr':
            filepath = config.GRAPH_OLMOCR
        elif name == 'legacy_raw':
            filepath = config.GRAPH_LEGACY_RAW
        elif name == 'legacy_cleaned':
            filepath = config.GRAPH_LEGACY_CLEANED
        else:
            continue

        nx.write_graphml(G, filepath)
        print(f"Saved {name} graph to: {filepath}")

    # Save metrics comparison
    with open(config.METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {config.METRICS_FILE}")

    return graphs, comparison


if __name__ == "__main__":
    main()
