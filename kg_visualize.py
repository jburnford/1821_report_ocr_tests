"""
Phase 4: Visualization

Creates visual comparisons of the knowledge graphs:
1. Side-by-side network diagrams
2. Entity explosion bar chart
3. Cleanup limits chart
4. Interactive HTML visualization
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

import config


# Color scheme for entity types
ENTITY_COLORS = {
    'PERSON': '#e41a1c',      # Red
    'ORG': '#377eb8',         # Blue
    'GPE': '#4daf4a',         # Green
    'LOC': '#4daf4a',         # Green (same as GPE)
    'DATE': '#984ea3',        # Purple
    'MONEY': '#ff7f00',       # Orange
    'PRODUCT': '#ffff33',     # Yellow
    'NORP': '#a65628',        # Brown
    'TOPIC': '#f781bf',       # Pink
    'PROPERTY': '#999999',    # Gray
    'POLICY': '#66c2a5',      # Teal
    'UNKNOWN': '#cccccc',     # Light gray
}


def load_graph(filepath: Path) -> nx.Graph:
    """Load a graph from GraphML file."""
    return nx.read_graphml(filepath)


def load_metrics(filepath: Path) -> Dict:
    """Load metrics comparison from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_network_comparison(graphs: Dict[str, nx.Graph], output_path: Path):
    """
    Create side-by-side network visualizations.

    Shows "tight network" (OLMoCR) vs "hairball" (Legacy Raw) vs "partial cleanup" (Legacy Cleaned)
    """
    n_graphs = len(graphs)
    fig, axes = plt.subplots(1, n_graphs, figsize=(8 * n_graphs, 8))

    if n_graphs == 1:
        axes = [axes]

    for ax, (name, G) in zip(axes, graphs.items()):
        # Filter to largest connected component for cleaner visualization
        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            if components:
                largest = max(components, key=len)
                subgraph = G.subgraph(largest).copy()
            else:
                subgraph = G
        else:
            subgraph = G

        # Limit nodes for visualization (take highest degree nodes)
        if subgraph.number_of_nodes() > 100:
            degrees = dict(subgraph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
            subgraph = subgraph.subgraph(top_nodes).copy()

        # Get node colors based on type
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('label', 'UNKNOWN')
            node_colors.append(ENTITY_COLORS.get(node_type, '#cccccc'))

        # Layout
        if subgraph.number_of_nodes() > 0:
            pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

            # Draw
            nx.draw_networkx_nodes(
                subgraph, pos, ax=ax,
                node_color=node_colors,
                node_size=100,
                alpha=0.7
            )
            nx.draw_networkx_edges(
                subgraph, pos, ax=ax,
                edge_color='gray',
                alpha=0.3,
                width=0.5
            )

            # Only label high-degree nodes
            labels = {}
            for node in subgraph.nodes():
                if subgraph.degree(node) >= 3:
                    labels[node] = node[:20]  # Truncate long names

            nx.draw_networkx_labels(
                subgraph, pos, labels, ax=ax,
                font_size=6,
                font_color='black'
            )

        # Title with stats
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        components = nx.number_connected_components(G) if total_nodes > 0 else 0

        title = f"{name.replace('_', ' ').title()}\n"
        title += f"Nodes: {total_nodes}, Edges: {total_edges}, Components: {components}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved network comparison to: {output_path}")


def create_entity_explosion_chart(metrics: Dict, output_path: Path):
    """
    Create bar chart showing entity explosion from OCR errors.

    Shows node counts by type across all three OCR versions.
    """
    sources = [k for k in metrics.keys() if not k.startswith('fragmentation') and not k.startswith('node_diff')]

    # Get all entity types
    all_types = set()
    for src in sources:
        if 'nodes_by_type' in metrics[src]:
            all_types.update(metrics[src]['nodes_by_type'].keys())

    # Filter to main entity types
    main_types = [t for t in all_types if t in config.ENTITY_TYPES]

    if not main_types:
        main_types = list(all_types)[:8]  # Fallback

    # Prepare data
    x = range(len(main_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    offsets = [-width, 0, width]

    for i, (src, color) in enumerate(zip(sources, colors[:len(sources)])):
        counts = [metrics[src].get('nodes_by_type', {}).get(t, 0) for t in main_types]
        bars = ax.bar(
            [xi + offsets[i] for xi in x],
            counts,
            width,
            label=src.replace('_', ' ').title(),
            color=color,
            alpha=0.8
        )

    ax.set_xlabel('Entity Type', fontsize=12)
    ax.set_ylabel('Number of Unique Entities', fontsize=12)
    ax.set_title('Entity Explosion: How OCR Quality Affects Entity Extraction', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(main_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    if 'fragmentation_ratio_raw' in metrics:
        ratio = metrics['fragmentation_ratio_raw']
        ax.annotate(
            f'Entity Inflation: {ratio:.1f}x\n(Legacy vs OLMoCR)',
            xy=(0.98, 0.98),
            xycoords='axes fraction',
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved entity explosion chart to: {output_path}")


def create_cleanup_limits_chart(metrics: Dict, output_path: Path):
    """
    Create chart showing what regex cleanup can and cannot fix.

    Compares metrics across Raw Legacy → Cleaned Legacy → OLMoCR
    """
    # Metrics to compare
    metric_labels = [
        ('node_count', 'Total Nodes'),
        ('connected_components', 'Connected Components'),
        ('orphan_pct', 'Orphan Nodes %'),
        ('avg_degree', 'Avg Node Degree'),
        ('largest_component_pct', 'Largest Component %'),
    ]

    sources = ['legacy_raw', 'legacy_cleaned', 'olmocr']
    source_labels = ['Legacy Raw', 'Legacy + Regex', 'OLMoCR']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Orange, Green

    # Filter to available sources
    available_sources = [s for s in sources if s in metrics]
    available_labels = [source_labels[sources.index(s)] for s in available_sources]
    available_colors = [colors[sources.index(s)] for s in available_sources]

    if len(available_sources) < 2:
        print("Not enough data for cleanup limits chart")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(metric_labels):
        ax = axes[idx]

        values = [metrics[src].get(metric_key, 0) for src in available_sources]

        bars = ax.bar(
            available_labels, values,
            color=available_colors,
            alpha=0.8
        )

        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_label)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.1f}' if isinstance(val, float) else str(val),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9
            )

        ax.tick_params(axis='x', rotation=15)

    # Hide extra subplot
    axes[5].axis('off')

    # Add summary text
    axes[5].text(
        0.5, 0.7,
        "Key Insight:\n\n"
        "Regex cleanup improves metrics,\n"
        "but cannot match modern OCR.\n\n"
        "The gap between 'Legacy + Regex'\n"
        "and 'OLMoCR' represents what\n"
        "post-hoc cleanup cannot fix.",
        ha='center', va='center',
        fontsize=11,
        transform=axes[5].transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    plt.suptitle(
        'Limits of Regex Cleanup: What 2015-era Python Could (and Could Not) Fix',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cleanup limits chart to: {output_path}")


def create_interactive_html(graphs: Dict[str, nx.Graph], output_path: Path):
    """
    Create an interactive HTML visualization using pyvis.

    Creates a single HTML file with tabs or dropdown for different graphs.
    """
    # Use the OLMoCR graph as the primary visualization
    if 'olmocr' in graphs:
        G = graphs['olmocr']
        title = "OLMoCR Knowledge Graph (Clean OCR)"
    elif graphs:
        name = list(graphs.keys())[0]
        G = graphs[name]
        title = f"{name} Knowledge Graph"
    else:
        print("No graphs to visualize")
        return

    # Create pyvis network
    net = Network(
        height='800px',
        width='100%',
        bgcolor='#ffffff',
        font_color='#333333',
        directed=False
    )

    # Set physics options for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 150,
                "updateInterval": 25
            }
        },
        "nodes": {
            "font": {"size": 12}
        },
        "edges": {
            "color": {"inherit": true},
            "smooth": {"enabled": true, "type": "continuous"}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    # Limit nodes for performance
    if G.number_of_nodes() > 200:
        # Take subgraph of highest-degree nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:200]
        G = G.subgraph(top_nodes).copy()

    # Add nodes
    for node in G.nodes():
        attrs = G.nodes[node]
        node_type = attrs.get('label', 'UNKNOWN')
        color = ENTITY_COLORS.get(node_type, '#cccccc')
        mentions = attrs.get('mentions', 1)

        # Size based on mentions/degree
        size = min(10 + G.degree(node) * 3, 40)

        net.add_node(
            node,
            label=node[:30],  # Truncate for display
            title=f"{node}\nType: {node_type}\nMentions: {mentions}\nConnections: {G.degree(node)}",
            color=color,
            size=size
        )

    # Add edges
    for u, v, attrs in G.edges(data=True):
        rel_type = attrs.get('relation_type', 'RELATED')
        weight = attrs.get('weight', 1)

        net.add_edge(
            u, v,
            title=rel_type,
            width=min(1 + weight * 0.5, 5)
        )

    # Save HTML
    net.save_graph(str(output_path))
    print(f"Saved interactive HTML to: {output_path}")

    # Add legend to HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background: white; padding: 15px;
                border: 1px solid #ccc; border-radius: 5px; z-index: 1000; font-family: Arial;">
        <h4 style="margin-top: 0;">Entity Types</h4>
    """
    for entity_type, color in ENTITY_COLORS.items():
        if entity_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PRODUCT']:
            legend_html += f'<div><span style="color: {color}; font-size: 20px;">●</span> {entity_type}</div>'

    legend_html += """
        <hr>
        <small>Hover over nodes for details.<br>Scroll to zoom, drag to pan.</small>
    </div>
    """

    html_content = html_content.replace('<body>', f'<body>{legend_html}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """Generate all visualizations."""

    print("=" * 60)
    print("Phase 4: Visualization")
    print("=" * 60)

    # Load graphs
    graphs = {}
    graph_files = [
        (config.GRAPH_OLMOCR, 'olmocr'),
        (config.GRAPH_LEGACY_RAW, 'legacy_raw'),
        (config.GRAPH_LEGACY_CLEANED, 'legacy_cleaned'),
    ]

    for filepath, name in graph_files:
        if filepath.exists():
            print(f"Loading {name} graph...")
            graphs[name] = load_graph(filepath)
        else:
            print(f"Warning: {filepath} not found")

    if not graphs:
        print("ERROR: No graphs found. Run previous phases first.")
        return

    # Load metrics
    if config.METRICS_FILE.exists():
        metrics = load_metrics(config.METRICS_FILE)
    else:
        print("Warning: Metrics file not found, computing from graphs...")
        from kg_build_compare import compute_graph_metrics
        metrics = {name: compute_graph_metrics(G) for name, G in graphs.items()}

    # Create visualizations
    print("\n--- Creating Network Comparison ---")
    create_network_comparison(graphs, config.NETWORK_COMPARISON_FIG)

    print("\n--- Creating Entity Explosion Chart ---")
    create_entity_explosion_chart(metrics, config.ENTITY_EXPLOSION_FIG)

    print("\n--- Creating Cleanup Limits Chart ---")
    create_cleanup_limits_chart(metrics, config.CLEANUP_LIMITS_FIG)

    print("\n--- Creating Interactive HTML ---")
    create_interactive_html(graphs, config.INTERACTIVE_HTML)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {config.NETWORK_COMPARISON_FIG}")
    print(f"  - {config.ENTITY_EXPLOSION_FIG}")
    print(f"  - {config.CLEANUP_LIMITS_FIG}")
    print(f"  - {config.INTERACTIVE_HTML}")

    return graphs, metrics


if __name__ == "__main__":
    main()
