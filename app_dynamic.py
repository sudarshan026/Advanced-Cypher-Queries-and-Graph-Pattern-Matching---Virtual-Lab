"""
Virtual Laboratory Experiment: Advanced Cypher Queries and Graph Pattern Matching
DYNAMIC VERSION — with custom graph builder + Groq AI natural language queries.

Sections:
  1. Theory: Concepts, objectives, procedure, and terminology.
  2. Simulation: Graph Builder | Cypher Query Lab | AI Assistant (Groq).
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded trials, and downloadable PDF report.

Graph experiments use in-memory simulation with pyvis visualization.
Neo4j implementation is NOT used — all queries run against an in-memory graph.
"""

import os
import json
import traceback
from datetime import datetime
from collections import defaultdict, deque
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from pyvis.network import Network
import streamlit.components.v1 as components

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Advanced Cypher Queries and Graph Pattern Matching",
    "objectives": [
        "Understand Cypher query language syntax for nodes, relationships, and properties.",
        "Build custom knowledge graphs with labeled nodes and typed relationships.",
        "Perform multi-hop graph traversals to discover indirect connections.",
        "Apply filtering, aggregation (COUNT, COLLECT), and ordering on graph data.",
        "Execute complex graph pattern matching queries with variable-length paths.",
        "Use AI-assisted natural language querying to explore graph structures.",
        "Efficiently retrieve complex relationships and hidden connections in a knowledge graph."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles

**Cypher** is a declarative graph query language originally developed for **Neo4j** and now
standardized as **GQL (Graph Query Language)** by ISO. It uses ASCII-art style syntax to
represent graph patterns — making it intuitive to express complex relationships.

#### Why Graph Pattern Matching?

Traditional relational databases use JOINs to connect tables, which becomes increasingly
expensive as the number of hops grows. Graph databases store relationships as first-class
citizens, enabling **constant-time traversal** regardless of dataset size. This makes them
ideal for:

- **Social network analysis** — friends-of-friends, influence propagation
- **Recommendation engines** — "users who liked X also liked Y"
- **Fraud detection** — discovering hidden rings of suspicious transactions
- **Knowledge graphs** — traversing ontologies and semantic relationships

#### Core Cypher Concepts

| Concept | Syntax | Description |
|---------|--------|-------------|
| **Node** | `(n:Label {prop: value})` | An entity (person, movie, etc.) |
| **Relationship** | `-[r:TYPE {prop: value}]->` | Directed edge between nodes |
| **Pattern Matching** | `MATCH (a)-[r]->(b)` | Find subgraphs matching the pattern |
| **Filtering** | `WHERE a.name = 'Alice'` | Filter by property conditions |
| **Return** | `RETURN a.name, r.type` | Specify output columns |
| **Aggregation** | `COUNT(*)`, `COLLECT()`, `AVG()` | Summarize across matches |
| **Variable-Length Paths** | `(a)-[*1..3]->(b)` | Match paths of 1 to 3 hops |
| **Optional Match** | `OPTIONAL MATCH` | Left-outer-join style matching |

### Workflow & System Overview

1. **Graph Construction**: Build a knowledge graph with labeled nodes and typed relationships.
2. **Query Formulation**: Write Cypher queries or use natural language with AI assistance.
3. **Traversal & Filtering**: Execute multi-hop traversals with property filters.
4. **Aggregation & Analysis**: Use COUNT, COLLECT, and other aggregations to summarize results.
5. **Pattern Discovery**: Identify hidden connections through complex pattern matching.
    """,

    "procedure": [
        "Step 1: Review the theoretical background on Cypher syntax and graph patterns.",
        "Step 2: Navigate to the **Simulation** section using the sidebar.",
        "Step 3: Use the **Graph Studio** tab to build a custom graph or load a template.",
        "Step 4: Switch to the **Cypher Query Lab** tab to run predefined or custom queries.",
        "Step 5: Use the **AI Assistant** tab to ask questions in natural language (requires Groq API key).",
        "Step 6: Experiment with different query types: basic match, multi-hop, aggregation, complex patterns.",
        "Step 7: Record at least 3–4 distinct query trials using the **Record Trial** button.",
        "Step 8: Complete the **Quiz** to assess your understanding of Cypher concepts.",
        "Step 9: Generate and download your **Lab Report** from the Report Generation section."
    ],

    "key_terms": {
        "Node (Vertex)": "A fundamental entity in a graph — e.g., a Person, Movie, or City.",
        "Relationship (Edge)": "A directed connection between two nodes — e.g., ACTED_IN, FRIENDS_WITH.",
        "Label": "A tag to categorize nodes — e.g., :Person, :Movie. A node can have multiple labels.",
        "Property": "A key-value pair on a node or relationship — e.g., name: 'Alice', year: 2020.",
        "MATCH Clause": "The primary read clause in Cypher; specifies a graph pattern to search for.",
        "WHERE Clause": "Filters matched results based on boolean conditions on properties.",
        "RETURN Clause": "Specifies the output columns/values from a query.",
        "Multi-hop Traversal": "Following a chain of relationships across multiple nodes.",
        "Variable-Length Path": "A path pattern like [*1..3] matching paths 1–3 hops long.",
        "Aggregation": "Functions like COUNT(), SUM(), COLLECT() that summarize across matches."
    }
}


# ======================================================================================
# 2. TEMPLATE GRAPHS
# ======================================================================================

def get_movie_graph():
    """Movie knowledge graph template."""
    nodes = [
        {"id": "alice",  "label": "Person",   "properties": {"name": "Alice",       "age": 32, "city": "Mumbai"}},
        {"id": "bob",    "label": "Person",   "properties": {"name": "Bob",         "age": 28, "city": "Delhi"}},
        {"id": "carol",  "label": "Person",   "properties": {"name": "Carol",       "age": 35, "city": "Mumbai"}},
        {"id": "dave",   "label": "Person",   "properties": {"name": "Dave",        "age": 41, "city": "Bangalore"}},
        {"id": "eve",    "label": "Person",   "properties": {"name": "Eve",         "age": 26, "city": "Chennai"}},
        {"id": "frank",  "label": "Person",   "properties": {"name": "Frank",       "age": 38, "city": "Delhi"}},
        {"id": "m1",     "label": "Movie",    "properties": {"title": "GraphWorld",  "year": 2021, "genre": "Sci-Fi"}},
        {"id": "m2",     "label": "Movie",    "properties": {"title": "Query Quest", "year": 2019, "genre": "Adventure"}},
        {"id": "m3",     "label": "Movie",    "properties": {"title": "Node Noir",   "year": 2022, "genre": "Thriller"}},
        {"id": "m4",     "label": "Movie",    "properties": {"title": "Edge of Logic","year": 2020, "genre": "Drama"}},
        {"id": "dir1",   "label": "Director", "properties": {"name": "Raj Kumar",    "awards": 3}},
        {"id": "dir2",   "label": "Director", "properties": {"name": "Meera Nair",   "awards": 5}},
    ]
    edges = [
        {"from": "alice","to": "m1", "type": "ACTED_IN",     "properties": {"role": "Lead"}},
        {"from": "alice","to": "m2", "type": "ACTED_IN",     "properties": {"role": "Supporting"}},
        {"from": "bob",  "to": "m1", "type": "ACTED_IN",     "properties": {"role": "Supporting"}},
        {"from": "bob",  "to": "m3", "type": "ACTED_IN",     "properties": {"role": "Lead"}},
        {"from": "carol","to": "m2", "type": "ACTED_IN",     "properties": {"role": "Lead"}},
        {"from": "carol","to": "m4", "type": "ACTED_IN",     "properties": {"role": "Lead"}},
        {"from": "dave", "to": "m3", "type": "ACTED_IN",     "properties": {"role": "Supporting"}},
        {"from": "dave", "to": "m4", "type": "ACTED_IN",     "properties": {"role": "Lead"}},
        {"from": "eve",  "to": "m1", "type": "ACTED_IN",     "properties": {"role": "Cameo"}},
        {"from": "frank","to": "m2", "type": "ACTED_IN",     "properties": {"role": "Lead"}},
        {"from": "frank","to": "m3", "type": "ACTED_IN",     "properties": {"role": "Supporting"}},
        {"from": "dir1", "to": "m1", "type": "DIRECTED",     "properties": {}},
        {"from": "dir1", "to": "m3", "type": "DIRECTED",     "properties": {}},
        {"from": "dir2", "to": "m2", "type": "DIRECTED",     "properties": {}},
        {"from": "dir2", "to": "m4", "type": "DIRECTED",     "properties": {}},
        {"from": "alice","to": "bob",   "type": "FRIENDS_WITH","properties": {"since": 2018}},
        {"from": "bob",  "to": "carol", "type": "FRIENDS_WITH","properties": {"since": 2019}},
        {"from": "carol","to": "dave",  "type": "FRIENDS_WITH","properties": {"since": 2017}},
        {"from": "dave", "to": "eve",   "type": "FRIENDS_WITH","properties": {"since": 2020}},
        {"from": "alice","to": "frank", "type": "FRIENDS_WITH","properties": {"since": 2021}},
        {"from": "eve",  "to": "frank", "type": "FRIENDS_WITH","properties": {"since": 2022}},
        {"from": "alice","to": "m3", "type": "REVIEWED",  "properties": {"rating": 4.5}},
        {"from": "bob",  "to": "m2", "type": "REVIEWED",  "properties": {"rating": 3.8}},
        {"from": "carol","to": "m1", "type": "REVIEWED",  "properties": {"rating": 4.9}},
        {"from": "eve",  "to": "m4", "type": "REVIEWED",  "properties": {"rating": 4.2}},
        {"from": "frank","to": "m1", "type": "REVIEWED",  "properties": {"rating": 4.0}},
    ]
    return nodes, edges


def get_social_network_graph():
    """Social network template."""
    nodes = [
        {"id": "u1", "label": "User", "properties": {"name": "Priya",   "age": 25, "interests": "AI"}},
        {"id": "u2", "label": "User", "properties": {"name": "Arjun",   "age": 27, "interests": "Web Dev"}},
        {"id": "u3", "label": "User", "properties": {"name": "Sneha",   "age": 24, "interests": "Data Science"}},
        {"id": "u4", "label": "User", "properties": {"name": "Rohan",   "age": 29, "interests": "AI"}},
        {"id": "u5", "label": "User", "properties": {"name": "Kavya",   "age": 23, "interests": "Cybersecurity"}},
        {"id": "p1", "label": "Post", "properties": {"title": "Intro to Graphs",  "likes": 42}},
        {"id": "p2", "label": "Post", "properties": {"title": "Neo4j Tips",       "likes": 78}},
        {"id": "p3", "label": "Post", "properties": {"title": "Python for ML",    "likes": 115}},
        {"id": "g1", "label": "Group","properties": {"name": "AI Enthusiasts",    "members": 320}},
        {"id": "g2", "label": "Group","properties": {"name": "Graph DB Club",     "members": 150}},
    ]
    edges = [
        {"from": "u1", "to": "u2", "type": "FOLLOWS",     "properties": {}},
        {"from": "u1", "to": "u3", "type": "FOLLOWS",     "properties": {}},
        {"from": "u2", "to": "u4", "type": "FOLLOWS",     "properties": {}},
        {"from": "u3", "to": "u1", "type": "FOLLOWS",     "properties": {}},
        {"from": "u4", "to": "u5", "type": "FOLLOWS",     "properties": {}},
        {"from": "u5", "to": "u1", "type": "FOLLOWS",     "properties": {}},
        {"from": "u1", "to": "p1", "type": "POSTED",      "properties": {}},
        {"from": "u2", "to": "p2", "type": "POSTED",      "properties": {}},
        {"from": "u3", "to": "p3", "type": "POSTED",      "properties": {}},
        {"from": "u4", "to": "p1", "type": "LIKED",       "properties": {}},
        {"from": "u5", "to": "p2", "type": "LIKED",       "properties": {}},
        {"from": "u1", "to": "p3", "type": "LIKED",       "properties": {}},
        {"from": "u3", "to": "p2", "type": "LIKED",       "properties": {}},
        {"from": "u1", "to": "g1", "type": "MEMBER_OF",   "properties": {"role": "Admin"}},
        {"from": "u2", "to": "g2", "type": "MEMBER_OF",   "properties": {"role": "Member"}},
        {"from": "u3", "to": "g1", "type": "MEMBER_OF",   "properties": {"role": "Member"}},
        {"from": "u4", "to": "g1", "type": "MEMBER_OF",   "properties": {"role": "Member"}},
        {"from": "u5", "to": "g2", "type": "MEMBER_OF",   "properties": {"role": "Admin"}},
    ]
    return nodes, edges


def get_academic_graph():
    """Academic/university knowledge graph template."""
    nodes = [
        {"id": "s1",  "label": "Student",    "properties": {"name": "Amit",     "semester": 6, "gpa": 8.5}},
        {"id": "s2",  "label": "Student",    "properties": {"name": "Neha",     "semester": 6, "gpa": 9.1}},
        {"id": "s3",  "label": "Student",    "properties": {"name": "Rahul",    "semester": 4, "gpa": 7.8}},
        {"id": "s4",  "label": "Student",    "properties": {"name": "Divya",    "semester": 8, "gpa": 9.4}},
        {"id": "c1",  "label": "Course",     "properties": {"name": "DBMS",     "credits": 4}},
        {"id": "c2",  "label": "Course",     "properties": {"name": "ML",       "credits": 4}},
        {"id": "c3",  "label": "Course",     "properties": {"name": "Networks", "credits": 3}},
        {"id": "pr1", "label": "Professor",  "properties": {"name": "Dr. Sharma",  "dept": "CSE"}},
        {"id": "pr2", "label": "Professor",  "properties": {"name": "Dr. Gupta",   "dept": "CSE"}},
        {"id": "d1",  "label": "Department", "properties": {"name": "CSE",         "hod": "Dr. Verma"}},
    ]
    edges = [
        {"from": "s1", "to": "c1", "type": "ENROLLED_IN", "properties": {"grade": "A"}},
        {"from": "s1", "to": "c2", "type": "ENROLLED_IN", "properties": {"grade": "A+"}},
        {"from": "s2", "to": "c1", "type": "ENROLLED_IN", "properties": {"grade": "A+"}},
        {"from": "s2", "to": "c3", "type": "ENROLLED_IN", "properties": {"grade": "B+"}},
        {"from": "s3", "to": "c2", "type": "ENROLLED_IN", "properties": {"grade": "B"}},
        {"from": "s3", "to": "c3", "type": "ENROLLED_IN", "properties": {"grade": "A"}},
        {"from": "s4", "to": "c1", "type": "ENROLLED_IN", "properties": {"grade": "A+"}},
        {"from": "s4", "to": "c2", "type": "ENROLLED_IN", "properties": {"grade": "A"}},
        {"from": "pr1","to": "c1", "type": "TEACHES",     "properties": {}},
        {"from": "pr1","to": "c3", "type": "TEACHES",     "properties": {}},
        {"from": "pr2","to": "c2", "type": "TEACHES",     "properties": {}},
        {"from": "pr1","to": "d1", "type": "BELONGS_TO",  "properties": {}},
        {"from": "pr2","to": "d1", "type": "BELONGS_TO",  "properties": {}},
        {"from": "s1", "to": "s2", "type": "FRIENDS_WITH","properties": {}},
        {"from": "s2", "to": "s3", "type": "FRIENDS_WITH","properties": {}},
        {"from": "s3", "to": "s4", "type": "FRIENDS_WITH","properties": {}},
    ]
    return nodes, edges


def get_empty_graph():
    """Empty knowledge graph template to build from scratch."""
    return [], []


TEMPLATE_GRAPHS = {
    "🎬 Movie Knowledge Graph": get_movie_graph,
    "👥 Social Network": get_social_network_graph,
    "🎓 Academic Network": get_academic_graph,
    "🆕 Empty Graph (Custom)": get_empty_graph,
}


# ======================================================================================
# 3. GRAPH HELPERS
# ======================================================================================

def get_node_display_name(node):
    """Get a human-readable display name for a node."""
    props = node["properties"]
    name = props.get("name") or props.get("title") or props.get("label") or node["id"]
    return f"{name} ({node['label']})"


def parse_properties(prop_string):
    """Parse 'key1=val1, key2=val2' into a dict. Auto-detects int/float values."""
    props = {}
    if not prop_string or not prop_string.strip():
        return props
    for pair in prop_string.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, val = pair.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Auto-detect numeric types
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass  # keep as string
            props[key] = val
    return props


def serialize_graph_for_ai(nodes, edges):
    """Serialize graph structure as a readable string for the AI prompt."""
    lines = []
    lines.append(f"=== GRAPH STRUCTURE ({len(nodes)} nodes, {len(edges)} edges) ===\n")

    lines.append("NODES:")
    for n in nodes:
        props_str = ", ".join(f'{k}: {json.dumps(v)}' for k, v in n["properties"].items())
        lines.append(f"  ({n['id']}:{n['label']} {{ {props_str} }})")

    lines.append("\nEDGES:")
    for i, e in enumerate(edges):
        props_str = ", ".join(f'{k}: {json.dumps(v)}' for k, v in e.get("properties", {}).items())
        props_display = f" {{ {props_str} }}" if props_str else ""
        lines.append(f"  [{i}] ({e['from']})-[:{e['type']}{props_display}]->({e['to']})")

    return "\n".join(lines)


# ======================================================================================
# 4. GRAPH VISUALIZATION (PYVIS)
# ======================================================================================

LABEL_PALETTE = [
    {"bg": "#4F46E5", "border": "#3730A3", "font": "#FFFFFF"},  # Indigo
    {"bg": "#DC2626", "border": "#991B1B", "font": "#FFFFFF"},  # Red
    {"bg": "#059669", "border": "#047857", "font": "#FFFFFF"},  # Emerald
    {"bg": "#D97706", "border": "#B45309", "font": "#FFFFFF"},  # Amber
    {"bg": "#7C3AED", "border": "#6D28D9", "font": "#FFFFFF"},  # Violet
    {"bg": "#DB2777", "border": "#BE185D", "font": "#FFFFFF"},  # Pink
    {"bg": "#0891B2", "border": "#0E7490", "font": "#FFFFFF"},  # Cyan
    {"bg": "#65A30D", "border": "#4D7C0F", "font": "#FFFFFF"},  # Lime
]

EDGE_PALETTE = ["#F59E0B", "#10B981", "#6366F1", "#EC4899", "#06B6D4", "#EF4444", "#8B5CF6", "#14B8A6"]


def get_label_color(label, all_labels):
    sorted_labels = sorted(set(all_labels))
    idx = sorted_labels.index(label) % len(LABEL_PALETTE)
    return LABEL_PALETTE[idx]


def get_edge_color(rel_type, all_types):
    sorted_types = sorted(set(all_types))
    idx = sorted_types.index(rel_type) % len(EDGE_PALETTE)
    return EDGE_PALETTE[idx]


def render_pyvis_graph(nodes, edges, highlight_nodes=None, highlight_edges=None, height="520px"):
    """Render an interactive graph using pyvis. Works with any graph structure."""
    if highlight_nodes is None:
        highlight_nodes = set()
    if highlight_edges is None:
        highlight_edges = set()

    if not nodes:
        return "<div style='color:white;text-align:center;padding:40px;background:#0F172A;'>No nodes to display. Add nodes in Graph Studio.</div>"

    all_labels = [n["label"] for n in nodes]
    all_rel_types = [e["type"] for e in edges]

    net = Network(height=height, width="100%", bgcolor="#0F172A", font_color="white",
                  directed=True, notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150, spring_strength=0.05)

    for n in nodes:
        colors = get_label_color(n["label"], all_labels)
        display = n["properties"].get("name") or n["properties"].get("title") or n["id"]
        props_str = "\n".join(f"  {k}: {v}" for k, v in n["properties"].items())
        tooltip = f"[{n['label']}] {display}\n{props_str}"

        is_hl = n["id"] in highlight_nodes
        bg = "#FACC15" if is_hl else colors["bg"]
        bd = "#EAB308" if is_hl else colors["border"]
        fc = "#000000" if is_hl else colors["font"]
        sz = 38 if is_hl else 30
        bw = 4 if is_hl else 2

        net.add_node(n["id"], label=display, title=tooltip, size=sz, shape="dot", borderWidth=bw,
                     color={"background": bg, "border": bd,
                            "highlight": {"background": "#FACC15", "border": "#EAB308"}},
                     font={"color": fc, "size": 13, "face": "Inter, Arial, sans-serif"})

    for i, e in enumerate(edges):
        is_hl = i in highlight_edges
        ec = "#FACC15" if is_hl else get_edge_color(e["type"], all_rel_types)
        ew = 4 if is_hl else 2
        props_str = ", ".join(f"{k}: {v}" for k, v in e.get("properties", {}).items())
        tooltip = f"{e['type']}" + (f" ({props_str})" if props_str else "")

        net.add_edge(e["from"], e["to"], title=tooltip, label=e["type"],
                     color={"color": ec, "highlight": "#FACC15"}, width=ew,
                     arrows="to",
                     font={"size": 9, "color": "#94A3B8", "strokeWidth": 0, "align": "middle"},
                     smooth={"type": "curvedCW", "roundness": 0.15})

    html = net.generate_html()
    html = html.replace("<body>", '<body style="margin:0;padding:0;background:#0F172A;">')
    return html


# ======================================================================================
# 5. IN-MEMORY QUERY ENGINE (WORKS ON ANY GRAPH)
# ======================================================================================

def op_find_all_nodes(nodes, edges, label=None):
    """Find all nodes, optionally filtered by label."""
    h_nodes = set()
    results = []
    for n in nodes:
        if label and n["label"] != label:
            continue
        row = {"id": n["id"], "label": n["label"]}
        row.update(n["properties"])
        results.append(row)
        h_nodes.add(n["id"])
    cypher = f"MATCH (n:{label})\nRETURN n" if label else "MATCH (n)\nRETURN n"
    return results, h_nodes, set(), cypher


def op_find_neighbors(nodes, edges, node_id, rel_type=None, direction="both"):
    """Find direct neighbors of a node."""
    node_map = {n["id"]: n for n in nodes}
    h_nodes = {node_id}
    h_edges = set()
    results = []
    for i, e in enumerate(edges):
        match_type = (rel_type is None or e["type"] == rel_type)
        neighbor_id = None
        if direction in ("out", "both") and e["from"] == node_id and match_type:
            neighbor_id = e["to"]
        if direction in ("in", "both") and e["to"] == node_id and match_type:
            neighbor_id = e["from"]
        if neighbor_id and neighbor_id in node_map:
            nb = node_map[neighbor_id]
            row = {"neighbor": neighbor_id, "label": nb["label"], "relationship": e["type"]}
            row.update(nb["properties"])
            results.append(row)
            h_nodes.add(neighbor_id)
            h_edges.add(i)

    src = node_map.get(node_id, {})
    src_name = src.get("properties", {}).get("name") or src.get("properties", {}).get("title") or node_id
    rel_clause = f":{rel_type}" if rel_type else ""
    if direction == "out":
        cypher = f"MATCH (a {{name: '{src_name}'}})-[r{rel_clause}]->(b)\nRETURN b, type(r)"
    elif direction == "in":
        cypher = f"MATCH (a {{name: '{src_name}'}})<-[r{rel_clause}]-(b)\nRETURN b, type(r)"
    else:
        cypher = f"MATCH (a {{name: '{src_name}'}})-[r{rel_clause}]-(b)\nRETURN b, type(r)"
    return results, h_nodes, h_edges, cypher


def op_multi_hop(nodes, edges, start_id, max_hops=3, rel_type=None):
    """BFS traversal from a start node up to max_hops."""
    node_map = {n["id"]: n for n in nodes}
    visited = {start_id: 0}
    queue = deque([(start_id, 0)])
    h_nodes = {start_id}
    h_edges = set()

    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for i, e in enumerate(edges):
            match_type = (rel_type is None or e["type"] == rel_type)
            neighbor = None
            if e["from"] == current and match_type:
                neighbor = e["to"]
            elif e["to"] == current and match_type:
                neighbor = e["from"]
            if neighbor and neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
                h_nodes.add(neighbor)
                h_edges.add(i)

    results = []
    for nid, hops in visited.items():
        if nid == start_id:
            continue
        n = node_map.get(nid, {})
        row = {"id": nid, "label": n.get("label", ""), "hops": hops}
        row.update(n.get("properties", {}))
        results.append(row)
    results.sort(key=lambda x: x["hops"])

    src = node_map.get(start_id, {})
    src_name = src.get("properties", {}).get("name") or src.get("properties", {}).get("title") or start_id
    rel_clause = f":{rel_type}" if rel_type else ""
    cypher = f"MATCH path = (a {{name: '{src_name}'}})-[{rel_clause}*1..{max_hops}]-(b)\nWHERE b <> a\nRETURN DISTINCT b, length(path) AS hops\nORDER BY hops"
    return results, h_nodes, h_edges, cypher


def op_shortest_path(nodes, edges, from_id, to_id):
    """Find shortest undirected path between two nodes via BFS."""
    node_map = {n["id"]: n for n in nodes}
    if from_id == to_id:
        return [{"note": "Start and end are the same node"}], {from_id}, set(), "-- same node --"

    parent = {from_id: (None, None)}
    queue = deque([from_id])

    while queue:
        current = queue.popleft()
        if current == to_id:
            break
        for i, e in enumerate(edges):
            neighbor = None
            if e["from"] == current:
                neighbor = e["to"]
            elif e["to"] == current:
                neighbor = e["from"]
            if neighbor and neighbor not in parent:
                parent[neighbor] = (current, i)
                queue.append(neighbor)

    if to_id not in parent:
        return [{"note": "No path found"}], set(), set(), "-- no path --"

    # Reconstruct
    path_nodes = []
    path_edges = set()
    cur = to_id
    while cur is not None:
        path_nodes.append(cur)
        prev, edge_idx = parent[cur]
        if edge_idx is not None:
            path_edges.add(edge_idx)
        cur = prev
    path_nodes.reverse()

    results = []
    for step, nid in enumerate(path_nodes):
        n = node_map.get(nid, {})
        row = {"step": step, "id": nid, "label": n.get("label", "")}
        row.update(n.get("properties", {}))
        results.append(row)

    h_nodes = set(path_nodes)
    src_name = node_map.get(from_id, {}).get("properties", {}).get("name") or from_id
    tgt_name = node_map.get(to_id, {}).get("properties", {}).get("name") or to_id
    cypher = f"MATCH path = shortestPath((a {{name: '{src_name}'}})-[*]-(b {{name: '{tgt_name}'}}))\nRETURN path, length(path) AS hops"
    return results, h_nodes, path_edges, cypher


def op_pattern_match(nodes, edges, src_label, rel_type, tgt_label):
    """Match pattern: (a:SrcLabel)-[:REL_TYPE]->(b:TgtLabel)."""
    node_map = {n["id"]: n for n in nodes}
    h_nodes = set()
    h_edges = set()
    results = []

    for i, e in enumerate(edges):
        if e["type"] != rel_type:
            continue
        src_node = node_map.get(e["from"])
        tgt_node = node_map.get(e["to"])
        if not src_node or not tgt_node:
            continue
        if (src_label == "Any" or src_node["label"] == src_label) and \
           (tgt_label == "Any" or tgt_node["label"] == tgt_label):
            src_name = src_node["properties"].get("name") or src_node["properties"].get("title") or e["from"]
            tgt_name = tgt_node["properties"].get("name") or tgt_node["properties"].get("title") or e["to"]
            row = {"source": src_name, "source_label": src_node["label"],
                   "relationship": rel_type,
                   "target": tgt_name, "target_label": tgt_node["label"]}
            row.update({f"rel_{k}": v for k, v in e.get("properties", {}).items()})
            results.append(row)
            h_nodes.add(e["from"])
            h_nodes.add(e["to"])
            h_edges.add(i)

    sl = f":{src_label}" if src_label != "Any" else ""
    tl = f":{tgt_label}" if tgt_label != "Any" else ""
    cypher = f"MATCH (a{sl})-[r:{rel_type}]->(b{tl})\nRETURN a, r, b"
    return results, h_nodes, h_edges, cypher


def op_aggregate_count(nodes, edges, rel_type, group_by="source"):
    """Count relationships per node (grouped by source or target)."""
    node_map = {n["id"]: n for n in nodes}
    counts = defaultdict(int)
    h_nodes = set()
    h_edges = set()

    for i, e in enumerate(edges):
        if rel_type != "Any" and e["type"] != rel_type:
            continue
        key = e["from"] if group_by == "source" else e["to"]
        counts[key] += 1
        h_nodes.add(e["from"])
        h_nodes.add(e["to"])
        h_edges.add(i)

    results = []
    for nid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        n = node_map.get(nid, {})
        name = n.get("properties", {}).get("name") or n.get("properties", {}).get("title") or nid
        results.append({"node": name, "label": n.get("label", ""), "count": cnt})

    rel_clause = f":{rel_type}" if rel_type != "Any" else ""
    if group_by == "source":
        cypher = f"MATCH (a)-[r{rel_clause}]->(b)\nRETURN a.name, COUNT(b) AS count\nORDER BY count DESC"
    else:
        cypher = f"MATCH (a)-[r{rel_clause}]->(b)\nRETURN b.name, COUNT(a) AS count\nORDER BY count DESC"
    return results, h_nodes, h_edges, cypher


def op_co_occurrence(nodes, edges, via_label, via_rel_type):
    """Find nodes that connect to the same intermediate node (co-occurrence / diamond pattern)."""
    node_map = {n["id"]: n for n in nodes}
    h_nodes = set()
    h_edges = set()
    results = []
    seen_pairs = set()

    # Build: intermediate_node_id -> list of source node ids
    intermediate_sources = defaultdict(list)
    intermediate_edge_map = defaultdict(list)
    for i, e in enumerate(edges):
        if e["type"] != via_rel_type:
            continue
        tgt = node_map.get(e["to"])
        if not tgt or (via_label != "Any" and tgt["label"] != via_label):
            continue
        intermediate_sources[e["to"]].append(e["from"])
        intermediate_edge_map[e["to"]].append(i)

    for mid, sources in intermediate_sources.items():
        for x in range(len(sources)):
            for y in range(x + 1, len(sources)):
                a, b = sources[x], sources[y]
                na = node_map.get(a, {}).get("properties", {}).get("name") or a
                nb = node_map.get(b, {}).get("properties", {}).get("name") or b
                if na > nb:
                    na, nb = nb, na
                    a, b = b, a
                pair_key = (na, nb, mid)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                mid_name = node_map.get(mid, {}).get("properties", {}).get("name") or \
                           node_map.get(mid, {}).get("properties", {}).get("title") or mid
                results.append({"node_1": na, "node_2": nb, "shared": mid_name,
                                "shared_label": node_map.get(mid, {}).get("label", "")})
                h_nodes.update([a, b, mid])
                for ei in intermediate_edge_map[mid]:
                    h_edges.add(ei)

    vl = f":{via_label}" if via_label != "Any" else ""
    cypher = f"MATCH (a)-[:{via_rel_type}]->(m{vl})<-[:{via_rel_type}]-(b)\nWHERE a <> b\nRETURN a.name, b.name, m.name"
    return results, h_nodes, h_edges, cypher


# ======================================================================================
# 6. GROQ AI INTEGRATION
# ======================================================================================

def query_groq_ai(api_key, graph_text, question):
    """Send a natural language question to Groq AI along with graph structure."""
    if not GROQ_AVAILABLE:
        return None, "Groq Python package not installed. Run: `pip install groq`"

    try:
        client = Groq(api_key=api_key)
        system_prompt = """You are a Cypher query language expert and graph database tutor.
You help students learn graph pattern matching by analyzing their knowledge graphs.

When the student asks a question about their graph:
1. Write the equivalent Cypher query
2. Execute the query mentally against the provided graph and give the ACTUAL results
3. Explain how the query works step by step

IMPORTANT: Always respond in this exact JSON format (no markdown, no extra text):
{
    "cypher_query": "The Cypher query that answers this question",
    "results": [{"column1": "value1", "column2": "value2"}],
    "explanation": "Step-by-step explanation of how the query traverses the graph",
    "highlighted_node_ids": ["id1", "id2"],
    "highlighted_edge_indices": [0, 1, 2],
    "query_type": "basic_match | traversal | multi_hop | aggregation | complex_pattern"
}

Rules:
- Use valid Cypher syntax
- Results MUST contain actual data from the provided graph, not made-up data
- highlighted_node_ids should be the node IDs involved in the answer
- highlighted_edge_indices should be 0-based indices of edges from the EDGES list
- Keep explanations educational and beginner-friendly"""

        user_prompt = f"""{graph_text}

STUDENT QUESTION: {question}

Respond ONLY with the JSON object, no other text."""

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()
        # Try to extract JSON from the response
        if content.startswith("```"):
            # Remove markdown code fences
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        result = json.loads(content)
        return result, None

    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text
        return {"cypher_query": "-- AI response was not valid JSON --",
                "results": [],
                "explanation": content,
                "highlighted_node_ids": [],
                "highlighted_edge_indices": [],
                "query_type": "unknown"}, None
    except Exception as e:
        return None, str(e)


# ======================================================================================
# 7. QUIZ QUESTIONS
# ======================================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What does the MATCH clause do in Cypher?",
        "options": [
            "A) Creates new nodes in the graph",
            "B) Specifies a graph pattern to search for in the database",
            "C) Deletes relationships from the graph",
            "D) Exports query results to a CSV file"
        ],
        "answer_index": 1,
        "explanation": "MATCH is the primary read clause in Cypher. It specifies a graph pattern and finds all subgraphs that match it."
    },
    {
        "id": 2,
        "question": "How is a relationship represented in Cypher syntax?",
        "options": [
            "A) Using curly braces: {r:TYPE}",
            "B) Using square brackets inside dashes: -[r:TYPE]->",
            "C) Using angle brackets: <r:TYPE>",
            "D) Using parentheses: (r:TYPE)"
        ],
        "answer_index": 1,
        "explanation": "Relationships use square brackets between dashes/arrows: -[r:TYPE]-> for directed edges."
    },
    {
        "id": 3,
        "question": "What does the variable-length path pattern [*1..3] mean?",
        "options": [
            "A) Match exactly 3 relationships",
            "B) Match paths containing between 1 and 3 relationships",
            "C) Match the first 3 nodes in the graph",
            "D) Skip the first relationship and match the next 3"
        ],
        "answer_index": 1,
        "explanation": "[*1..3] matches paths with 1 to 3 hops, enabling multi-hop traversal with bounded depth."
    },
    {
        "id": 4,
        "question": "In MATCH (a)-[:ACTED_IN]->(m)<-[:ACTED_IN]-(b), what pattern is matched?",
        "options": [
            "A) Two people who directed the same movie",
            "B) A person and the movie's director",
            "C) Two people (co-actors) who both acted in the same movie",
            "D) A person reviewing a movie they directed"
        ],
        "answer_index": 2,
        "explanation": "This 'diamond' pattern finds pairs of Person nodes that both have ACTED_IN relationships to the same Movie node — i.e., co-actors."
    },
    {
        "id": 5,
        "question": "What is the purpose of the WHERE clause in a Cypher query?",
        "options": [
            "A) To create new properties on existing nodes",
            "B) To filter matched results based on boolean conditions",
            "C) To specify the order of returned results",
            "D) To group results for aggregation"
        ],
        "answer_index": 1,
        "explanation": "WHERE filters after pattern matching. It evaluates boolean conditions on properties to narrow results."
    },
    {
        "id": 6,
        "question": "Which aggregation function counts how many movies each actor appeared in?",
        "options": [
            "A) SUM(m.title)",
            "B) AVG(m)",
            "C) COUNT(m)",
            "D) COLLECT(m.year)"
        ],
        "answer_index": 2,
        "explanation": "COUNT(m) counts matched Movie nodes per actor group."
    },
    {
        "id": 7,
        "question": "What advantage do graph databases have over relational databases for multi-hop queries?",
        "options": [
            "A) They use less disk space for all data types",
            "B) They provide constant-time traversal regardless of dataset size",
            "C) They automatically generate SQL queries",
            "D) They don't require any query language"
        ],
        "answer_index": 1,
        "explanation": "Graphs store relationships as pointers, enabling O(1) per-hop traversal vs. expensive JOINs."
    },
    {
        "id": 8,
        "question": "What does RETURN DISTINCT do in a Cypher query?",
        "options": [
            "A) Returns only the first result",
            "B) Removes duplicate rows from the result set",
            "C) Returns results in random order",
            "D) Converts all values to uppercase"
        ],
        "answer_index": 1,
        "explanation": "DISTINCT removes duplicate result rows, ensuring each unique combination appears only once."
    },
    {
        "id": 9,
        "question": "What is a 'hidden connection' in a knowledge graph?",
        "options": [
            "A) A deleted node",
            "B) An indirect relationship discoverable only through multi-hop traversal",
            "C) An encrypted property",
            "D) A relationship with no properties"
        ],
        "answer_index": 1,
        "explanation": "Hidden connections are indirect relationships discovered by traversing intermediate nodes and edges."
    },
    {
        "id": 10,
        "question": "What does COLLECT() do in Cypher?",
        "options": [
            "A) Counts matched patterns",
            "B) Aggregates values into a list/array",
            "C) Removes duplicate nodes",
            "D) Sorts results ascending"
        ],
        "answer_index": 1,
        "explanation": "COLLECT() gathers matched values into a list. E.g., COLLECT(m.title) returns all titles as an array per group."
    }
]


# ======================================================================================
# 8. PDF REPORT GENERATOR
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name, student_id, date_str,
                        trials_df, quiz_score, quiz_total, student_notes, nodes, edges):
    """Compile experiment records into a formatted PDF report."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Info box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")
    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score/quiz_total)*100 if quiz_total else 0)}%)", 1)
    pdf.ln(12)

    # Objectives
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Trials table
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Query Trials & Results", new_x="LMARGIN", new_y="NEXT")
    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No query trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        cols = list(trials_df.columns)
        col_w = max(18, int(190 / max(1, len(cols))))
        pdf.set_fill_color(37, 99, 235); pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        for c in cols:
            pdf.cell(col_w, 6, str(c)[:18], 1, 0, "C", True)
        pdf.ln()
        pdf.set_fill_color(248, 250, 252); pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        fill = False
        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, val_str[:18], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    # Observations
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The Cypher query experiments demonstrated effective graph pattern matching across varying "
        "traversal depths and aggregation operations, confirming graph-based retrieval efficiency."
    )
    notes_text = notes_text.encode('latin-1', 'replace').decode('latin-1')
    for line in textwrap.wrap(notes_text, width=100, break_long_words=True):
        pdf.set_x(18)
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # Graph Structure Details
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "4. Graph Structure Snapshot", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 6, f"Total Nodes: {len(nodes)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for n in nodes:
        props = str(n.get("properties", {}))
        txt = f"Node {n['id']} ({n.get('label', '')}): {props}"
        txt = txt.encode('latin-1', 'replace').decode('latin-1')
        for line in textwrap.wrap(txt, width=90, break_long_words=True):
            pdf.set_x(18)
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, f"Total Edges: {len(edges)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for e in edges:
        props = str(e.get("properties", {}))
        source = e.get('from', e.get('source', 'Unknown'))
        target = e.get('to', e.get('target', 'Unknown'))
        txt = f"Edge: {source} -[{e.get('type', '')}]-> {target} : {props}"
        txt = txt.encode('latin-1', 'replace').decode('latin-1')
        for line in textwrap.wrap(txt, width=90, break_long_words=True):
            pdf.set_x(18)
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Sign-off
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode('latin-1')
    return bytes(out)


# ======================================================================================
# 9. SECTION RENDERERS
# ======================================================================================

def render_theory_section():
    """Section 1: Theory, Background, Objectives, Procedure."""
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"])
    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")
    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")
    st.divider()
    with st.expander("Key Terminology & Variable Reference"):
        st.table(pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Variable", "Definition & Role"]
        ))


def render_graph_studio_tab():
    """Tab 1 of Simulation: Build / edit the graph + visualization."""
    nodes = st.session_state["graph_nodes"]
    edges = st.session_state["graph_edges"]

    # ---- Template loader ----
    st.subheader("📂 Load Template Graph")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        template = st.selectbox("Choose a template", list(TEMPLATE_GRAPHS.keys()),
                                index=0, key="template_select")
    with col_t2:
        st.write("")  # spacer
        st.write("")
        if st.button("Load Template", type="primary"):
            loader = TEMPLATE_GRAPHS[template]
            n, e = loader()
            st.session_state["graph_nodes"] = n
            st.session_state["graph_edges"] = e
            st.toast(f"✅ Loaded {template}")
            st.rerun()

    st.divider()

    # ---- Add Node ----
    st.subheader("➕ Add Node")
    with st.form("add_node_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            existing_labels = sorted(set(n["label"] for n in nodes)) if nodes else []
            node_label = st.text_input("Label (e.g., Person, Movie)", placeholder="Person")
            if existing_labels:
                st.caption(f"Existing: {', '.join(existing_labels)}")
        with c2:
            node_name = st.text_input("Name / Title", placeholder="Alice")
        with c3:
            node_props_str = st.text_input(
                "Additional Properties (key=value, …)",
                placeholder="age=30, city=Mumbai"
            )
        add_node_btn = st.form_submit_button("Add Node", type="primary")

    if add_node_btn and node_label and node_name:
        props = parse_properties(node_props_str)
        # Determine display key
        if node_label.lower() in ("movie", "film", "course", "post", "book"):
            props["title"] = node_name
        else:
            props["name"] = node_name
        # Generate unique ID
        node_id = node_name.lower().replace(" ", "_")
        existing_ids = {n["id"] for n in nodes}
        base_id = node_id
        counter = 1
        while node_id in existing_ids:
            node_id = f"{base_id}_{counter}"
            counter += 1
        nodes.append({"id": node_id, "label": node_label, "properties": props})
        st.session_state["graph_nodes"] = nodes
        st.toast(f"✅ Added node: {node_name} ({node_label})")
        st.rerun()

    # ---- Add Edge ----
    st.subheader("🔗 Add Relationship")
    if len(nodes) < 2:
        st.info("Add at least 2 nodes before creating relationships.")
    else:
        node_options = {get_node_display_name(n): n["id"] for n in nodes}
        with st.form("add_edge_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                edge_from_display = st.selectbox("From Node", list(node_options.keys()), key="ef")
            with c2:
                edge_type = st.text_input("Relationship Type", placeholder="ACTED_IN")
                existing_rels = sorted(set(e["type"] for e in edges)) if edges else []
                if existing_rels:
                    st.caption(f"Existing: {', '.join(existing_rels)}")
            with c3:
                edge_to_display = st.selectbox("To Node", list(node_options.keys()), key="et")
            edge_props_str = st.text_input("Relationship Properties (optional)", placeholder="role=Lead, since=2020")
            add_edge_btn = st.form_submit_button("Add Relationship", type="primary")

        if add_edge_btn and edge_type:
            props = parse_properties(edge_props_str)
            edges.append({
                "from": node_options[edge_from_display],
                "to": node_options[edge_to_display],
                "type": edge_type.upper().replace(" ", "_"),
                "properties": props
            })
            st.session_state["graph_edges"] = edges
            st.toast(f"✅ Added: {edge_from_display} -[{edge_type}]-> {edge_to_display}")
            st.rerun()

    st.divider()

    # ---- Current Graph Stats & Tables ----
    st.subheader("📊 Current Graph")
    if nodes:
        labels = set(n["label"] for n in nodes)
        rel_types = set(e["type"] for e in edges)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Nodes", len(nodes))
        mc2.metric("Edges", len(edges))
        mc3.metric("Labels", len(labels))
        mc4.metric("Rel Types", len(rel_types))

        # Node table with delete
        st.markdown("**Nodes:**")
        node_data = []
        for n in nodes:
            display = n["properties"].get("name") or n["properties"].get("title") or n["id"]
            props_short = ", ".join(f"{k}={v}" for k, v in n["properties"].items()
                                   if k not in ("name", "title"))
            node_data.append({"ID": n["id"], "Label": n["label"], "Name": display,
                              "Properties": props_short})
        st.dataframe(pd.DataFrame(node_data), hide_index=True, use_container_width=True)

        # Edge table
        if edges:
            st.markdown("**Relationships:**")
            edge_data = []
            node_map = {n["id"]: n for n in nodes}
            for e in edges:
                fn = node_map.get(e["from"], {}).get("properties", {})
                tn = node_map.get(e["to"], {}).get("properties", {})
                f_name = fn.get("name") or fn.get("title") or e["from"]
                t_name = tn.get("name") or tn.get("title") or e["to"]
                props_short = ", ".join(f"{k}={v}" for k, v in e.get("properties", {}).items())
                edge_data.append({"From": f_name, "Type": e["type"], "To": t_name,
                                  "Properties": props_short})
            st.dataframe(pd.DataFrame(edge_data), hide_index=True, use_container_width=True)

        # Delete controls
        with st.expander("🗑️ Delete Nodes / Edges"):
            dc1, dc2 = st.columns(2)
            with dc1:
                del_node_id = st.selectbox("Delete Node",
                    [f"{get_node_display_name(n)} [{n['id']}]" for n in nodes], key="del_n")
                if st.button("Delete Node", key="del_node_btn"):
                    target_id = del_node_id.split("[")[-1].rstrip("]")
                    st.session_state["graph_nodes"] = [n for n in nodes if n["id"] != target_id]
                    st.session_state["graph_edges"] = [e for e in edges
                                                       if e["from"] != target_id and e["to"] != target_id]
                    st.toast(f"🗑️ Deleted node {target_id} and its edges")
                    st.rerun()
            with dc2:
                if edges:
                    edge_display_options = []
                    for i, e in enumerate(edges):
                        f_name = node_map.get(e["from"], {}).get("properties", {}).get("name") or e["from"]
                        t_name = node_map.get(e["to"], {}).get("properties", {}).get("name") or e["to"]
                        edge_display_options.append(f"[{i}] {f_name} -[{e['type']}]-> {t_name}")
                    del_edge_idx = st.selectbox("Delete Edge", edge_display_options, key="del_e")
                    if st.button("Delete Edge", key="del_edge_btn"):
                        idx = int(del_edge_idx.split("]")[0].lstrip("["))
                        st.session_state["graph_edges"] = [e for j, e in enumerate(edges) if j != idx]
                        st.toast("🗑️ Edge deleted")
                        st.rerun()

    else:
        st.info("Your graph is empty. Load a template or add nodes/edges to view the data tables.")

    st.divider()
    st.subheader("💾 Data Management & Visualization")

    # Export / Import
    with st.expander("📤 Export / Import Graph (JSON)"):
        ec1, ec2 = st.columns(2)
        with ec1:
            graph_json = json.dumps({"nodes": nodes, "edges": edges}, indent=2)
            st.download_button("Download Graph JSON", data=graph_json,
                               file_name="graph_export.json", mime="application/json")
        with ec2:
            uploaded = st.file_uploader("Import Graph JSON", type=["json"], key="import_graph")
            if uploaded:
                try:
                    data = json.load(uploaded)
                    st.session_state["graph_nodes"] = data["nodes"]
                    st.session_state["graph_edges"] = data["edges"]
                    st.toast("✅ Graph imported!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Invalid JSON: {ex}")

    # Import CSV
    # Import CSV
    with st.expander("📥 Import Graph from CSV (Edge List)"):
        st.caption("Upload a CSV with at least 3 columns: `Source, Relationship, Target` to append to the graph.")
        with st.form("csv_import_form", clear_on_submit=True):
            csv_file = st.file_uploader("Upload CSV", type=["csv"])
            submitted = st.form_submit_button("Import CSV Data")
            
            if submitted and csv_file:
                try:
                    df_csv = pd.read_csv(csv_file)
                    if len(df_csv.columns) >= 3:
                        new_nodes = {}
                        new_edges = []
                        existing_names = {str(n["properties"].get("name", n["id"])).lower(): n["id"] for n in nodes}
                        
                        def get_or_create_node(name_val):
                            name_str = str(name_val).strip()
                            name_lower = name_str.lower()
                            if name_lower in existing_names:
                                return existing_names[name_lower]
                            if name_lower in new_nodes:
                                return new_nodes[name_lower]["id"]
                            
                            node_id = name_lower.replace(" ", "_")
                            all_ids = {n["id"] for n in nodes} | {n["id"] for n in new_nodes.values()}
                            base_id = node_id
                            c = 1
                            while node_id in all_ids:
                                node_id = f"{base_id}_{c}"
                                c += 1
                                
                            new_nodes[name_lower] = {
                                "id": node_id,
                                "label": "Entity",
                                "properties": {"name": name_str}
                            }
                            return node_id

                        for _, row in df_csv.iterrows():
                            src_val = row.iloc[0]
                            rel_val = row.iloc[1]
                            tgt_val = row.iloc[2]
                            
                            if pd.isna(src_val) or pd.isna(rel_val) or pd.isna(tgt_val):
                                continue
                                
                            src_id = get_or_create_node(src_val)
                            tgt_id = get_or_create_node(tgt_val)
                            
                            new_edges.append({
                                "from": src_id,
                                "to": tgt_id,
                                "type": str(rel_val).strip().upper().replace(" ", "_"),
                                "properties": {}
                            })
                            
                        if new_nodes or new_edges:
                            st.session_state["graph_nodes"].extend(list(new_nodes.values()))
                            st.session_state["graph_edges"].extend(new_edges)
                            st.toast(f"✅ Imported {len(new_nodes)} new nodes & {len(new_edges)} edges!")
                            st.rerun()
                    else:
                        st.error("CSV must have at least 3 columns (Source, Relationship, Target)")
                except Exception as ex:
                    st.error(f"Error parsing CSV: {ex}")

    # Visualization
    if nodes:
        st.divider()
        st.subheader("🌐 Graph Visualization")
        graph_html = render_pyvis_graph(nodes, edges)
        components.html(graph_html, height=550, scrolling=False)


def render_query_lab_tab():
    """Tab 2 of Simulation: Run predefined/custom operations on the current graph."""
    nodes = st.session_state["graph_nodes"]
    edges = st.session_state["graph_edges"]

    if not nodes:
        st.warning("⚠️ Graph is empty. Go to **Graph Studio** tab to add nodes first.")
        return

    st.subheader("🔍 Query Operation")
    node_map = {n["id"]: n for n in nodes}
    all_labels = sorted(set(n["label"] for n in nodes))
    all_rel_types = sorted(set(e["type"] for e in edges))

    operation = st.selectbox("Select Operation", [
        "Find All Nodes (by label)",
        "Find Neighbors (of a node)",
        "Multi-hop Traversal (BFS)",
        "Shortest Path (between two nodes)",
        "Pattern Match (label → rel → label)",
        "Aggregation (count relationships)",
        "Co-occurrence (shared connections)"
    ], key="op_select")

    results, h_nodes, h_edges, cypher = [], set(), set(), ""

    # ---- Operation Parameters ----
    if operation == "Find All Nodes (by label)":
        label = st.selectbox("Filter by Label", ["All"] + all_labels, key="fa_label")
        if st.button("▶ Run Query", type="primary", key="run_fa"):
            lbl = None if label == "All" else label
            results, h_nodes, h_edges, cypher = op_find_all_nodes(nodes, edges, lbl)

    elif operation == "Find Neighbors (of a node)":
        node_opts = {get_node_display_name(n): n["id"] for n in nodes}
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_node = st.selectbox("Select Node", list(node_opts.keys()), key="fn_node")
        with c2:
            rel_filter = st.selectbox("Relationship Type", ["Any"] + all_rel_types, key="fn_rel")
        with c3:
            direction = st.selectbox("Direction", ["both", "out", "in"], key="fn_dir")
        if st.button("▶ Run Query", type="primary", key="run_fn"):
            rt = None if rel_filter == "Any" else rel_filter
            results, h_nodes, h_edges, cypher = op_find_neighbors(
                nodes, edges, node_opts[sel_node], rt, direction)

    elif operation == "Multi-hop Traversal (BFS)":
        node_opts = {get_node_display_name(n): n["id"] for n in nodes}
        c1, c2, c3 = st.columns(3)
        with c1:
            start_node = st.selectbox("Start Node", list(node_opts.keys()), key="mh_start")
        with c2:
            max_hops = st.slider("Max Hops", 1, 5, 3, key="mh_hops")
        with c3:
            rel_filter = st.selectbox("Relationship Type", ["Any"] + all_rel_types, key="mh_rel")
        if st.button("▶ Run Query", type="primary", key="run_mh"):
            rt = None if rel_filter == "Any" else rel_filter
            results, h_nodes, h_edges, cypher = op_multi_hop(
                nodes, edges, node_opts[start_node], max_hops, rt)

    elif operation == "Shortest Path (between two nodes)":
        node_opts = {get_node_display_name(n): n["id"] for n in nodes}
        c1, c2 = st.columns(2)
        with c1:
            from_node = st.selectbox("From Node", list(node_opts.keys()), key="sp_from")
        with c2:
            to_node = st.selectbox("To Node", list(node_opts.keys()), key="sp_to")
        if st.button("▶ Run Query", type="primary", key="run_sp"):
            results, h_nodes, h_edges, cypher = op_shortest_path(
                nodes, edges, node_opts[from_node], node_opts[to_node])

    elif operation == "Pattern Match (label → rel → label)":
        c1, c2, c3 = st.columns(3)
        with c1:
            src_label = st.selectbox("Source Label", ["Any"] + all_labels, key="pm_src")
        with c2:
            rel_type = st.selectbox("Relationship Type", all_rel_types, key="pm_rel") if all_rel_types else None
        with c3:
            tgt_label = st.selectbox("Target Label", ["Any"] + all_labels, key="pm_tgt")
        if rel_type and st.button("▶ Run Query", type="primary", key="run_pm"):
            results, h_nodes, h_edges, cypher = op_pattern_match(
                nodes, edges, src_label, rel_type, tgt_label)

    elif operation == "Aggregation (count relationships)":
        c1, c2 = st.columns(2)
        with c1:
            rel_type = st.selectbox("Relationship Type", ["Any"] + all_rel_types, key="ag_rel")
        with c2:
            group_by = st.selectbox("Group By", ["source", "target"], key="ag_grp")
        if st.button("▶ Run Query", type="primary", key="run_ag"):
            results, h_nodes, h_edges, cypher = op_aggregate_count(
                nodes, edges, rel_type, group_by)

    elif operation == "Co-occurrence (shared connections)":
        c1, c2 = st.columns(2)
        with c1:
            via_rel = st.selectbox("Via Relationship", all_rel_types, key="co_rel") if all_rel_types else None
        with c2:
            via_label = st.selectbox("Shared Node Label", ["Any"] + all_labels, key="co_lbl")
        if via_rel and st.button("▶ Run Query", type="primary", key="run_co"):
            results, h_nodes, h_edges, cypher = op_co_occurrence(
                nodes, edges, via_label, via_rel)

    # ---- Display Results ----
    if results or h_nodes:
        st.divider()

        # Cypher equivalent
        st.markdown("**Equivalent Cypher Query:**")
        st.code(cypher, language="cypher")

        # Metrics
        m1, m2, m3 = st.columns(3)
        results_df = pd.DataFrame(results) if results else pd.DataFrame()
        m1.metric("Rows Returned", len(results_df))
        m2.metric("Nodes Matched", len(h_nodes))
        m3.metric("Edges Traversed", len(h_edges))

        # Results table
        if not results_df.empty:
            st.subheader("Query Results")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Highlighted graph
        st.subheader("Matched Pattern (Highlighted)")
        graph_html = render_pyvis_graph(nodes, edges, h_nodes, h_edges)
        components.html(graph_html, height=550, scrolling=False)

        # Store for trial logging
        st.session_state["current_sim_result"] = {
            "operation": operation,
            "cypher": cypher.replace("\n", " "),
            "rows_returned": len(results_df),
            "nodes_matched": len(h_nodes),
            "edges_traversed": len(h_edges),
        }

    # ---- Trial Logger ----
    st.divider()
    st.subheader("📓 Experimental Data Log Book")
    lc1, lc2 = st.columns([1.5, 3.5])
    with lc1:
        st.caption("Record the current query trial:")
        if st.button("Record Current Trial", type="primary", key="record_trial"):
            sim = st.session_state.get("current_sim_result")
            if sim:
                trial = {
                    "Trial #": len(st.session_state["trials"]) + 1,
                    "Operation": sim["operation"],
                    "Rows": sim["rows_returned"],
                    "Nodes": sim["nodes_matched"],
                    "Edges": sim["edges_traversed"],
                    "Time": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["trials"].append(trial)
                st.toast(f"✅ Trial #{trial['Trial #']} recorded!")
            else:
                st.toast("⚠️ Run a query first.")
        if st.button("Clear Trials", key="clear_trials"):
            st.session_state["trials"] = []
            st.toast("🗑️ Cleared.")
    with lc2:
        if st.session_state["trials"]:
            df = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", data=df.to_csv(index=False).encode(),
                               file_name="cypher_trials.csv", mime="text/csv")
        else:
            st.info("No trials recorded yet.")


def render_ai_assistant_tab():
    """Tab 3 of Simulation: Groq AI natural language queries."""
    nodes = st.session_state["graph_nodes"]
    edges = st.session_state["graph_edges"]
    api_key = st.session_state.get("groq_api_key", "")

    st.subheader("🤖 AI-Powered Graph Query Assistant")
    st.markdown(
        "Ask questions about your graph in **plain English**. The AI will generate the "
        "equivalent **Cypher query**, execute it against your graph, and explain the results."
    )

    if not GROQ_AVAILABLE:
        st.error("❌ The `groq` Python package is not installed. Run: `pip install groq`")
        return

    if not api_key:
        st.warning("⚠️ Enter your **Groq API Key** in the sidebar to use the AI Assistant. "
                    "Get a free key at [console.groq.com](https://console.groq.com)")
        return

    if not nodes:
        st.warning("⚠️ Graph is empty. Go to **Graph Studio** to build a graph first.")
        return

    # Suggested questions based on graph
    labels = set(n["label"] for n in nodes)
    rel_types = set(e["type"] for e in edges)
    sample_node = nodes[0]["properties"].get("name") or nodes[0]["properties"].get("title") or nodes[0]["id"]

    suggestions = [
        f"What are all the {list(labels)[0]} nodes?",
        f"Who is connected to {sample_node}?",
    ]
    if len(rel_types) > 0:
        suggestions.append(f"Find all {list(rel_types)[0]} relationships")
    if len(nodes) >= 3:
        n1 = nodes[0]["properties"].get("name") or nodes[0]["id"]
        n2 = nodes[-1]["properties"].get("name") or nodes[-1]["id"]
        suggestions.append(f"Is there a path between {n1} and {n2}?")
    suggestions.append("What hidden connections exist in this graph?")

    st.markdown("**💡 Suggested questions:**")
    suggestion_cols = st.columns(min(len(suggestions), 3))
    for i, s in enumerate(suggestions[:3]):
        with suggestion_cols[i]:
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state["ai_question"] = s

    question = st.text_input(
        "Ask a question about your graph:",
        value=st.session_state.get("ai_question", ""),
        placeholder="e.g., Find all friends of Alice who acted in a movie",
        key="ai_q_input"
    )

    if st.button("🚀 Ask AI", type="primary", disabled=not question):
        with st.spinner("🧠 AI is analyzing your graph..."):
            graph_text = serialize_graph_for_ai(nodes, edges)
            result, error = query_groq_ai(api_key, graph_text, question)

        if error:
            st.error(f"❌ AI Error: {error}")
        elif result:
            st.divider()

            # Cypher query
            st.markdown("**📝 Generated Cypher Query:**")
            st.code(result.get("cypher_query", "N/A"), language="cypher")

            # Query type badge
            qt = result.get("query_type", "unknown")
            st.caption(f"Query Type: `{qt}`")

            # Results table
            ai_results = result.get("results", [])
            if ai_results and isinstance(ai_results, list) and isinstance(ai_results[0], dict):
                st.subheader("📊 Results")
                st.dataframe(pd.DataFrame(ai_results), use_container_width=True, hide_index=True)
            elif ai_results:
                st.subheader("📊 Results")
                st.write(ai_results)

            # Explanation
            explanation = result.get("explanation", "")
            if explanation:
                st.subheader("📖 Explanation")
                st.info(explanation)

            # Highlighted graph
            h_nodes = set(result.get("highlighted_node_ids", []))
            h_edges_raw = result.get("highlighted_edge_indices", [])
            h_edges = set()
            for idx in h_edges_raw:
                if isinstance(idx, int) and 0 <= idx < len(edges):
                    h_edges.add(idx)

            if h_nodes or h_edges:
                st.subheader("🌐 Matched Subgraph")
                graph_html = render_pyvis_graph(nodes, edges, h_nodes, h_edges)
                components.html(graph_html, height=500, scrolling=False)

            # Store for trials
            st.session_state["current_sim_result"] = {
                "operation": f"AI: {question[:40]}...",
                "cypher": result.get("cypher_query", "N/A")[:60],
                "rows_returned": len(ai_results) if isinstance(ai_results, list) else 0,
                "nodes_matched": len(h_nodes),
                "edges_traversed": len(h_edges),
            }

    # Show history
    if "ai_history" not in st.session_state:
        st.session_state["ai_history"] = []


def render_simulation_section():
    """Section 2: Interactive Simulation with 3 tabs."""
    st.header("Interactive Cypher Query Simulation")

    tab1, tab2, tab3 = st.tabs(["📊 Graph Studio", "🔍 Cypher Query Lab", "🤖 AI Assistant"])

    with tab1:
        render_graph_studio_tab()
    with tab2:
        render_query_lab_tab()
    with tab3:
        render_ai_assistant_tab()


def render_quiz_section():
    """Section 3: Quiz with self-grading."""
    st.header("Concept Assessment Quiz")
    st.write("Test your understanding of Cypher queries and graph pattern matching.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Q{q['id']}:", options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_{q['id']}", label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)
        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True
        st.divider()
        st.subheader("Results & Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct = q["answer_index"]
            if user_ans == correct:
                score += 1
                st.success(f"**Q{q['id']}: Correct!** _{q['explanation']}_")
            else:
                st.error(f"**Q{q['id']}: Incorrect.** Your answer: {q['options'][user_ans]}\n\n"
                         f"**Correct:** {q['options'][correct]}\n\n**Reason:** _{q['explanation']}_")
        st.session_state["quiz_score"] = score
        st.info(f"Score: **{score} / {len(QUIZ_QUESTIONS)}** ({score*100//len(QUIZ_QUESTIONS)}%)")
    elif st.session_state.get("quiz_submitted"):
        st.success(f"Quiz submitted. Score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Section 4: Report generation with PDF export."""
    st.header("Report Generation")
    st.write("Compile your details, query trials, and quiz score into a downloadable PDF report.")

    c1, c2, c3 = st.columns(3)
    with c1:
        student_name = st.text_input("Student Name",
                                     value=st.session_state["student_info"].get("name", ""))
    with c2:
        student_id = st.text_input("Student Roll / ID",
                                   value=st.session_state["student_info"].get("id", ""))
    with c3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your observations and conclusions:",
        value=st.session_state.get("student_notes", (
            "The Cypher query experiments demonstrated effective graph pattern matching across varying "
            "traversal depths and aggregation operations, confirming graph-based retrieval efficiency "
            "for discovering hidden connections."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Graph:** {len(st.session_state['graph_nodes'])} nodes, {len(st.session_state['graph_edges'])} edges")
    st.write(f"**Quiz:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("No trials recorded yet.")

    pdf_bytes = generate_pdf_report(
        student_name, student_id, str(lab_date), trials_df,
        st.session_state.get("quiz_score", 0), len(QUIZ_QUESTIONS), student_notes,
        st.session_state["graph_nodes"], st.session_state["graph_edges"]
    )

    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Lab Report")
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Open PDF", url="/app/static/lab_report.pdf",
                        type="primary", use_container_width=True)
    with c2:
        st.download_button("Download lab_report.pdf", data=pdf_bytes,
                           file_name="lab_report.pdf", mime="application/pdf",
                           key="dl_pdf", use_container_width=True)


# ======================================================================================
# 10. MAIN ENTRYPOINT
# ======================================================================================

def init_session_state():
    """Initialize all session state variables."""
    n_movie, e_movie = get_movie_graph()
    defaults = {
        "graph_nodes": n_movie,
        "graph_edges": e_movie,
        "trials": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_score": 0,
        "student_info": {"name": "", "id": "", "date": str(datetime.now().date())},
        "student_notes": "",
        "current_sim_result": None,
        "groq_api_key": "",
        "ai_question": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    st.set_page_config(
        page_title="Advanced Cypher Queries & Graph Pattern Matching — Virtual Lab",
        page_icon="🔗",
        layout="wide"
    )
    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])

    # Sidebar Navigation
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    # Groq API Key in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("🤖 AI Configuration")
    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=st.session_state.get("groq_api_key", ""),
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )
    st.session_state["groq_api_key"] = api_key

    # Progress Tracker
    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    st.sidebar.write(f"- **Graph:** {len(st.session_state['graph_nodes'])} nodes, "
                     f"{len(st.session_state['graph_edges'])} edges")
    st.sidebar.write(f"- **Trials Recorded:** {len(st.session_state['trials'])}")
    quiz_status = "✅ Done" if st.session_state.get("quiz_submitted") else "⏳ Pending"
    st.sidebar.write(f"- **Quiz:** {quiz_status}")
    if st.session_state.get("quiz_submitted"):
        st.sidebar.write(f"- **Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")

    st.sidebar.divider()
    st.sidebar.caption("Virtual Lab — IIT Kharagpur Format")
    st.sidebar.caption("Dynamic graph builder + Groq AI assistant")

    # Section Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
