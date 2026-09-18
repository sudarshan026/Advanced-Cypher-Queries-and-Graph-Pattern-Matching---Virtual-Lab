"""
Virtual Laboratory Experiment: Advanced Cypher Queries and Graph Pattern Matching
A Streamlit-based Virtual Lab following IIT Kharagpur's Virtual Lab format.
Sections:
  1. Theory: Concepts, objectives, procedure, and terminology.
  2. Simulation: Interactive graph visualization with Cypher query execution.
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded trials, observations, and downloadable PDF report.

Graph experiments are designed using frontend visualization (pyvis/streamlit) with fewer nodes.
Neo4j implementation is NOT used — all queries run against an in-memory graph.
"""

import os
import re
import json
import textwrap
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from pyvis.network import Network
import streamlit.components.v1 as components


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Advanced Cypher Queries and Graph Pattern Matching",
    "objectives": [
        "Understand Cypher query language syntax for nodes, relationships, and properties.",
        "Perform multi-hop graph traversals to discover indirect connections.",
        "Apply filtering (WHERE), aggregation (COUNT, COLLECT), and ordering on graph data.",
        "Execute complex graph pattern matching queries with variable-length paths.",
        "Efficiently retrieve complex relationships and hidden connections in a knowledge graph."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles

**Cypher** is a declarative graph query language originally developed for **Neo4j** and now 
standardized as **GQL (Graph Query Language)** by ISO. It is designed to be intuitive, using 
ASCII-art style syntax to represent graph patterns — making it easy to express complex 
relationships in a readable form.

#### Why Graph Pattern Matching?

Traditional relational databases use JOINs to connect tables, which becomes increasingly 
expensive as the number of hops grows. Graph databases store relationships as first-class 
citizens, enabling **constant-time traversal** regardless of dataset size. This makes them 
ideal for:

- **Social network analysis** — finding friends-of-friends, influence propagation
- **Recommendation engines** — "users who liked X also liked Y"
- **Fraud detection** — discovering hidden rings of suspicious transactions
- **Knowledge graphs** — traversing ontologies and semantic relationships

#### Core Cypher Concepts

| Concept | Syntax | Description |
|---------|--------|-------------|
| **Node** | `(n:Label {prop: value})` | Represents an entity (person, movie, etc.) |
| **Relationship** | `-[r:TYPE {prop: value}]->` | Directed edge connecting two nodes |
| **Pattern Matching** | `MATCH (a)-[r]->(b)` | Find subgraphs matching the pattern |
| **Filtering** | `WHERE a.name = 'Alice'` | Filter results by property conditions |
| **Return** | `RETURN a.name, r.type` | Specify which data to output |
| **Aggregation** | `COUNT(*)`, `COLLECT()`, `AVG()` | Aggregate results across matches |
| **Variable-Length Paths** | `(a)-[*1..3]->(b)` | Match paths of 1 to 3 hops |
| **Optional Match** | `OPTIONAL MATCH` | Left-outer-join style matching |

### Workflow & System Overview

1. **Graph Construction**: Build a small knowledge graph with labeled nodes and typed relationships.
2. **Query Formulation**: Write Cypher-like queries to match patterns in the graph.
3. **Traversal & Filtering**: Execute multi-hop traversals with property filters.
4. **Aggregation & Analysis**: Use COUNT, COLLECT, and other aggregations to summarize results.
5. **Pattern Discovery**: Identify hidden connections through complex pattern matching.
    """,

    "procedure": [
        "Step 1: Review the theoretical background on Cypher syntax, graph patterns, and traversal algorithms.",
        "Step 2: Navigate to the **Simulation** section using the sidebar.",
        "Step 3: Explore the **interactive knowledge graph** to understand its structure (nodes, edges, properties).",
        "Step 4: Select from **predefined Cypher queries** or write your own to query the graph.",
        "Step 5: Run the query and observe the **results table** and **highlighted subgraph**.",
        "Step 6: Experiment with different query types: simple match, multi-hop, filtering, aggregation.",
        "Step 7: Record at least 3–4 distinct query trials using the **Record Trial** button.",
        "Step 8: Complete the **Quiz** to assess your understanding of Cypher concepts.",
        "Step 9: Generate and download your **Lab Report** from the Report Generation section."
    ],

    "key_terms": {
        "Node (Vertex)": "A fundamental entity in a graph, representing a real-world object like a Person, Movie, or City.",
        "Relationship (Edge)": "A directed connection between two nodes, representing how entities are related (e.g., ACTED_IN, FRIENDS_WITH).",
        "Label": "A tag assigned to a node to categorize it (e.g., :Person, :Movie). A node can have multiple labels.",
        "Property": "A key-value pair stored on a node or relationship (e.g., name: 'Alice', year: 2020).",
        "MATCH Clause": "The primary read clause in Cypher; specifies a graph pattern to search for.",
        "WHERE Clause": "Filters the matched results based on boolean conditions on properties.",
        "RETURN Clause": "Specifies the output columns/values from a query.",
        "Multi-hop Traversal": "Following a chain of relationships across multiple nodes (e.g., friend-of-friend).",
        "Variable-Length Path": "A path pattern like [*1..3] that matches paths between 1 and 3 hops long.",
        "Aggregation": "Functions like COUNT(), SUM(), COLLECT(), AVG() that summarize data across multiple matches."
    }
}


# ======================================================================================
# 2. IN-MEMORY GRAPH DATABASE & CYPHER ENGINE
# ======================================================================================

def build_sample_graph():
    """
    Build a small movie knowledge graph suitable for demonstrating Cypher queries.
    Returns nodes (list of dicts) and edges (list of dicts).
    """
    nodes = [
        # People
        {"id": "alice",   "label": "Person", "properties": {"name": "Alice",   "age": 32, "city": "Mumbai"}},
        {"id": "bob",     "label": "Person", "properties": {"name": "Bob",     "age": 28, "city": "Delhi"}},
        {"id": "carol",   "label": "Person", "properties": {"name": "Carol",   "age": 35, "city": "Mumbai"}},
        {"id": "dave",    "label": "Person", "properties": {"name": "Dave",    "age": 41, "city": "Bangalore"}},
        {"id": "eve",     "label": "Person", "properties": {"name": "Eve",     "age": 26, "city": "Chennai"}},
        {"id": "frank",   "label": "Person", "properties": {"name": "Frank",   "age": 38, "city": "Delhi"}},
        # Movies
        {"id": "m1", "label": "Movie", "properties": {"title": "GraphWorld",    "year": 2021, "genre": "Sci-Fi"}},
        {"id": "m2", "label": "Movie", "properties": {"title": "Query Quest",   "year": 2019, "genre": "Adventure"}},
        {"id": "m3", "label": "Movie", "properties": {"title": "Node Noir",     "year": 2022, "genre": "Thriller"}},
        {"id": "m4", "label": "Movie", "properties": {"title": "Edge of Logic",  "year": 2020, "genre": "Drama"}},
        # Directors
        {"id": "dir1", "label": "Director", "properties": {"name": "Raj Kumar",  "awards": 3}},
        {"id": "dir2", "label": "Director", "properties": {"name": "Meera Nair", "awards": 5}},
    ]

    edges = [
        # ACTED_IN relationships
        {"from": "alice", "to": "m1", "type": "ACTED_IN", "properties": {"role": "Lead"}},
        {"from": "alice", "to": "m2", "properties": {"role": "Supporting"}, "type": "ACTED_IN"},
        {"from": "bob",   "to": "m1", "type": "ACTED_IN", "properties": {"role": "Supporting"}},
        {"from": "bob",   "to": "m3", "type": "ACTED_IN", "properties": {"role": "Lead"}},
        {"from": "carol", "to": "m2", "type": "ACTED_IN", "properties": {"role": "Lead"}},
        {"from": "carol", "to": "m4", "type": "ACTED_IN", "properties": {"role": "Lead"}},
        {"from": "dave",  "to": "m3", "type": "ACTED_IN", "properties": {"role": "Supporting"}},
        {"from": "dave",  "to": "m4", "type": "ACTED_IN", "properties": {"role": "Lead"}},
        {"from": "eve",   "to": "m1", "type": "ACTED_IN", "properties": {"role": "Cameo"}},
        {"from": "frank", "to": "m2", "type": "ACTED_IN", "properties": {"role": "Lead"}},
        {"from": "frank", "to": "m3", "type": "ACTED_IN", "properties": {"role": "Supporting"}},
        # DIRECTED relationships
        {"from": "dir1", "to": "m1", "type": "DIRECTED", "properties": {}},
        {"from": "dir1", "to": "m3", "type": "DIRECTED", "properties": {}},
        {"from": "dir2", "to": "m2", "type": "DIRECTED", "properties": {}},
        {"from": "dir2", "to": "m4", "type": "DIRECTED", "properties": {}},
        # FRIENDS_WITH relationships (bidirectional conceptually, stored as directed)
        {"from": "alice", "to": "bob",   "type": "FRIENDS_WITH", "properties": {"since": 2018}},
        {"from": "bob",   "to": "carol", "type": "FRIENDS_WITH", "properties": {"since": 2019}},
        {"from": "carol", "to": "dave",  "type": "FRIENDS_WITH", "properties": {"since": 2017}},
        {"from": "dave",  "to": "eve",   "type": "FRIENDS_WITH", "properties": {"since": 2020}},
        {"from": "alice", "to": "frank", "type": "FRIENDS_WITH", "properties": {"since": 2021}},
        {"from": "eve",   "to": "frank", "type": "FRIENDS_WITH", "properties": {"since": 2022}},
        # REVIEWED relationships
        {"from": "alice", "to": "m3", "type": "REVIEWED", "properties": {"rating": 4.5}},
        {"from": "bob",   "to": "m2", "type": "REVIEWED", "properties": {"rating": 3.8}},
        {"from": "carol", "to": "m1", "type": "REVIEWED", "properties": {"rating": 4.9}},
        {"from": "eve",   "to": "m4", "type": "REVIEWED", "properties": {"rating": 4.2}},
        {"from": "frank", "to": "m1", "type": "REVIEWED", "properties": {"rating": 4.0}},
    ]

    return nodes, edges


# Node ID -> node dict lookup
def _build_index(nodes, edges):
    node_map = {n["id"]: n for n in nodes}
    # adjacency: node_id -> list of (edge, neighbor_id)
    adj_out = defaultdict(list)  # outgoing
    adj_in  = defaultdict(list)  # incoming
    adj_any = defaultdict(list)  # both directions
    for e in edges:
        adj_out[e["from"]].append((e, e["to"]))
        adj_in[e["to"]].append((e, e["from"]))
        adj_any[e["from"]].append((e, e["to"]))
        adj_any[e["to"]].append((e, e["from"]))
    return node_map, adj_out, adj_in, adj_any


# ------------- Predefined query library ------------------------------------------

PREDEFINED_QUERIES = [
    {
        "category": "Basic Pattern Match",
        "name": "Find all Person nodes",
        "cypher": "MATCH (p:Person)\nRETURN p.name, p.age, p.city",
        "description": "Retrieve all nodes labeled 'Person' with their properties.",
    },
    {
        "category": "Basic Pattern Match",
        "name": "Find all Movies released after 2020",
        "cypher": "MATCH (m:Movie)\nWHERE m.year > 2020\nRETURN m.title, m.year, m.genre",
        "description": "Filter Movie nodes using a WHERE clause on the 'year' property.",
    },
    {
        "category": "Relationship Traversal",
        "name": "Who acted in 'GraphWorld'?",
        "cypher": "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)\nWHERE m.title = 'GraphWorld'\nRETURN p.name, r.role",
        "description": "Traverse ACTED_IN relationships to find actors of a specific movie.",
    },
    {
        "category": "Relationship Traversal",
        "name": "Which movies did Alice act in?",
        "cypher": "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)\nWHERE p.name = 'Alice'\nRETURN m.title, r.role, m.year",
        "description": "Find all movies a specific person acted in by traversing outgoing ACTED_IN edges.",
    },
    {
        "category": "Multi-hop Traversal",
        "name": "Friends of Alice (1 hop)",
        "cypher": "MATCH (a:Person)-[:FRIENDS_WITH]-(b:Person)\nWHERE a.name = 'Alice'\nRETURN b.name, b.city",
        "description": "Find direct friends of Alice (1-hop undirected traversal).",
    },
    {
        "category": "Multi-hop Traversal",
        "name": "Friends-of-friends of Alice (2 hops)",
        "cypher": "MATCH (a:Person)-[:FRIENDS_WITH]-(b:Person)-[:FRIENDS_WITH]-(c:Person)\nWHERE a.name = 'Alice' AND c.name <> a.name\nRETURN DISTINCT c.name, c.city",
        "description": "Discover indirect connections — people two friendship hops away from Alice.",
    },
    {
        "category": "Multi-hop Traversal",
        "name": "All people within 3 hops of Alice",
        "cypher": "MATCH path = (a:Person)-[:FRIENDS_WITH*1..3]-(b:Person)\nWHERE a.name = 'Alice' AND b.name <> 'Alice'\nRETURN DISTINCT b.name, b.city, length(path) AS hops",
        "description": "Variable-length path query: find everyone reachable within 3 friendship hops.",
    },
    {
        "category": "Aggregation",
        "name": "Count movies per actor",
        "cypher": "MATCH (p:Person)-[:ACTED_IN]->(m:Movie)\nRETURN p.name, COUNT(m) AS movie_count\nORDER BY movie_count DESC",
        "description": "Use COUNT aggregation to find how many movies each person acted in.",
    },
    {
        "category": "Aggregation",
        "name": "Average review rating per movie",
        "cypher": "MATCH (p:Person)-[r:REVIEWED]->(m:Movie)\nRETURN m.title, AVG(r.rating) AS avg_rating, COUNT(p) AS num_reviews\nORDER BY avg_rating DESC",
        "description": "Aggregate REVIEWED relationships to compute average ratings.",
    },
    {
        "category": "Complex Pattern",
        "name": "Co-actors: people who acted in the same movie",
        "cypher": "MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(b:Person)\nWHERE a.name < b.name\nRETURN a.name, b.name, m.title",
        "description": "Find pairs of people who co-acted in the same movie using a diamond pattern.",
    },
    {
        "category": "Complex Pattern",
        "name": "Actors who are also friends",
        "cypher": "MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(b:Person),\n      (a)-[:FRIENDS_WITH]-(b)\nRETURN a.name, b.name, m.title",
        "description": "Multi-pattern query: find co-actors who are also friends with each other.",
    },
    {
        "category": "Complex Pattern",
        "name": "Director → Movie ← Actor chain",
        "cypher": "MATCH (d:Director)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(p:Person)\nRETURN d.name AS director, m.title AS movie, p.name AS actor",
        "description": "Traverse a Director→Movie←Actor pattern to see which actors worked with which directors.",
    },
]


def execute_query(query_name, nodes, edges):
    """
    Execute a predefined query against the in-memory graph.
    Returns (results_df, highlighted_node_ids, highlighted_edge_indices, explanation).
    """
    node_map, adj_out, adj_in, adj_any = _build_index(nodes, edges)
    results = []
    h_nodes = set()
    h_edges = set()
    explanation = ""

    # ---- Basic Pattern Matches ----
    if query_name == "Find all Person nodes":
        for n in nodes:
            if n["label"] == "Person":
                results.append({"name": n["properties"]["name"],
                                "age": n["properties"]["age"],
                                "city": n["properties"]["city"]})
                h_nodes.add(n["id"])
        explanation = "Scanned all nodes, filtered by label = 'Person'. Returned 6 Person nodes with their properties."

    elif query_name == "Find all Movies released after 2020":
        for n in nodes:
            if n["label"] == "Movie" and n["properties"].get("year", 0) > 2020:
                results.append({"title": n["properties"]["title"],
                                "year": n["properties"]["year"],
                                "genre": n["properties"]["genre"]})
                h_nodes.add(n["id"])
        explanation = "Filtered Movie nodes where year > 2020. Demonstrates property-based filtering with WHERE."

    # ---- Relationship Traversal ----
    elif query_name == "Who acted in 'GraphWorld'?":
        for i, e in enumerate(edges):
            if e["type"] == "ACTED_IN" and e["to"] == "m1":
                actor = node_map[e["from"]]
                results.append({"name": actor["properties"]["name"],
                                "role": e["properties"].get("role", "")})
                h_nodes.add(e["from"])
                h_nodes.add("m1")
                h_edges.add(i)
        explanation = "Traversed ACTED_IN edges pointing to 'GraphWorld' (m1). Found all actors and their roles."

    elif query_name == "Which movies did Alice act in?":
        for i, e in enumerate(edges):
            if e["type"] == "ACTED_IN" and e["from"] == "alice":
                movie = node_map[e["to"]]
                results.append({"title": movie["properties"]["title"],
                                "role": e["properties"].get("role", ""),
                                "year": movie["properties"]["year"]})
                h_nodes.add("alice")
                h_nodes.add(e["to"])
                h_edges.add(i)
        explanation = "Starting from Alice node, followed all outgoing ACTED_IN relationships to find her movies."

    # ---- Multi-hop Traversal ----
    elif query_name == "Friends of Alice (1 hop)":
        for i, e in enumerate(edges):
            if e["type"] == "FRIENDS_WITH":
                if e["from"] == "alice":
                    friend = node_map[e["to"]]
                    results.append({"name": friend["properties"]["name"],
                                    "city": friend["properties"]["city"]})
                    h_nodes.add("alice")
                    h_nodes.add(e["to"])
                    h_edges.add(i)
                elif e["to"] == "alice":
                    friend = node_map[e["from"]]
                    results.append({"name": friend["properties"]["name"],
                                    "city": friend["properties"]["city"]})
                    h_nodes.add("alice")
                    h_nodes.add(e["from"])
                    h_edges.add(i)
        explanation = "Found direct friends of Alice by following FRIENDS_WITH edges in both directions (1-hop)."

    elif query_name == "Friends-of-friends of Alice (2 hops)":
        # 1-hop friends
        friends_1 = set()
        for e in edges:
            if e["type"] == "FRIENDS_WITH":
                if e["from"] == "alice":
                    friends_1.add(e["to"])
                elif e["to"] == "alice":
                    friends_1.add(e["from"])
        # 2-hop
        friends_2 = set()
        for f1 in friends_1:
            for i, e in enumerate(edges):
                if e["type"] == "FRIENDS_WITH":
                    other = None
                    if e["from"] == f1:
                        other = e["to"]
                    elif e["to"] == f1:
                        other = e["from"]
                    if other and other != "alice" and other not in friends_1:
                        friends_2.add(other)
                        h_edges.add(i)
            # Also mark the first hop edges
            for i, e in enumerate(edges):
                if e["type"] == "FRIENDS_WITH" and (
                    (e["from"] == "alice" and e["to"] == f1) or
                    (e["to"] == "alice" and e["from"] == f1)
                ):
                    h_edges.add(i)

        h_nodes.add("alice")
        h_nodes.update(friends_1)
        h_nodes.update(friends_2)
        for f2 in friends_2:
            n = node_map[f2]
            results.append({"name": n["properties"]["name"],
                            "city": n["properties"]["city"]})
        explanation = "2-hop traversal: Alice → friends → friends-of-friends. Excludes Alice and direct friends from results."

    elif query_name == "All people within 3 hops of Alice":
        # BFS up to 3 hops
        visited = {"alice": 0}
        queue = [("alice", 0)]
        path_edges = set()
        while queue:
            current, depth = queue.pop(0)
            if depth >= 3:
                continue
            for i, e in enumerate(edges):
                if e["type"] == "FRIENDS_WITH":
                    neighbor = None
                    if e["from"] == current:
                        neighbor = e["to"]
                    elif e["to"] == current:
                        neighbor = e["from"]
                    if neighbor and neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
                        path_edges.add(i)
        h_nodes = set(visited.keys())
        h_edges = path_edges
        for nid, hops in visited.items():
            if nid != "alice" and node_map[nid]["label"] == "Person":
                n = node_map[nid]
                results.append({"name": n["properties"]["name"],
                                "city": n["properties"]["city"],
                                "hops": hops})
        results.sort(key=lambda x: x["hops"])
        explanation = "BFS traversal up to 3 hops from Alice via FRIENDS_WITH. Shows variable-length path matching [*1..3]."

    # ---- Aggregation ----
    elif query_name == "Count movies per actor":
        counts = defaultdict(int)
        actor_edges = set()
        for i, e in enumerate(edges):
            if e["type"] == "ACTED_IN":
                counts[e["from"]] += 1
                h_nodes.add(e["from"])
                h_nodes.add(e["to"])
                actor_edges.add(i)
        h_edges = actor_edges
        for nid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            results.append({"name": node_map[nid]["properties"]["name"],
                            "movie_count": cnt})
        explanation = "Grouped ACTED_IN edges by source Person, counted target Movies. Ordered descending by count."

    elif query_name == "Average review rating per movie":
        movie_ratings = defaultdict(list)
        for i, e in enumerate(edges):
            if e["type"] == "REVIEWED":
                movie_ratings[e["to"]].append(e["properties"].get("rating", 0))
                h_nodes.add(e["from"])
                h_nodes.add(e["to"])
                h_edges.add(i)
        for mid, ratings in sorted(movie_ratings.items(), key=lambda x: -np.mean(x[1])):
            results.append({
                "title": node_map[mid]["properties"]["title"],
                "avg_rating": round(np.mean(ratings), 2),
                "num_reviews": len(ratings)
            })
        explanation = "Aggregated REVIEWED relationships: computed AVG(rating) and COUNT per Movie. Sorted by rating DESC."

    # ---- Complex Patterns ----
    elif query_name == "Co-actors: people who acted in the same movie":
        # Find pairs
        movie_actors = defaultdict(list)
        for i, e in enumerate(edges):
            if e["type"] == "ACTED_IN":
                movie_actors[e["to"]].append(e["from"])
                h_edges.add(i)
        for mid, actors in movie_actors.items():
            for i_a in range(len(actors)):
                for j_a in range(i_a + 1, len(actors)):
                    a, b = actors[i_a], actors[j_a]
                    na, nb = node_map[a]["properties"]["name"], node_map[b]["properties"]["name"]
                    if na < nb:
                        results.append({"actor_1": na, "actor_2": nb,
                                        "movie": node_map[mid]["properties"]["title"]})
                    else:
                        results.append({"actor_1": nb, "actor_2": na,
                                        "movie": node_map[mid]["properties"]["title"]})
                    h_nodes.add(a)
                    h_nodes.add(b)
                    h_nodes.add(mid)
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = (r["actor_1"], r["actor_2"], r["movie"])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        results = unique
        explanation = "Diamond pattern: (a)-[:ACTED_IN]->(m)<-[:ACTED_IN]-(b). Found all co-actor pairs per movie."

    elif query_name == "Actors who are also friends":
        # First find co-actors
        movie_actors = defaultdict(list)
        for e in edges:
            if e["type"] == "ACTED_IN":
                movie_actors[e["to"]].append(e["from"])
        # Find friend pairs
        friend_pairs = set()
        for i, e in enumerate(edges):
            if e["type"] == "FRIENDS_WITH":
                friend_pairs.add(frozenset([e["from"], e["to"]]))
        # Intersect
        for mid, actors in movie_actors.items():
            for i_a in range(len(actors)):
                for j_a in range(i_a + 1, len(actors)):
                    pair = frozenset([actors[i_a], actors[j_a]])
                    if pair in friend_pairs:
                        a, b = sorted([node_map[actors[i_a]]["properties"]["name"],
                                       node_map[actors[j_a]]["properties"]["name"]])
                        results.append({"actor_1": a, "actor_2": b,
                                        "movie": node_map[mid]["properties"]["title"]})
                        h_nodes.add(actors[i_a])
                        h_nodes.add(actors[j_a])
                        h_nodes.add(mid)
        # Mark relevant edges
        for i, e in enumerate(edges):
            if e["from"] in h_nodes and e["to"] in h_nodes:
                h_edges.add(i)
        explanation = "Multi-pattern match: co-actors who are also FRIENDS_WITH each other. Combines two pattern clauses."

    elif query_name == "Director → Movie ← Actor chain":
        for i, e_dir in enumerate(edges):
            if e_dir["type"] == "DIRECTED":
                director = node_map[e_dir["from"]]
                movie = node_map[e_dir["to"]]
                h_nodes.add(e_dir["from"])
                h_nodes.add(e_dir["to"])
                h_edges.add(i)
                for j, e_act in enumerate(edges):
                    if e_act["type"] == "ACTED_IN" and e_act["to"] == e_dir["to"]:
                        actor = node_map[e_act["from"]]
                        results.append({
                            "director": director["properties"]["name"],
                            "movie": movie["properties"]["title"],
                            "actor": actor["properties"]["name"]
                        })
                        h_nodes.add(e_act["from"])
                        h_edges.add(j)
        explanation = "Pattern: (d:Director)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(p:Person). Maps directors to their actors."

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    return results_df, h_nodes, h_edges, explanation


# ======================================================================================
# 3. GRAPH VISUALIZATION HELPERS
# ======================================================================================

LABEL_COLORS = {
    "Person":   {"bg": "#4F46E5", "border": "#3730A3", "font": "#FFFFFF"},   # Indigo
    "Movie":    {"bg": "#DC2626", "border": "#991B1B", "font": "#FFFFFF"},   # Red
    "Director": {"bg": "#059669", "border": "#047857", "font": "#FFFFFF"},   # Emerald
}

EDGE_COLORS = {
    "ACTED_IN":     "#F59E0B",   # Amber
    "DIRECTED":     "#10B981",   # Green
    "FRIENDS_WITH": "#6366F1",   # Indigo
    "REVIEWED":     "#EC4899",   # Pink
}


def render_pyvis_graph(nodes, edges, highlight_nodes=None, highlight_edges=None, height="520px"):
    """
    Render an interactive graph using pyvis.
    Returns an HTML string.
    """
    if highlight_nodes is None:
        highlight_nodes = set()
    if highlight_edges is None:
        highlight_edges = set()

    net = Network(height=height, width="100%", bgcolor="#0F172A", font_color="white",
                  directed=True, notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150, spring_strength=0.05)

    for n in nodes:
        lbl = n["label"]
        colors = LABEL_COLORS.get(lbl, {"bg": "#64748B", "border": "#475569", "font": "#FFFFFF"})
        # Display name
        if lbl == "Movie":
            display = n["properties"].get("title", n["id"])
        else:
            display = n["properties"].get("name", n["id"])

        # Tooltip
        props_str = "\n".join(f"  {k}: {v}" for k, v in n["properties"].items())
        tooltip = f"[{lbl}] {display}\n{props_str}"

        is_highlighted = n["id"] in highlight_nodes
        node_color = colors["bg"] if not is_highlighted else "#FACC15"  # Yellow highlight
        border_color = colors["border"] if not is_highlighted else "#EAB308"
        size = 30 if not is_highlighted else 38

        net.add_node(
            n["id"],
            label=display,
            title=tooltip,
            color={"background": node_color, "border": border_color,
                   "highlight": {"background": "#FACC15", "border": "#EAB308"}},
            font={"color": colors["font"] if not is_highlighted else "#000000", "size": 13,
                  "face": "Inter, Arial, sans-serif", "bold": {"color": colors["font"]}},
            size=size,
            shape="dot",
            borderWidth=2 if not is_highlighted else 4,
        )

    for i, e in enumerate(edges):
        is_highlighted = i in highlight_edges
        edge_color = EDGE_COLORS.get(e["type"], "#94A3B8")
        if is_highlighted:
            edge_color = "#FACC15"

        props_str = ", ".join(f"{k}: {v}" for k, v in e.get("properties", {}).items())
        tooltip = f"{e['type']}" + (f" ({props_str})" if props_str else "")

        # For FRIENDS_WITH, use undirected arrow style
        arrows = "to" if e["type"] != "FRIENDS_WITH" else ""

        net.add_edge(
            e["from"], e["to"],
            title=tooltip,
            label=e["type"],
            color={"color": edge_color, "highlight": "#FACC15"},
            width=2 if not is_highlighted else 4,
            arrows=arrows,
            font={"size": 9, "color": "#94A3B8", "strokeWidth": 0, "align": "middle"},
            smooth={"type": "curvedCW", "roundness": 0.15}
        )

    # Generate HTML
    html = net.generate_html()
    # Inject a small style fix for the canvas
    html = html.replace("<body>", '<body style="margin:0; padding:0; background:#0F172A;">')
    return html


# ======================================================================================
# 4. QUIZ QUESTIONS
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
        "explanation": "MATCH is the primary read clause in Cypher. It specifies a graph pattern and finds all subgraphs in the database that match it."
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
        "explanation": "Relationships in Cypher are written as -[r:TYPE]-> for directed edges, using square brackets for the relationship variable and type, and dashes/arrows for direction."
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
        "explanation": "The [*1..3] syntax matches variable-length paths with a minimum of 1 hop and a maximum of 3 hops, enabling multi-hop traversal."
    },
    {
        "id": 4,
        "question": "In the query MATCH (a)-[:ACTED_IN]->(m)<-[:ACTED_IN]-(b), what pattern is being matched?",
        "options": [
            "A) Two people who directed the same movie",
            "B) A person and the movie's director",
            "C) Two people (co-actors) who both acted in the same movie",
            "D) A person reviewing a movie they directed"
        ],
        "answer_index": 2,
        "explanation": "This 'diamond' pattern finds pairs of Person nodes that both have ACTED_IN relationships pointing to the same Movie node — i.e., co-actors."
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
        "explanation": "WHERE acts as a filter applied after pattern matching. It evaluates boolean conditions on node/relationship properties to narrow results."
    },
    {
        "id": 6,
        "question": "Which aggregation function would you use to find how many movies each actor appeared in?",
        "options": [
            "A) SUM(m.title)",
            "B) AVG(m)",
            "C) COUNT(m)",
            "D) COLLECT(m.year)"
        ],
        "answer_index": 2,
        "explanation": "COUNT(m) counts the number of matched Movie nodes per group (per actor), which gives the number of movies each actor appeared in."
    },
    {
        "id": 7,
        "question": "What advantage do graph databases have over relational databases for multi-hop queries?",
        "options": [
            "A) They use less disk space for all types of data",
            "B) They provide constant-time traversal regardless of dataset size",
            "C) They automatically generate SQL queries",
            "D) They don't require any query language"
        ],
        "answer_index": 1,
        "explanation": "Graph databases store relationships as pointers/edges, enabling O(1) traversal per hop. Relational databases require expensive JOINs that scale with table size."
    },
    {
        "id": 8,
        "question": "What does RETURN DISTINCT do in a Cypher query?",
        "options": [
            "A) Returns only the first result",
            "B) Removes duplicate rows from the result set",
            "C) Returns results in a random order",
            "D) Converts all values to uppercase"
        ],
        "answer_index": 1,
        "explanation": "DISTINCT eliminates duplicate result rows, ensuring each unique combination of returned values appears only once."
    },
    {
        "id": 9,
        "question": "In a knowledge graph, what is a 'hidden connection'?",
        "options": [
            "A) A node that has been deleted from the database",
            "B) An indirect relationship between entities discoverable only through multi-hop traversal",
            "C) A property that is encrypted",
            "D) A relationship with no properties"
        ],
        "answer_index": 1,
        "explanation": "Hidden connections are indirect relationships between entities that are not directly linked but can be discovered by traversing intermediate nodes and edges."
    },
    {
        "id": 10,
        "question": "What does the COLLECT() function do in Cypher?",
        "options": [
            "A) Counts the number of matched patterns",
            "B) Aggregates values into a list/array",
            "C) Removes duplicate nodes from results",
            "D) Sorts results in ascending order"
        ],
        "answer_index": 1,
        "explanation": "COLLECT() aggregates matched values into a list. For example, COLLECT(m.title) would return all movie titles as an array for each group."
    }
]


# ======================================================================================
# 5. LAB REPORT PDF EXPORTER
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name, student_id, date_str,
                        trials_df, quiz_score, quiz_total, student_notes):
    """Compiles experiment records into a formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")

    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score/quiz_total)*100 if quiz_total else 0)}%)", 1)

    pdf.ln(12)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Query Trials & Results", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No query trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = list(trials_df.columns)
        num_cols = len(cols)
        col_w = max(18, int(190 / max(1, num_cols)))

        for c in cols:
            pdf.cell(col_w, 6, str(c)[:18], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
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

    # 3. Discussion & Notes
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The Cypher query experiments demonstrated effective graph pattern matching across varying "
        "traversal depths and aggregation operations, confirming the efficiency of graph-based retrieval."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 6. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, and Procedure."""
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
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Variable", "Definition & Role"]
        )
        st.table(var_df)


def render_simulation_section():
    """Renders Section 2: Interactive Graph Simulation with Cypher Query Execution."""
    st.header("Interactive Cypher Query Simulation")
    st.info(
        "Explore the knowledge graph below and run Cypher-like queries to discover patterns, "
        "traverse relationships, and aggregate data. No Neo4j required — all queries execute "
        "against an in-memory graph."
    )

    nodes, edges = build_sample_graph()

    # --- Graph Legend ---
    st.subheader("Knowledge Graph")
    legend_cols = st.columns(7)
    with legend_cols[0]:
        st.markdown("🟣 **Person**")
    with legend_cols[1]:
        st.markdown("🔴 **Movie**")
    with legend_cols[2]:
        st.markdown("🟢 **Director**")
    with legend_cols[3]:
        st.markdown("🟡 ACTED_IN")
    with legend_cols[4]:
        st.markdown("🟢 DIRECTED")
    with legend_cols[5]:
        st.markdown("🟣 FRIENDS_WITH")
    with legend_cols[6]:
        st.markdown("🩷 REVIEWED")

    # --- Query Selection ---
    st.divider()
    st.subheader("Query Workbench")

    # Group queries by category
    categories = {}
    for q in PREDEFINED_QUERIES:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)

    col_cat, col_query = st.columns([1, 2])
    with col_cat:
        selected_category = st.selectbox(
            "Query Category",
            options=list(categories.keys()),
            index=0
        )
    with col_query:
        query_names = [q["name"] for q in categories[selected_category]]
        selected_query_name = st.selectbox(
            "Select Query",
            options=query_names,
            index=0
        )

    # Find the selected query object
    selected_query = None
    for q in PREDEFINED_QUERIES:
        if q["name"] == selected_query_name:
            selected_query = q
            break

    # Display Cypher code
    st.markdown("**Cypher Query:**")
    st.code(selected_query["cypher"], language="cypher")
    st.caption(f"📝 {selected_query['description']}")

    # Run button
    run_pressed = st.button("▶ Run Query", type="primary", use_container_width=True)

    if run_pressed or st.session_state.get("last_query") == selected_query_name:
        st.session_state["last_query"] = selected_query_name

        results_df, h_nodes, h_edges, explanation = execute_query(
            selected_query_name, nodes, edges
        )

        # Metrics
        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Rows Returned", len(results_df))
        with m2:
            st.metric("Nodes Matched", len(h_nodes))
        with m3:
            st.metric("Edges Traversed", len(h_edges))

        # Results Table
        if not results_df.empty:
            st.subheader("Query Results")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No results returned for this query.")

        # Explanation
        st.info(f"**Execution Explanation:** {explanation}")

        # Highlighted Graph
        st.subheader("Graph Visualization (Matched Pattern Highlighted)")
        graph_html = render_pyvis_graph(nodes, edges, h_nodes, h_edges)
        components.html(graph_html, height=550, scrolling=False)

        # Store for trial logging
        st.session_state["current_sim_result"] = {
            "query_name": selected_query_name,
            "category": selected_query["category"],
            "rows_returned": len(results_df),
            "nodes_matched": len(h_nodes),
            "edges_traversed": len(h_edges),
        }
    else:
        # Show full graph without highlights
        st.subheader("Full Knowledge Graph (Interactive)")
        graph_html = render_pyvis_graph(nodes, edges)
        components.html(graph_html, height=550, scrolling=False)

    # --- Data Logger ---
    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture the current query trial into your session log:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            sim_result = st.session_state.get("current_sim_result")
            if sim_result:
                trial_record = {
                    "Trial #": len(st.session_state["trials"]) + 1,
                    "Query": sim_result["query_name"],
                    "Category": sim_result["category"],
                    "Rows": sim_result["rows_returned"],
                    "Nodes": sim_result["nodes_matched"],
                    "Edges": sim_result["edges_traversed"],
                    "Timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["trials"].append(trial_record)
                st.toast(f"✅ Trial #{trial_record['Trial #']} recorded!")
            else:
                st.toast("⚠️ Run a query first before recording a trial.")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("🗑️ Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="cypher_query_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Run a query and click 'Record Current Trial' to begin.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Concept Assessment Quiz")
    st.write("Answer the questions below to evaluate your understanding of Cypher queries and graph pattern matching.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                         f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                         f"**Reasoning:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already submitted. Current score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with PDF Export."""
    st.header("Report Generation")
    st.write("Compile your student details, recorded query trials, and quiz evaluation into an official PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name",
                                     value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID",
                                   value=st.session_state["student_info"].get("id", "EXP-001"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", (
            "The Cypher query experiments demonstrated effective graph pattern matching across varying "
            "traversal depths and aggregation operations, confirming the efficiency of graph-based retrieval "
            "for discovering hidden connections."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: You have not recorded any query trials in the Simulation tab yet. Your report will indicate 0 trials.")

    # Generate PDF
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes
    )

    # Save to local files
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# ======================================================================================
# 7. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = None
    if "current_sim_result" not in st.session_state:
        st.session_state["current_sim_result"] = None


def main():
    st.set_page_config(
        page_title="Advanced Cypher Queries & Graph Pattern Matching — Virtual Lab",
        page_icon="🔗",
        layout="wide"
    )

    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])

    # Navigation Sidebar
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    st.sidebar.write(f"- **Trials Recorded:** {len(st.session_state['trials'])}")
    quiz_status = "✅ Done" if st.session_state.get("quiz_submitted", False) else "⏳ Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")

    st.sidebar.divider()
    st.sidebar.caption("Virtual Lab — IIT Kharagpur Format")
    st.sidebar.caption("Graph experiments use in-memory simulation (no Neo4j).")

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
