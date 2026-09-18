# 🔗 Advanced Cypher Queries and Graph Pattern Matching — Virtual Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://advanced-cypher-queries-and-graph-pattern-matching---vl.streamlit.app/)

Welcome to the **Advanced Cypher Queries and Graph Pattern Matching** Virtual Lab! This interactive application is designed to help students, developers, and data enthusiasts understand and experiment with graph databases, pattern matching, and the Cypher query language without needing to install or configure a dedicated graph database like Neo4j.

**🚀 Live Demo:** [Play with the Virtual Lab Here](https://advanced-cypher-queries-and-graph-pattern-matching---vl.streamlit.app/)

---

## 📖 Overview

Traditional relational databases use expensive JOIN operations to connect tables, which become painfully slow as the number of "hops" (connections) grows. Graph databases solve this by treating relationships as first-class citizens, enabling constant-time traversal (O(1)) regardless of the dataset size. 

This virtual lab simulates an in-memory graph engine and visualizer to teach the core concepts of graph theory and **Cypher**, the declarative graph query language standardized by ISO (GQL). 

### Key Features
* **Graph Studio:** Build a custom knowledge graph from scratch or load pre-built templates (Movie Knowledge Graph, Social Network, Academic Network).
* **Interactive Visualization:** Explore nodes and relationships dynamically using PyVis physics-based rendering.
* **Cypher Query Lab:** Execute predefined advanced graph operations and view the equivalent Cypher syntax.
* **🤖 AI Graph Assistant:** Powered by Groq, ask questions in plain English. The AI translates them into Cypher, runs them against your graph, and explains the results step-by-step!
* **Self-Assessment & Reports:** Take a quiz to test your knowledge and generate a downloadable PDF Lab Report tracking your experimental queries.

---

## 🔍 Advanced Graph Queries Explored

The core of this lab is the **Cypher Query Lab**, which demonstrates how to traverse and analyze graph data. Here are the types of advanced queries you can perform:

### 1. Multi-hop Traversal (BFS)
* **What it does:** Discovers indirect connections by following a chain of relationships across multiple nodes (e.g., "Friends of friends").
* **Cypher Syntax:** `MATCH (a)-[*1..3]->(b)`
* **Use Case:** Finding a person's extended network within 3 degrees of separation.

### 2. Shortest Path 
* **What it does:** Finds the most direct route between two specific nodes in the graph using Breadth-First Search.
* **Cypher Syntax:** `MATCH path = shortestPath((a)-[*]-(b)) RETURN path`
* **Use Case:** Finding the shortest connection between an actor and a director in a movie graph.

### 3. Complex Pattern Matching
* **What it does:** Searches for specific structural patterns in the graph based on node labels and relationship types.
* **Cypher Syntax:** `MATCH (a:Person)-[:ACTED_IN]->(b:Movie)`
* **Use Case:** Finding all "Person" entities that have an "ACTED_IN" relationship targeting a "Movie" entity.

### 4. Co-occurrence (Diamond Patterns)
* **What it does:** Finds pairs of nodes that share a connection to the exact same intermediate node.
* **Cypher Syntax:** `MATCH (a)-[:ACTED_IN]->(m)<-[:ACTED_IN]-(b) WHERE a <> b`
* **Use Case:** Discovering co-actors (two different actors who acted in the same movie) or users who liked the same post.

### 5. Aggregation
* **What it does:** Summarizes graph data by grouping results and applying functions like `COUNT()`, `COLLECT()`, or `AVG()`.
* **Cypher Syntax:** `MATCH (a)-[:ACTED_IN]->(m) RETURN a.name, COUNT(m) AS movies_count`
* **Use Case:** Finding out how many movies each actor has starred in, sorted by the busiest actor.

---

## 🛠️ Technology Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/)
* **Graph Visualization:** [PyVis](https://pyvis.readthedocs.io/) & NetworkX concepts
* **AI Integration:** [Groq API](https://groq.com/) (Using instruction-tuned OSS Models)
* **Data Manipulation:** Pandas & NumPy
* **Report Generation:** FPDF

---

## 💻 Running Locally

If you want to run this virtual lab on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sudarshan026/Advanced-Cypher-Queries-and-Graph-Pattern-Matching---Virtual-Lab.git
   cd Advanced-Cypher-Queries-and-Graph-Pattern-Matching---Virtual-Lab
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app_dynamic.py
   ```

*(Optional)* To use the AI Assistant locally, you can paste your Groq API key directly into the sidebar of the running app!

---

*This project was developed to provide an accessible, hands-on learning environment for graph database concepts and Cypher querying.*
