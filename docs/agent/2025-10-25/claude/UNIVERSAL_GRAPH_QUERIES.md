# Universal Graph Queries for Memgraph

## Query 1: Complete Graph - All Nodes and Edges (SELECT *)

```cypher
// Universal query - shows ALL nodes and ALL relationships
MATCH path = (n)-[r]-(m)
RETURN path
LIMIT 500
```

**Use this for:** Complete graph visualization in Memgraph Lab

---

## Query 2: All Nodes with Their Relationships

```cypher
// Returns nodes and their outgoing relationships
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 500
```

---

## Query 3: Full Graph Starting from Patterns

```cypher
// Expand all connections from patterns (up to 3 hops)
MATCH path = (p:Pattern)-[*0..3]-(other)
RETURN path
LIMIT 300
```

---

## Query 4: Specific Pattern with Full Context

```cypher
// Replace pattern_id with your actual pattern ID
MATCH path = (p:Pattern {pattern_id: 'scc_f781cdfa045326d1'})-[*0..4]-(other)
RETURN path
```

---

## Query 5: All Relationship Types in Graph

```cypher
// See what relationship types exist
MATCH ()-[r]-()
RETURN type(r) as relationship_type, count(*) as count
ORDER BY count DESC
```

---

## Query 6: Sample of Each Relationship Type

```cypher
// Get examples of each relationship
MATCH (n)-[r]->(m)
WITH type(r) as rel_type, collect({start: labels(n)[0], end: labels(m)[0], rel: r}) as examples
RETURN rel_type, examples[0..3] as sample_relationships
```

---

## Query 7: Pattern → Alert → Cluster Chain

```cypher
// See the complete chain from patterns to clusters
MATCH path = (p:Pattern)-[*1..3]-(c:AlertCluster)
RETURN path
LIMIT 100
```

---

## Query 8: Address-Centric Complete Graph

```cypher
// Everything connected to a specific address
MATCH path = (addr:Address {address: '5CEqWarVTxfwNfZbyRXT6sCEj4tkVwhnxtpvL94Nr112GYuA'})-[*0..3]-(other)
RETURN path
```

---

## Query 9: All Nodes by Type with Count

```cypher
// Inventory of all node types
MATCH (n)
RETURN labels(n)[0] as node_type, count(*) as count
ORDER BY count DESC
```

---

## Query 10: Complete Subgraph (Patterns + Related Entities)

```cypher
// Get patterns and everything connected to them
MATCH (p:Pattern)
OPTIONAL MATCH (p)-[r1]-(n1)
OPTIONAL MATCH (n1)-[r2]-(n2)
RETURN p, r1, n1, r2, n2
LIMIT 200
```

---

## Recommended Query for Your Use Case

Based on your data (796 patterns, 494 addresses, 86 alerts, 25 clusters):

```cypher
// Best query for complete visualization with edges shown
MATCH path = (n)-[r]-(m)
WHERE n:Pattern OR n:Address OR n:Alert OR n:AlertCluster
RETURN path
LIMIT 1000
```

This will show ALL relationships between your four node types.

---

## Export Full Graph Data

```cypher
// Export everything (use in Memgraph Lab)
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
```

Then use Memgraph Lab's export feature to save as JSON or CSV.

---

## Performance Note

For graphs with 1000+ nodes, use LIMIT to avoid browser slowdown:
- For visualization: `LIMIT 500`
- For export: Remove LIMIT but export to file instead of viewing