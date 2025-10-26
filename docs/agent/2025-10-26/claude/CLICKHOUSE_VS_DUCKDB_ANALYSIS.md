# ClickHouse vs DuckDB for Miner Template
## Database Technology Decision Analysis

**Date**: 2025-10-26  
**Decision**: Which database technology for miner template data storage?

---

## Context

The miner template needs to:
1. Store pre-computed results (alert scores, rankings, cluster scores)
2. Serve these results via API to Bittensor subnet miner
3. Support efficient queries by date, alert_id, etc.
4. Be easy to deploy and maintain

## Option 1: DuckDB (Embedded Analytics)

### Architecture
```
Miner Template:
  scripts/process_batch.py → Parquet files (output/{date}/)
                                    ↓
  DuckDB (in-process) ← reads Parquet directly (zero-copy)
                                    ↓
  FastAPI ← queries DuckDB ← serves results
```

### Pros
✅ **Zero infrastructure** - No separate database server  
✅ **Lightweight** - Just a Python package (`pip install duckdb`)  
✅ **Simple deployment** - No database administration  
✅ **Zero-copy Parquet** - Reads Parquet files directly without import  
✅ **In-process** - Fast, no network overhead  
✅ **Perfect for analytics** - Built for OLAP workloads  
✅ **Easy for miners** - No DevOps knowledge required  
✅ **Portable** - Can run on any machine  

### Cons
❌ **Different from SOT** - SOT uses ClickHouse  
❌ **Single-node only** - Can't scale horizontally (but miners are single-node anyway)  
❌ **No distributed queries** - Not needed for miner use case  

### Deployment Complexity
**Extremely Simple:**
```bash
# That's it - no server to run
pip install duckdb
python scripts/process_batch.py
python -m aml_miner.api.server
```

---

## Option 2: ClickHouse (Distributed OLAP Database)

### Architecture
```
Miner Template:
  scripts/process_batch.py → Parquet files
                                    ↓
  Load into ClickHouse (separate server process)
                                    ↓
  FastAPI ← queries ClickHouse ← serves results
```

### Pros
✅ **Consistency with SOT** - Same database as Source of Truth  
✅ **Powerful analytics** - Best-in-class for OLAP  
✅ **Mature ecosystem** - Production-proven  
✅ **Distributed** - Can scale (though not needed)  
✅ **Time-series optimized** - Good for date-based queries  

### Cons
❌ **Requires server** - Miners must run ClickHouse daemon  
❌ **Complex deployment** - Database administration needed  
❌ **Heavy dependency** - Large installation, memory overhead  
❌ **Network overhead** - Client-server architecture  
❌ **Data import needed** - Must load Parquet → ClickHouse  
❌ **Barrier to entry** - Many miners won't want to run database server  
❌ **Resource intensive** - RAM, CPU, disk for database process  

### Deployment Complexity
**Significantly More Complex:**
```bash
# Install ClickHouse server
sudo apt-get install clickhouse-server clickhouse-client

# Start ClickHouse daemon
sudo service clickhouse-server start

# Configure database, users, permissions
clickhouse-client --query "CREATE DATABASE miner_results"

# Import data (ongoing process)
clickhouse-client --query "INSERT INTO miner_results.alert_scores ..."

# Python client
pip install clickhouse-driver

# Run API
python -m aml_miner.api.server
```

**Plus:**
- Database monitoring
- Backup management
- Version upgrades
- Security configuration
- Network configuration

---

## Option 3: Hybrid - Parquet Only (No Database)

### Architecture
```
Miner Template:
  scripts/process_batch.py → Parquet files (output/{date}/)
                                    ↓
  FastAPI ← reads Parquet with pandas/polars ← serves results
```

### Pros
✅ **Simplest possible** - No database at all  
✅ **Zero dependencies** - Just pandas/polars  
✅ **Portable** - Parquet files are standard format  
✅ **No import step** - Direct file access  

### Cons
❌ **Inefficient for date ranges** - Must load many files  
❌ **No SQL queries** - Limited filtering capabilities  
❌ **Memory intensive** - Loading entire files  
❌ **Slow for large datasets** - No indexing  

---

## Comparison Matrix

| Feature | DuckDB | ClickHouse | Parquet Only |
|---------|--------|------------|--------------|
| **Installation** | `pip install duckdb` | Server setup, admin | `pip install pandas` |
| **Deployment** | ⭐⭐⭐⭐⭐ Trivial | ⭐⭐ Complex | ⭐⭐⭐⭐⭐ Trivial |
| **Performance** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good for small data |
| **Resource Use** | ⭐⭐⭐⭐⭐ Minimal | ⭐⭐⭐ Moderate-High | ⭐⭐⭐⭐ Minimal |
| **Query Power** | ⭐⭐⭐⭐⭐ Full SQL | ⭐⭐⭐⭐⭐ Full SQL | ⭐⭐ Pandas only |
| **Scalability** | ⭐⭐⭐⭐ Single-node | ⭐⭐⭐⭐⭐ Distributed | ⭐⭐ Limited |
| **Miner-Friendly** | ⭐⭐⭐⭐⭐ Yes | ⭐⭐ Requires expertise | ⭐⭐⭐⭐⭐ Yes |
| **Maintenance** | ⭐⭐⭐⭐⭐ None | ⭐⭐⭐ Ongoing DBA work | ⭐⭐⭐⭐⭐ None |

---

## Use Case Analysis

### Miner Template Requirements
1. **Easy deployment** - Miners should be able to run it with minimal setup
2. **Local processing** - Process Parquet files from `input/`, write to `output/`
3. **API serving** - Serve pre-computed results efficiently
4. **Date-based queries** - Filter by processing_date
5. **Single-node** - Each miner runs independently

### SOT (Source of Truth) vs Miner Template

**Different Use Cases:**

| Aspect | SOT (ClickHouse) | Miner Template |
|--------|------------------|----------------|
| **Purpose** | Centralized data warehouse | Local result storage |
| **Scale** | Petabytes, billions of rows | Gigabytes, millions of rows |
| **Users** | Validators, analysts, multiple consumers | Single miner's API |
| **Deployment** | Production infrastructure | Developer laptop/VPS |
| **Queries** | Complex analytics, joins, aggregations | Simple lookups by date |
| **Updates** | Continuous ingestion | Batch processing (daily) |

**Conclusion:** SOT and miner template have **different requirements**. What works for centralized infrastructure doesn't need to be used for local processing.

---

## Recommendation

### **Use DuckDB for Miner Template**

**Reasoning:**

1. **Target Audience** - Miners are developers, not DevOps engineers
   - DuckDB: `pip install duckdb` → done
   - ClickHouse: Server setup, database admin, ongoing maintenance

2. **Use Case Fit** - Local analytics, not distributed system
   - Single miner = single node → DuckDB is perfect
   - Don't need ClickHouse's distributed capabilities

3. **Simplicity** - Lower barrier to entry = more miners = better network
   - Complex setup → fewer miners willing to participate
   - Simple setup → innovation and competition thrive

4. **Performance** - DuckDB is excellent for this workload
   - Zero-copy Parquet reading
   - Analytical queries are fast
   - In-process = no network overhead

5. **Consistency Not Critical** - Miners don't need same DB as SOT
   - They process Parquet files (standard format)
   - They serve results via API (standard interface)
   - Internal implementation can differ

### Alternative: Start Simple, Upgrade Later

**Phase 1** (MVP): Parquet only with pandas
- Simplest possible
- Works for small datasets
- No database dependency

**Phase 2** (Optimization): Add DuckDB
- When performance becomes issue
- Easy migration (just add DuckDB queries)
- Backward compatible (Parquet files remain)

**Phase 3** (Advanced): ClickHouse as option
- For miners who want it
- Document as advanced deployment
- Most miners stick with DuckDB

---

## Implementation Plan

### Recommended: DuckDB Integration

```python
# aml_miner/api/database.py
import duckdb
from pathlib import Path

class ResultsDatabase:
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.conn = duckdb.connect(":memory:")  # In-memory, reads Parquet directly
        self._register_parquet_files()
    
    def _register_parquet_files(self):
        # Create views over Parquet files (zero-copy)
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW alert_scores AS
            SELECT *,
                CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
            FROM read_parquet('{self.output_dir}/*/alert_scores.parquet', filename=true)
        """)
```

### Optional: ClickHouse Support

For advanced users, document ClickHouse deployment as optional enhancement:

```markdown
# Advanced: ClickHouse Deployment (Optional)

For large-scale deployments or integration with existing ClickHouse infrastructure:

1. Install ClickHouse server
2. Configure database schema
3. Modify `aml_miner/api/database.py` to use clickhouse-driver
4. Set `DATABASE_TYPE=clickhouse` in config

Most miners should use the default DuckDB configuration.
```

---

## Summary

| Question | Answer |
|----------|--------|
| **For miner template?** | **DuckDB** |
| **Why not ClickHouse?** | Too heavy, complex deployment, barrier to entry |
| **Why not Parquet only?** | DuckDB adds SQL queries with minimal complexity |
| **Consistency with SOT?** | Not required - different use cases |
| **Future flexibility?** | Can add ClickHouse as optional advanced feature |

**Decision: Use DuckDB with option to upgrade to ClickHouse later if needed.**