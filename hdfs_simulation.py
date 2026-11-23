import os
import json
import shutil
import pandas as pd
from datetime import datetime

SOURCE_FILE = "data/customers.csv"
HDFS_ROOT = "hdfs_simulation"
BLOCKS_DIR = os.path.join(HDFS_ROOT, "blocks")
REPLICAS_DIR = os.path.join(HDFS_ROOT, "replicas")

BLOCK_SIZE_ROWS = 250
REPLICATION_FACTOR = 2

# 1. Verify dataset exists
if not os.path.exists(SOURCE_FILE):
    print("Dataset missing! Run python generate_data.py first.")
    raise SystemExit

# 2. Read customer dataset
df = pd.read_csv(SOURCE_FILE)

# 3. Create simulated HDFS directories
os.makedirs(BLOCKS_DIR, exist_ok=True)
os.makedirs(REPLICAS_DIR, exist_ok=True)

# 4. Clear old block files to avoid duplicate data from previous runs
for folder in [BLOCKS_DIR, REPLICAS_DIR]:
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath)

# 5. Split the dataset into blocks
block_records = []

for start in range(0, len(df), BLOCK_SIZE_ROWS):
    end = min(start + BLOCK_SIZE_ROWS, len(df))

    block_df = df.iloc[start:end]

    block_number = len(block_records) + 1
    block_name = f"block_{block_number:03d}.csv"

    # Save original block
    original_path = os.path.join(BLOCKS_DIR, block_name)
    block_df.to_csv(original_path, index=False)

    # Create simulated replica copies
    replica_paths = []

    for replica_number in range(1, REPLICATION_FACTOR + 1):
        replica_folder = os.path.join(
            REPLICAS_DIR,
            f"node_{replica_number}"
        )

        os.makedirs(replica_folder, exist_ok=True)

        replica_path = os.path.join(replica_folder, block_name)

        shutil.copy2(original_path, replica_path)

        replica_paths.append(replica_path)

    block_records.append({
        "block_name": block_name,
        "rows": len(block_df),
        "original_location": original_path,
        "replicas": replica_paths
    })

# 6. Create storage report
report = {
    "system": "HDFS Storage Simulation",
    "dataset": SOURCE_FILE,
    "total_records": len(df),
    "block_size_rows": BLOCK_SIZE_ROWS,
    "total_blocks": len(block_records),
    "replication_factor": REPLICATION_FACTOR,
    "simulated_replica_copies": len(block_records) * REPLICATION_FACTOR,
    "storage_status": "SUCCESS",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "blocks": block_records
}

report_path = os.path.join(HDFS_ROOT, "storage_report.json")

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

# 7. Display results
print("=" * 50)
print("HDFS STORAGE SIMULATION COMPLETE")
print("=" * 50)

print(f"Dataset records      : {len(df)}")
print(f"Block size          : {BLOCK_SIZE_ROWS} rows")
print(f"Total blocks        : {len(block_records)}")
print(f"Replication factor   : {REPLICATION_FACTOR}")
print(f"Replica copies      : {len(block_records) * REPLICATION_FACTOR}")
print(f"Storage report      : {report_path}")

print("\nBlock details:")

for block in block_records:
    print(f"{block['block_name']} | Rows: {block['rows']}")

print("\nStorage status: SUCCESS")
print("Note: This is a local HDFS concept simulation, not a real Hadoop cluster.")