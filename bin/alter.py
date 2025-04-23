import pickle
from pathlib import Path

# Load the manifest - use the correct filename
manifest_path = Path("cache/queries_item_pool.pkl")
with open(manifest_path, 'rb') as f:
    manifest = pickle.load(f)

# Update paths to point to the new data directory
manifest["requests_file"] = str(Path("cache/temp/requests.pkl"))
manifest["item_pool_chunks"] = [
    str(Path(f"cache/temp/{Path(chunk_path).name}"))
    for chunk_path in manifest["item_pool_chunks"]
]

# Save the updated manifest
with open(manifest_path, 'wb') as f:
    pickle.dump(manifest, f)