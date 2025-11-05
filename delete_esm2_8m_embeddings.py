"""
Script to delete ESM2-8M embedding files from Modal volume.

This script connects to the Modal volume used by protify_modal_app.py
and deletes embedding files related to ESM2-8M/ESM2-8 model.

Usage:
    modal run delete_esm2_8m_embeddings.py
"""

import modal
from pathlib import Path

# Constants from protify_modal_app.py
APP_NAME = "protify-app"
VOLUME_NAME = "protify-data"
EMBEDDING_DIR = "/data/embeddings"

# Create Modal app
app = modal.App(f"{APP_NAME}-delete-embeddings")

# Use the same image as protify_modal_app.py (minimal requirements)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
)

# Get the volume
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,  # 5 minutes should be plenty
)
def delete_esm2_8m_embeddings():
    """
    Delete ESM2-8M embedding files from the Modal volume.
    
    Searches for files matching:
    - ESM2-8M_*.pth
    - ESM2-8_*.pth (ESM2-8 is the Protify model name for ESM2-8M)
    """
    embedding_dir = Path(EMBEDDING_DIR)
    
    if not embedding_dir.exists():
        print(f"❌ Embedding directory {EMBEDDING_DIR} does not exist on volume.")
        return {"success": False, "message": "Embedding directory not found", "deleted_files": []}
    
    # Patterns to match ESM2-8M embedding files
    # These cover both "ESM2-8M" and "ESM2-8" naming conventions
    patterns_to_match = [
        "ESM2-8M_*.pth",
        "ESM2-8_*.pth",
    ]
    
    deleted_files = []
    not_found_files = []
    
    # Search for files matching the patterns
    for pattern in patterns_to_match:
        matching_files = list(embedding_dir.glob(pattern))
        
        if not matching_files:
            print(f"ℹ️  No files found matching pattern: {pattern}")
            not_found_files.append(pattern)
        else:
            for file_path in matching_files:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()  # Delete the file
                    deleted_files.append({
                        "file": str(file_path),
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    })
                    print(f"✅ Deleted: {file_path} ({file_size / (1024 * 1024):.2f} MB)")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
                    return {
                        "success": False,
                        "message": f"Error deleting {file_path}: {e}",
                        "deleted_files": deleted_files
                    }
    
    # Commit volume changes
    try:
        volume.commit()
        print("✅ Volume changes committed successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not commit volume changes: {e}")
    
    # Summary
    if deleted_files:
        total_size_mb = sum(f["size_mb"] for f in deleted_files)
        print(f"\n📊 Summary:")
        print(f"   - Files deleted: {len(deleted_files)}")
        print(f"   - Total size freed: {total_size_mb:.2f} MB")
        return {
            "success": True,
            "message": f"Successfully deleted {len(deleted_files)} file(s)",
            "deleted_files": deleted_files,
            "total_size_mb": total_size_mb
        }
    else:
        print(f"\nℹ️  No ESM2-8M embedding files found to delete.")
        print(f"   Patterns searched: {', '.join(patterns_to_match)}")
        return {
            "success": True,
            "message": "No matching files found",
            "deleted_files": [],
            "patterns_searched": patterns_to_match
        }


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def list_esm2_8m_embeddings():
    """
    List ESM2-8M embedding files without deleting them.
    Useful for previewing what would be deleted.
    """
    embedding_dir = Path(EMBEDDING_DIR)
    
    if not embedding_dir.exists():
        print(f"❌ Embedding directory {EMBEDDING_DIR} does not exist on volume.")
        return {"success": False, "message": "Embedding directory not found", "files": []}
    
    patterns_to_match = [
        "ESM2-8M_*.pth",
        "ESM2-8_*.pth",
    ]
    
    found_files = []
    
    for pattern in patterns_to_match:
        matching_files = list(embedding_dir.glob(pattern))
        for file_path in matching_files:
            try:
                file_size = file_path.stat().st_size
                found_files.append({
                    "file": str(file_path),
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
    
    if found_files:
        total_size_mb = sum(f["size_mb"] for f in found_files)
        print(f"\n📋 Found {len(found_files)} ESM2-8M embedding file(s):")
        for f in found_files:
            print(f"   - {f['file']} ({f['size_mb']} MB)")
        print(f"\n   Total size: {total_size_mb:.2f} MB")
        return {
            "success": True,
            "files": found_files,
            "total_size_mb": total_size_mb
        }
    else:
        print(f"\nℹ️  No ESM2-8M embedding files found.")
        print(f"   Patterns searched: {', '.join(patterns_to_match)}")
        return {
            "success": True,
            "files": [],
            "patterns_searched": patterns_to_match
        }


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        # List files without deleting
        print("🔍 Listing ESM2-8M embedding files...")
        with app.run():
            result = list_esm2_8m_embeddings.remote()
            if result.get("success"):
                if result.get("files"):
                    print(f"\n✅ Found {len(result['files'])} file(s)")
                else:
                    print("\n✅ No matching files found")
            else:
                print(f"\n❌ Error: {result.get('message')}")
    else:
        # Delete files (default action)
        print("🗑️  Deleting ESM2-8M embedding files from Modal volume...")
        print(f"   Volume: {VOLUME_NAME}")
        print(f"   Directory: {EMBEDDING_DIR}")
        print()
        
        with app.run():
            result = delete_esm2_8m_embeddings.remote()
            if result.get("success"):
                print(f"\n✅ {result.get('message')}")
            else:
                print(f"\n❌ Error: {result.get('message')}")
                sys.exit(1)

