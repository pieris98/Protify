"""
Script to delete embedding files for any specified model from Modal volume.

This script connects to the Modal volume used by protify_modal_app.py
and deletes embedding files for the specified model.

Embedding files are stored as: {model_name}_{matrix_embed}.pth
where matrix_embed is typically 'False' or 'True' (boolean string).

Usage:
    # Delete embeddings for a specific model
    modal run delete_embeddings.py --model ESM2-8
    
    # List embeddings for a specific model (preview before deleting)
    modal run delete_embeddings.py --model ESM2-8 --list
    
    # List all embeddings (no model filter)
    modal run delete_embeddings.py --list-all
"""

import modal
import argparse
from pathlib import Path
from typing import Optional

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
def delete_model_embeddings(model_name: str):
    """
    Delete embedding files for the specified model from the Modal volume.
    
    Args:
        model_name: The model name (e.g., "ESM2-8", "ESM2-35", "ProtBert", etc.)
    
    Searches for files matching: {model_name}_*.pth
    """
    if not model_name or not model_name.strip():
        return {
            "success": False,
            "message": "Model name cannot be empty",
            "deleted_files": []
        }
    
    embedding_dir = Path(EMBEDDING_DIR)
    
    if not embedding_dir.exists():
        print(f"❌ Embedding directory {EMBEDDING_DIR} does not exist on volume.")
        return {"success": False, "message": "Embedding directory not found", "deleted_files": []}
    
    # Pattern to match embedding files for the specified model
    # Format: {model_name}_{matrix_embed}.pth
    pattern_to_match = f"{model_name}_*.pth"
    
    deleted_files = []
    
    # Search for files matching the pattern
    matching_files = list(embedding_dir.glob(pattern_to_match))
    
    if not matching_files:
        print(f"ℹ️  No files found matching pattern: {pattern_to_match}")
        return {
            "success": True,
            "message": f"No embedding files found for model '{model_name}'",
            "deleted_files": [],
            "pattern_searched": pattern_to_match
        }
    
    # Delete matching files
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
    total_size_mb = sum(f["size_mb"] for f in deleted_files)
    print(f"\n📊 Summary:")
    print(f"   - Model: {model_name}")
    print(f"   - Files deleted: {len(deleted_files)}")
    print(f"   - Total size freed: {total_size_mb:.2f} MB")
    return {
        "success": True,
        "message": f"Successfully deleted {len(deleted_files)} file(s) for model '{model_name}'",
        "deleted_files": deleted_files,
        "total_size_mb": total_size_mb,
        "model_name": model_name
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def list_model_embeddings(model_name: Optional[str] = None):
    """
    List embedding files for the specified model (or all models if None) without deleting them.
    
    Args:
        model_name: The model name to filter by. If None, lists all embedding files.
    
    Useful for previewing what would be deleted.
    """
    embedding_dir = Path(EMBEDDING_DIR)
    
    if not embedding_dir.exists():
        print(f"❌ Embedding directory {EMBEDDING_DIR} does not exist on volume.")
        return {"success": False, "message": "Embedding directory not found", "files": []}
    
    if model_name:
        # Filter by specific model
        pattern_to_match = f"{model_name}_*.pth"
        matching_files = list(embedding_dir.glob(pattern_to_match))
        patterns_searched = [pattern_to_match]
    else:
        # List all .pth files in the embedding directory
        matching_files = list(embedding_dir.glob("*.pth"))
        patterns_searched = ["*.pth"]
    
    found_files = []
    
    for file_path in matching_files:
        try:
            file_size = file_path.stat().st_size
            # Extract model name from filename (format: {model_name}_{matrix_embed}.pth)
            filename = file_path.name
            parts = filename.rsplit("_", 1)
            extracted_model = parts[0] if len(parts) > 1 else filename.replace(".pth", "")
            
            found_files.append({
                "file": str(file_path),
                "model": extracted_model,
                "filename": filename,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            })
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
    
    if found_files:
        total_size_mb = sum(f["size_mb"] for f in found_files)
        
        if model_name:
            print(f"\n📋 Found {len(found_files)} embedding file(s) for model '{model_name}':")
        else:
            print(f"\n📋 Found {len(found_files)} embedding file(s) (all models):")
        
        # Group by model if listing all
        if not model_name:
            files_by_model = {}
            for f in found_files:
                model = f["model"]
                if model not in files_by_model:
                    files_by_model[model] = []
                files_by_model[model].append(f)
            
            for model, model_files in sorted(files_by_model.items()):
                model_total_mb = sum(f["size_mb"] for f in model_files)
                print(f"\n   Model: {model} ({len(model_files)} file(s), {model_total_mb:.2f} MB)")
                for f in model_files:
                    print(f"      - {f['filename']} ({f['size_mb']} MB)")
        else:
            for f in found_files:
                print(f"   - {f['filename']} ({f['size_mb']} MB)")
        
        print(f"\n   Total size: {total_size_mb:.2f} MB")
        return {
            "success": True,
            "files": found_files,
            "total_size_mb": total_size_mb,
            "model_name": model_name
        }
    else:
        if model_name:
            print(f"\nℹ️  No embedding files found for model '{model_name}'.")
        else:
            print(f"\nℹ️  No embedding files found.")
        print(f"   Patterns searched: {', '.join(patterns_searched)}")
        return {
            "success": True,
            "files": [],
            "patterns_searched": patterns_searched,
            "model_name": model_name
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete or list embedding files for a specified model from Modal volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete embeddings for ESM2-8
  modal run delete_embeddings.py --model ESM2-8
    
  # Preview embeddings for ESM2-8 before deleting
  modal run delete_embeddings.py --model ESM2-8 --list
  
  # List all embeddings (all models)
  modal run delete_embeddings.py --list-all
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name whose embeddings should be deleted (e.g., 'ESM2-8', 'ESM2-35', 'ProtBert')"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List embeddings for the specified model without deleting"
    )
    
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all embeddings for all models without deleting"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.list_all:
        # List all embeddings
        print("🔍 Listing all embedding files...")
        with app.run():
            result = list_model_embeddings.remote(None)
            if result.get("success"):
                if result.get("files"):
                    print(f"\n✅ Found {len(result['files'])} file(s)")
                else:
                    print("\n✅ No embedding files found")
            else:
                print(f"\n❌ Error: {result.get('message')}")
    elif args.list:
        # List embeddings for specific model
        if not args.model:
            print("❌ Error: --model is required when using --list")
            parser.print_help()
            exit(1)
        
        print(f"🔍 Listing embedding files for model '{args.model}'...")
        with app.run():
            result = list_model_embeddings.remote(args.model)
            if result.get("success"):
                if result.get("files"):
                    print(f"\n✅ Found {len(result['files'])} file(s)")
                else:
                    print(f"\n✅ No embedding files found for model '{args.model}'")
            else:
                print(f"\n❌ Error: {result.get('message')}")
    else:
        # Delete embeddings for specific model
        if not args.model:
            print("❌ Error: --model is required for deletion")
            parser.print_help()
            exit(1)
        
        print(f"🗑️  Deleting embedding files for model '{args.model}' from Modal volume...")
        print(f"   Volume: {VOLUME_NAME}")
        print(f"   Directory: {EMBEDDING_DIR}")
        print()
        
        with app.run():
            result = delete_model_embeddings.remote(args.model)
            if result.get("success"):
                print(f"\n✅ {result.get('message')}")
            else:
                print(f"\n❌ Error: {result.get('message')}")
                exit(1)
