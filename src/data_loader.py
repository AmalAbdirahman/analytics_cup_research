import pandas as pd
import os
import zipfile
import glob

# MATCH CONFIG
MATCH_ID = 1886347


def load_match_data():
    print(f"📥 Loading Match {MATCH_ID}...")

    # --- 1. SMART PATH FINDER (Fixes the "src" vs "root" confusion) ---
    # Find where THIS script is currently sitting
    current_script_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_script_path)

    # If we are in 'src', go up one level to find 'data'
    if os.path.basename(current_folder) == 'src':
        project_root = os.path.dirname(current_folder)
    else:
        project_root = current_folder

    # Build the path to the specific match folder
    data_dir = os.path.join(project_root, 'data', 'matches', str(MATCH_ID))
    print(f"   🔍 Searching in: {data_dir}")

    # --- 2. FIND THE DATA (The "Hunter-Killer" Logic) ---
    # Look for ANY file ending in .jsonl or .zip
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))

    target_file = None

    if jsonl_files:
        # Scenario A: The raw file is already there (Local Dev)
        target_file = jsonl_files[0]
        print(f"   ✅ Found local file: {os.path.basename(target_file)}")

    elif zip_files:
        # Scenario B: Only the Zip exists (Judges / GitHub Repo)
        target_zip = zip_files[0]
        print(f"   📦 Found Zip file: {os.path.basename(target_zip)}")
        print("   ⚙️ Extracting data...")

        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Verify extraction worked
        jsonl_files_new = glob.glob(os.path.join(data_dir, "*.jsonl"))
        if not jsonl_files_new:
            raise FileNotFoundError("CRITICAL: Zip file was empty!")
        target_file = jsonl_files_new[0]

    else:
        # Scenario C: Critical Failure (Folder is empty)
        print("\n❌ DATA NOT FOUND.")
        print(f"   I looked in: {data_dir}")
        print("   -> Make sure you uploaded the 'data' folder to GitHub!")
        raise FileNotFoundError("No .jsonl or .zip files found.")

    # --- 3. LOAD ---
    print(f"   📂 Reading data...")
    return pd.read_json(target_file, lines=True)