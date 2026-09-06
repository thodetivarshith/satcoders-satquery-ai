import os
import glob

def show_images(folder, title):
    print(f"\n--- {title} ---")
    if not os.path.exists(folder):
        print(f"Folder not found: {folder} -> skipping, creating...")
        os.makedirs(folder, exist_ok=True)
        return
    files = os.listdir(folder)
    print(f"Found {len(files)} files in {folder}")
    for f in files[:3]:
        print(f"  - {f}")

# Correct paths - 
base = os.path.dirname(os.path.dirname(__file__)) if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data")) else ".."
sar_folder = os.path.join(base, "data", "sar")
optical_folder = os.path.join(base, "data", "optical")
samples_folder = os.path.join(base, "samples")

# Check all possible
show_images(sar_folder, "SAR Satellite Images")
show_images(optical_folder, "Optical Images")
show_images(samples_folder, "Samples")
show_images("data/sar", "SAR (relative)")

print("\nDone - No crash!")