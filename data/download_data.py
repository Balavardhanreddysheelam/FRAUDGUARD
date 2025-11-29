import os
import subprocess
import sys

def install_kaggle():
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def download_datasets():
    print("Downloading datasets from Kaggle...")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    datasets = [
        "mlg-ulb/creditcardfraud",
        "ieee-fraud-detection"
    ]
    
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", "data", "--unzip"], check=True)
            print(f"Successfully downloaded {dataset}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {dataset}: {e}")
            print("Please ensure you have set up your Kaggle API credentials (kaggle.json).")
            print("Instructions: https://www.kaggle.com/docs/api")

if __name__ == "__main__":
    install_kaggle()
    download_datasets()
