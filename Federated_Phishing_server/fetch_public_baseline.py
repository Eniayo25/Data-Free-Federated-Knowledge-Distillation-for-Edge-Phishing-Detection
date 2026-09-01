import pandas as pd
import requests
import zipfile
import io
import os

# Official public repository URL from the UCI Machine Learning Archive
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def main():
    print("=================================================================")
    print("[ Pipeline ] Accessing UCI Machine Learning Archive...")
    print("=================================================================")
    
    try:
        # Download raw public archive
        response = requests.get(UCI_URL, timeout=15)
        response.raise_for_status()
        
        # Decompress zip
        zip_archive = zipfile.ZipFile(io.BytesIO(response.content))
        
        
        with zip_archive.open("SMSSpamCollection") as file_stream:
            df = pd.read_csv(file_stream, sep="\t", names=["raw_label", "text"])
            
        print(f"[ Success ] Successfully pulled {len(df)} raw public data entries.")
        
        # Isolate targets 
        # Extract first 50 legitimate (ham) and 50 malicious (spam) strings
        benign_slice = df[df["raw_label"] == "ham"].head(50).copy()
        phishing_slice = df[df["raw_label"] == "spam"].head(50).copy()
        
        # Map labels (1 = Threat, 0 = Clean)
        benign_slice["label"] = 0
        phishing_slice["label"] = 1
        
        # Merge slices
        balanced_evaluation_set = pd.concat([benign_slice, phishing_slice], ignore_index=True)
        balanced_evaluation_set = balanced_evaluation_set[["text", "label"]]
        
        # Output local validation matrix
        output_path = "local_val.csv"
        balanced_evaluation_set.to_csv(output_path, index=False)
        
        print("\n" + "="*65)
        print(f" SCIENTIFIC BASELINE SECURED: '{output_path}' generated.")
        print("   Source: UCI SMS Spam Collection Repository (Almeida et al.)")
        print("   Composition: 50 Public Malicious Lures | 50 Public Legitimate Controls")
        print("="*65 + "\n")
        
    except Exception as e:
        print(f"[ Critical Failure ] Public data extraction pipeline aborted: {e}")

if __name__ == "__main__":
    main()