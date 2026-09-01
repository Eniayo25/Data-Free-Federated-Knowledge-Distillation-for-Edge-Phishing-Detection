import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_and_save_matrix(csv_path, output_image_name, title_label):
    
    if not os.path.exists(csv_path):
        print(f" Skipping {csv_path} - file not found in this directory.")
        return
        
    # Read data arrays captured by Pi
    df = pd.read_csv(csv_path)
    
    # Compute confusion array
    cm = confusion_matrix(df['true_label'], df['predicted_label'])
    
    
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    
   
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=['Safe (0)', 'Phishing (1)'],
        yticklabels=['Safe (0)', 'Phishing (1)'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    
    plt.title(f"Confusion Matrix: {title_label}", fontsize=14, pad=15, weight='bold')
    plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
    plt.ylabel("True Labels", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    
    plt.savefig(output_image_name, dpi=300)
    plt.close()
    print(f" Visual plot successfully saved to: '{output_image_name}'")

if __name__ == "__main__":
    print(" Initializing Chapter 4 Figure Generation...")
    generate_and_save_matrix("iid_quantized_predictions.csv", "confusion_matrix_iid.png", "IID Quantized Baseline")
    generate_and_save_matrix("non_iid_quantized_predictions.csv", "confusion_matrix_non_iid.png", "Non-IID Quantized Stress Test")
    print(" All figures exported successfully!")