import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_real_convergence_plot():
    print(" Ingesting raw CSV logs to compile federated convergence history...")
    
    # Define file paths
    iid_path = "iid_baseline_metrics.csv"
    non_iid_path = "non_iid_baseline_metrics.csv"
    output_image = "actual_federated_convergence.png"
    
    # Error checking
    if not os.path.exists(iid_path) or not os.path.exists(non_iid_path):
        print(" Error: Missing one or both metric files in this directory.")
        return
        
    
    iid_df = pd.read_csv(iid_path)
    non_iid_df = pd.read_csv(non_iid_path)
    
    
    iid_df['Accuracy_Pct'] = iid_df['Accuracy'] * 100
    non_iid_df['Accuracy_Pct'] = non_iid_df['Accuracy'] * 100

   
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    
    plt.plot(
        iid_df['Round'], iid_df['Accuracy_Pct'], 
        label='IID Baseline', color='#1f77b4', 
        linewidth=2.5, marker='o', markersize=4
    )
    plt.plot(
        non_iid_df['Round'], non_iid_df['Accuracy_Pct'], 
        label='Non-IID Stress Test', color='#d62728', 
        linewidth=2.5, marker='s', markersize=4
    )
    
    
    final_iid_acc = iid_df['Accuracy_Pct'].iloc[-1]
    final_non_iid_acc = non_iid_df['Accuracy_Pct'].iloc[-1]
    

    plt.annotate(
        f'IID Final: {final_iid_acc:.1f}%', xy=(50, final_iid_acc), xytext=(38, final_iid_acc + 4),
        arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1, headwidth=6)
    )
    plt.annotate(
        f'Non-IID Final: {final_non_iid_acc:.1f}%', xy=(50, final_non_iid_acc), xytext=(36, final_non_iid_acc - 9),
        arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1, headwidth=6)
    )


    plt.title("Actual Global Model Convergence Across 50 Federated Rounds", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Federated Learning Rounds", fontsize=12, labelpad=10)
    plt.ylabel("Global Validation Accuracy (%)", fontsize=12, labelpad=10)
    
    plt.xlim(0, 52)
    plt.ylim(45, 100)
    plt.xticks(range(0, 51, 5))
    
    plt.legend(title="Network Environment", title_fontsize='11', loc="lower right", frameon=True)
    plt.tight_layout()
    

    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f" Success! Your real training chart has been saved to: '{output_image}'")

if __name__ == "__main__":
    generate_real_convergence_plot()