import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


color_baseline = '#4C72B0'  # Steel Blue (Thapa et al.)
color_proposed = '#55A868'  # Muted Green (Proposed INT8 ONNX)
color_skew = '#C44E52'      # Muted Red (Non-IID Skew)

# ==============================================================================
# Graph 1: Balanced Baseline Performance (IID)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(6.2, 5.0), dpi=300)

categories_1 = ['F1-Score', 'Recall', 'Precision']
thapa_iid = [98.03, 97.06, 99.04]
proposed_iid = [93.07, 94.00, 92.16]

x1 = np.arange(len(categories_1))
w1 = 0.32

r1_1 = ax1.bar(x1 - w1/2, thapa_iid, w1, label='Thapa et al. (2023) [Global THEMIS, 5 Clients]', color=color_baseline)
r1_2 = ax1.bar(x1 + w1/2, proposed_iid, w1, label='Proposed Framework (INT8 ONNX) [Pi 5]', color=color_proposed)

ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('Balanced Baseline Performance (IID)', fontsize=12, fontweight='bold', pad=14)
ax1.set_xticks(x1)
ax1.set_xticklabels(categories_1, fontsize=10.5)
ax1.set_ylim(0, 125)
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

for rect in r1_1 + r1_2:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

fig1.tight_layout()
fig1.savefig('1_balanced_baseline_performance.png', dpi=300)
plt.close(fig1)

# ==============================================================================
# Graph 2: Non-IID Data Skew Impact on Recall
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(6.2, 5.0), dpi=300)

categories_2 = ['Thapa et al. (2023)\n[10/90 Asymmetric Skew]', 'Proposed Framework\n[Non-IID Stress Test]']
recall_iid = [97.1, 94.0]
recall_non_iid = [79.9, 56.0]

x2 = np.arange(len(categories_2))
w2 = 0.30

r2_1 = ax2.bar(x2 - w2/2, recall_iid, w2, label='Balanced IID Recall', color=color_baseline)
r2_2 = ax2.bar(x2 + w2/2, recall_non_iid, w2, label='Non-IID Skewed Recall', color=color_skew)

ax2.set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
ax2.set_title('Non-IID Data Skew Impact on Recall', fontsize=12, fontweight='bold', pad=14)
ax2.set_xticks(x2)
ax2.set_xticklabels(categories_2, fontsize=10)
ax2.set_ylim(0, 130)
ax2.legend(loc='upper left', frameon=True, fontsize=8.5)

for rect in r2_1 + r2_2:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')


ax2.annotate('−17.1% Recall\n(FNR = 20.1%)', 
             xy=(0 + w2/2, 85.0), xytext=(0 + w2/2 + 0.05, 108),
             arrowprops=dict(arrowstyle="->", color=color_skew, lw=1.2),
             fontsize=8.5, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec=color_skew, lw=1))

ax2.annotate('−38.0% Recall\n(22 False Negatives)', 
             xy=(1 + w2/2, 61.0), xytext=(1 + w2/2 + 0.05, 88),
             arrowprops=dict(arrowstyle="->", color=color_skew, lw=1.2),
             fontsize=8.5, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec=color_skew, lw=1))

fig2.tight_layout()
fig2.savefig('2_non_iid_recall_impact.png', dpi=300)
plt.close(fig2)

# ==============================================================================
# Graph 3: Model Footprint & Edge Deployment Viability
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(6.2, 5.0), dpi=300)

bars_3 = ['Thapa et al. (2023)\n(Uncompressed FP32 - Simulation)', 'Proposed Framework\n(Dynamic INT8 ONNX - Pi 5)']
sizes_3 = [268.00, 64.26]

r3_1 = ax3.bar(bars_3, sizes_3, width=0.45, color=[color_baseline, color_proposed])
ax3.set_ylabel('Model Footprint (MB)', fontsize=11, fontweight='bold')
ax3.set_title('Model Footprint & Edge Deployment Viability', fontsize=12, fontweight='bold', pad=14)
ax3.set_ylim(0, 360)

for rect in r3_1:
    h = rect.get_height()
    ax3.annotate(f'{h:.2f} MB', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')


ax3.annotate('76.0% Footprint Reduction\n(Executes natively in Pi 5 RAM)', 
             xy=(1, 74.0), xytext=(1, 185),
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
             fontsize=8.5, ha='center', bbox=dict(boxstyle="round,pad=0.35", fc="#EAEAF2", ec="#888888", lw=1))

fig3.tight_layout()
fig3.savefig('3_model_footprint_viability.png', dpi=300)
plt.close(fig3)

print("Saved all 3 individual figures successfully:")
print("1. 1_balanced_baseline_performance.png")
print("2. 2_non_iid_recall_impact.png")
print("3. 3_model_footprint_viability.png")