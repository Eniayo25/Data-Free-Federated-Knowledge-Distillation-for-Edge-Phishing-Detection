# Data-Free Federated Knowledge Distillation for Edge Phishing Detection: Architecture, Quantisation, and ARM Hardware Benchmarking

This repository contains the complete implementation, communication protocols, dataset generation utilities, model checkpoints, and evaluation scripts for reproducing the three-tier Data-Free Federated Knowledge Distillation (DF-FKD) framework across a Windows 11 host workstation and a physical Raspberry Pi 5 single-board computer[cite: 1, 2].

All files are structured for direct execution from a mirrored root directory named `federated_phishing`[cite: 2].

---

## Repository File Manifest & Execution Map

| File Name | Environment | Chronological Role & Description |
| :--- | :--- | :--- |
| `requirements.txt` | Both | Pinned software dependency configuration (`flwr==1.8.0`, `torch==2.2.1`, `transformers==4.38.1`, `tokenizers==0.15.2`)[cite: 2]. |
| `network_communication.proto` | Both | Protocol Buffer schema defining the `ModelOrchestrator` service and `ModelWeights` binary structure[cite: 2]. |
| `network_communication_pb2.py` | Both | Generated Protocol Buffer Python stubs for binary tensor packing[cite: 2]. |
| `network_communication_pb2_grpc.py` | Both | Generated gRPC transport stubs managing client-server streaming sockets[cite: 2]. |
| `fetch_public_baseline.py` | Workstation | Automated ingestion script extracting the stratified 100-sample UCI SMS Spam Collection[cite: 1, 2]. |
| `local_val.csv` | Both | Stratified 100-sample evaluation baseline (50 safe, 50 phishing)[cite: 1, 2]. |
| `orchestrator_pipeline.py` | Workstation | GPT-4o curriculum generator with structural isolation tags and defensive data guards[cite: 1, 2]. |
| `IID_federated_server.py` | Workstation | Flower federated server streaming 10-sample balanced lesson packages for IID baseline runs[cite: 2]. |
| `federated_server.py` | Workstation | Flower federated server broadcasting 20-sample lesson packages for Non-IID stress test runs[cite: 2]. |
| `IID_virtual_client_A.py` | Workstation | Simulated client node A training on balanced lesson packages for IID baseline runs[cite: 2]. |
| `virtual_client_A.py` | Workstation | Simulated client node A executing client-side slicing for benign-only data (`safe_user`)[cite: 2]. |
| `IID_virtual_client_B.py` | Workstation | Simulated client node B training on balanced lesson packages for IID baseline runs[cite: 2]. |
| `virtual_client_B.py` | Workstation | Simulated client node B executing client-side slicing for a 50/50 mix (`corporate_mix`)[cite: 2]. |
| `IID_federated_client_Pi.py` | Pi 5 | Physical edge client running balanced IID training with single-thread and pacing guards[cite: 2]. |
| `federated_client.py` | Pi 5 | Physical edge client executing client-side slicing for aggressive phishing lures[cite: 2]. |
| `IID_export_pytorch_model.py` | Workstation | Deserialises `global_model_round_50.pkl` into PyTorch state dictionary `iid_global_model.pth`[cite: 2]. |
| `export_pytorch_model.py` | Workstation | Deserialises `non_iid_model_round_50.pkl` into PyTorch state dictionary `non_iid_global_model.pth`[cite: 2]. |
| `compile_and_quantize.py` | Workstation | Freezes ONNX computational graphs and executes dynamic INT8 Post-Training Quantisation[cite: 1, 2]. |
| `edge_benchmark.py` | Pi 5 | Measures inference latency with `time.perf_counter()` and logs Scikit-learn evaluation metrics[cite: 2]. |
| `iid_baseline_quantized.onnx` | Pi 5 | Compiled INT8 graph for the balanced IID model (64.26 MB)[cite: 1, 2]. |
| `non_iid_stress_quantized.onnx` | Pi 5 | Compiled INT8 graph for the skewed Non-IID model (64.26 MB)[cite: 1, 2]. |
| `iid_baseline_metrics.csv` | Workstation | 50-round convergence telemetry log for the IID baseline experiment[cite: 2]. |
| `non_iid_baseline_metrics.csv` | Workstation | 50-round convergence telemetry log for the Non-IID stress test experiment[cite: 2]. |
| `iid_quantized_predictions.csv` | Pi 5 | Raw classification outputs on the validation set from the IID model[cite: 2]. |
| `non_iid_quantized_predictions.csv`| Pi 5 | Raw classification outputs on the validation set from the Non-IID model[cite: 2]. |
| `generate_plots.py` | Workstation | Generates 300 DPI annotated confusion matrix heatmaps[cite: 2]. |
| `plot_real_history.py` | Workstation | Generates 50-round comparative federated convergence trajectory curves[cite: 2]. |

> **External Checkpoints (>100 MB):** Raw 50-round training checkpoints (`global_model_round_50.pkl`, `non_iid_model_round_50.pkl`, `iid_global_model.pth`, and `non_iid_global_model.pth`) are hosted on OneDrive due to file size limits: `[INSERT_YOUR_ONEDRIVE_LINK_HERE]`[cite: 2].

---

## Hardware and Network Prerequisites

* **Central Workstation (Tier 2):** Windows 11 PC, Intel Core i7, 16 GB RAM, dedicated GPU[cite: 1, 2]. Static IP: `192.168.1.10`[cite: 2].
* **Physical Edge Node (Tier 3):** Raspberry Pi 5 (ARM Cortex-A76, 2 GB RAM, 64-bit Raspberry Pi OS Debian Bookworm)[cite: 1, 2]. Static IP: `192.168.1.20`[cite: 2].
* **Network Infrastructure:** Dedicated dual-band router (2.4 GHz / 5.0 GHz) isolating federated socket traffic[cite: 1, 2].

---

## Step-by-Step Reproduction Guide

### Step 1: Environment Setup & Network Binding

#### 1.1 Windows Host Workstation
1. Open **PowerShell as Administrator** and configure the execution policy:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```
2. Navigate to your project root folder, initialise the virtual environment `venv`, and activate it[cite: 2]:
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install locked project dependencies[cite: 2]:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install openai pandas scikit-learn onnx onnxruntime matplotlib seaborn
   pip install numpy==1.26.4
   pip install "protobuf>=4.25.2,<5.0.0"
   ```

#### 1.2 Raspberry Pi 5
1. Log into the Raspberry Pi 5 terminal via SSH or Raspberry Pi Connect Remote Shell[cite: 2].
2. Set up the production virtual environment `target_env`[cite: 2]:
   ```bash
   cd ~/federated_phishing
   sudo apt-get update && sudo apt-get install -y python3-venv python3-pip tmux
   python3 -m venv target_env
   source target_env/bin/activate
   pip install --upgrade pip
   ```
3. Purge pip cache and install the CPU-only build of PyTorch alongside required libraries[cite: 2]:
   ```bash
   pip cache purge
   pip install torch==2.2.1 --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) --no-cache-dir
   pip install transformers==4.38.1 tokenizers==0.15.2 flwr==1.8.0 onnxruntime scikit-learn pandas --no-cache-dir
   pip install "protobuf>=4.25.2,<5.0.0"
   ```

#### 1.3 Bi-Directional Network Ping Check
Verify clean socket connectivity across the subnet before launching scripts[cite: 2]:
* **On Windows Workstation:** `ping 192.168.1.20`[cite: 2]
* **On Raspberry Pi 5:** `ping -c 4 192.168.1.10`[cite: 2]

---

### Step 2: Raspberry Pi 5 Hardware Stabilisation

Run these commands on the Raspberry Pi 5 to eliminate undervoltage shutdowns (`0x50000`), out-of-memory kernel panics, and GUI RAM overhead on standard 15W (5V/3A) adapters[cite: 1, 2]:

1. **Expand Virtual Swap Space to 2048 MB:**
   ```bash
   sudo dphys-swapfile swapoff
   sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   free -m
   ```
2. **Reclaim 1.2 GB RAM via Headless Console Mode:**
   ```bash
   sudo systemctl set-default multi-user.target
   sudo reboot
   ```
*(Note: Hardware execution limits, including single-core CPU capping via `torch.set_num_threads(1)` and batch pacing delays via `time.sleep(0.1)`, are already pre-programmed inside `IID_federated_client_Pi.py` and `federated_client.py`)[cite: 1, 2].*

---

### Step 3: Extract the Independent Validation Baseline

On the **Windows Workstation**, run the ingestion script to download the peer-reviewed UCI SMS Spam Collection and construct the stratified 100-sample validation dataset[cite: 1, 2]:
```powershell
python fetch_public_baseline.py
```
Transfer the generated `local_val.csv` to the Raspberry Pi 5[cite: 2]:
```powershell
scp local_val.csv eniayo@192.168.1.20:~/federated_phishing/
```

---

### Step 4: Configure OpenAI API Authentication (Tier-1 Cloud Teacher)

On the **Windows Workstation**, bind your developer credentials into the system environment memory variables[cite: 2]:

```powershell
# Set for current process session
$env:OPENAI_API_KEY="your_actual_private_api_key_here"

# Set persistently for user environment
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_actual_private_api_key_here", "User")
```

---

### Step 5: Execute 50-Round Federated Training Experiments

#### Experiment 1: Balanced (IID) Baseline Run
Streams uniform 10-sample lesson packages (5 benign / 5 phishing) across all participating client nodes[cite: 2].

1. **Workstation Terminal 1 (Flower Server):**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   .\venv\Scripts\Activate.ps1
   $env:OPENAI_API_KEY="your_actual_private_api_key_here"
   python IID_federated_server.py
   ```
2. **Workstation Terminal 2 (Virtual Client A):**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   .\venv\Scripts\Activate.ps1
   python IID_virtual_client_A.py
   ```
3. **Workstation Terminal 3 (Virtual Client B):**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   .\venv\Scripts\Activate.ps1
   python IID_virtual_client_B.py
   ```
4. **Raspberry Pi 5 Remote Shell (Physical Edge Client in `tmux`):**
   ```bash
   tmux new -s fed_session
   cd ~/federated_phishing
   source target_env/bin/activate
   python3 IID_federated_client_Pi.py
   ```
*Outputs Generated:* `global_model_round_50.pkl` and `iid_baseline_metrics.csv` (Total runtime: ~42,536 seconds / 11h 48m, converging at 90.0% accuracy)[cite: 1, 2].

---

#### Experiment 2: Skewed (Non-IID) Stress Test Run
Broadcasts 20-sample lesson packages where clients autonomously slice data locally according to their assigned profile handshake[cite: 1, 2]:
* **Virtual Client A (`safe_user`):** Slices indices 0–9 (100% benign)[cite: 2].
* **Virtual Client B (`corporate_mix`):** Slices indices 0–4 and 10–14 (50/50 corporate mix)[cite: 2].
* **Raspberry Pi 5 (`raspberry_pi_5`):** Slices indices 10–19 (100% aggressive phishing)[cite: 2].

1. **Workstation Terminal 1 (Non-IID Server):**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   .\venv\Scripts\Activate.ps1
   $env:OPENAI_API_KEY="your_actual_private_api_key_here"
   python federated_server.py
   ```
2. **Workstation Terminal 2 (Non-IID Virtual Client A):**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   .\venv\Scripts\Activate.ps1
   python virtual_client_A.py
   ```
3. **Workstation Terminal 3 (Non-IID Virtual Client B):**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   .\venv\Scripts\Activate.ps1
   python virtual_client_B.py
   ```
4. **Raspberry Pi 5 Remote Shell (Non-IID Physical Client in `tmux`):**
   ```bash
   tmux a -t fed_session
   cd ~/federated_phishing
   source target_env/bin/activate
   python3 federated_client.py
   ```
*Outputs Generated:* `non_iid_model_round_50.pkl` and `non_iid_baseline_metrics.csv` (Total runtime: ~47,519 seconds / 13h 11m, consolidating at 77.0% accuracy)[cite: 1, 2].

---

### Step 6: Model Checkpoint Conversion & INT8 Dynamic Quantisation

On the **Windows Workstation**, map the unpickled parameters into PyTorch state dictionaries and compile them into INT8 ONNX computation graphs[cite: 1, 2]:

1. **Convert Checkpoint Pickle Files to PyTorch State Dictionaries:**
   ```powershell
   python IID_export_pytorch_model.py
   python export_pytorch_model.py
   ```
   *Outputs:* `iid_global_model.pth` and `non_iid_global_model.pth` (~261.58 MB each)[cite: 2].
2. **Execute Graph Tracing & INT8 Dynamic Quantisation:**
   ```powershell
   python compile_and_quantize.py
   ```
   *Outputs:* `iid_baseline_quantized.onnx` and `non_iid_stress_quantized.onnx` (Footprint reduced by 74.84% down to 64.26 MB each)[cite: 1, 2].

---

### Step 7: Native Edge Benchmarking on Raspberry Pi 5

1. **Transfer Quantised ONNX Models to Raspberry Pi 5:**
   From Windows PowerShell, push the compiled graphs across the network[cite: 2]:
   ```powershell
   scp iid_baseline_quantized.onnx eniayo@192.168.1.20:~/federated_phishing/
   scp non_iid_stress_quantized.onnx eniayo@192.168.1.20:~/federated_phishing/
   ```
2. **Run Inference Speed and Telemetry Evaluation on Pi 5:**
   On the **Raspberry Pi 5** within `target_env`, execute `edge_benchmark.py` to evaluate the 100-sample validation set via `time.perf_counter()` on the ARM processor[cite: 2]:
   ```bash
   cd ~/federated_phishing
   source target_env/bin/activate
   python3 edge_benchmark.py
   ```
   *Outputs Generated:* `iid_quantized_predictions.csv` and `non_iid_quantized_predictions.csv`[cite: 2].

---

### Step 8: Telemetry Retrieval & Visual Figure Generation

1. **Pull Prediction CSVs to Windows Workstation:**
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   scp eniayo@192.168.1.20:~/federated_phishing/*_predictions.csv .
   ```
2. **Generate 300 DPI Publication-Ready Figures:**
   ```powershell
   python generate_plots.py
   python plot_real_history.py
   ```
   *Generated Outputs in Project Folder:*
   * `confusion_matrix_iid_2.png`: 300 DPI annotated heatmap displaying 46 TN, 4 FP, 3 FN, 47 TP[cite: 1, 2].
   * `confusion_matrix_non_iid_2.png`: 300 DPI annotated heatmap displaying 49 TN, 1 FP, 22 FN, 28 TP[cite: 1, 2].
   * `actual_federated_convergence.jpg`: 50-round convergence trajectories comparing balanced training against Non-IID client drift[cite: 1, 2].

---

## Empirical Benchmark Reference (Raspberry Pi 5)

| Metric / Parameter | Balanced IID (Quantised ONNX) | Skewed Non-IID (Quantised ONNX) | Operational Status |
| :--- | :--- | :--- | :--- |
| **Model Storage Footprint** | 64.26 MB | 64.26 MB | 74.84% Footprint Compression[cite: 1, 2] |
| **Classification Accuracy** | 93.00% | 77.00% | Target (>90.0%) Achieved under IID[cite: 1, 2] |
| **Precision** | 0.9216 | 0.9655 | 1 False Alarm under Non-IID[cite: 1, 2] |
| **Recall** | 0.9400 | 0.5600 | 22 False Negatives under Non-IID[cite: 1, 2] |
| **F1-Score** | 0.9307 | 0.7089 | Target (>0.85) Achieved under IID[cite: 1, 2] |
| **Mean Inference Latency** | 50.31 ms | 50.43 ms | Real-Time ARM Execution[cite: 1, 2] |
