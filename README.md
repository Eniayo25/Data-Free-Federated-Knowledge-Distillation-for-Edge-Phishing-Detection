# Data-Free Federated Knowledge Distillation for Edge Phishing Detection: Architecture, Quantisation, and ARM Hardware Benchmarking

This repository contains the complete implementation, communication protocols, dataset generation utilities, model checkpoints, and evaluation scripts for reproducing the three-tier Data-Free Federated Knowledge Distillation (DF-FKD) framework across a Windows 11 host workstation and a physical Raspberry Pi 5 single-board computer.

All files are structured for direct execution from a mirrored root directory named `federated_phishing`.

---

## Repository File Manifest & Execution Map

| File Name | Environment | Chronological Role & Description |
| :--- | :--- | :--- |
| `requirements.txt` | Both | Pinned environment package dependencies (`flwr`, `torch`, `transformers`). |
| `network_communication.proto` | Both | Protocol Buffer schema for gRPC service and message contracts. |
| `network_communication_pb2.py` | Both | Generated Python stubs for binary tensor serialisation. |
| `network_communication_pb2_grpc.py` | Both | Generated Python stubs for gRPC streaming transport. |
| `fetch_public_baseline.py` | Workstation | Downloads and extracts the 100-sample UCI SMS validation set. |
| `local_val.csv` | Both | Stratified 100-sample validation dataset (50 safe, 50 phishing). |
| `orchestrator_pipeline.py` | Workstation | GPT-4o generator with prompt guards and in-memory tokenisation. |
| `IID_federated_server.py` | Workstation | Flower server for balanced 10-sample IID training rounds. |
| `IID_virtual_client_A.py` | Workstation | Virtual client A training on balanced IID payloads. |
| `IID_virtual_client_B.py` | Workstation | Virtual client B training on balanced IID payloads. |
| `IID_federated_client_Pi.py` | Pi 5 | Pi 5 client for balanced IID training with hardware shields. |
| `global_model_round_50.pkl` | Workstation | Final round 50 aggregated model checkpoint (IID). |
| `IID_export_pytorch_model.py` | Workstation | Converts `global_model_round_50.pkl` into `.pth` format. |
| `iid_global_model.pth` | Workstation | PyTorch state dictionary for the IID baseline model. |
| `federated_server.py` | Workstation | Flower server broadcasting 20-sample Non-IID packages. |
| `virtual_client_A.py` | Workstation | Virtual client A slicing benign-only data (`safe_user`). |
| `virtual_client_B.py` | Workstation | Virtual client B slicing a 50/50 mix (`corporate_mix`). |
| `federated_client.py` | Pi 5 | Pi 5 client slicing phishing lures (`high_risk_target`). |
| `non_iid_model_round_50.pkl` | Workstation | Final round 50 aggregated model checkpoint (Non-IID). |
| `export_pytorch_model.py` | Workstation | Converts `non_iid_model_round_50.pkl` into `.pth` format. |
| `non_iid_global_model.pth` | Workstation | PyTorch state dictionary for the Non-IID model. |
| `compile_and_quantize.py` | Workstation | Traces and compiles `.pth` models into INT8 ONNX graphs. |
| `iid_baseline_quantized.onnx` | Pi 5 | INT8 quantised ONNX model for IID baseline (64.26 MB). |
| `non_iid_stress_quantized.onnx` | Pi 5 | INT8 quantised ONNX model for Non-IID stress test (64.26 MB). |
| `edge_benchmark.py` | Pi 5 | Measures inference latency and classification metrics on Pi 5. |
| `iid_baseline_metrics.csv` | Workstation | 50-round convergence telemetry log for the IID run. |
| `non_iid_baseline_metrics.csv` | Workstation | 50-round convergence telemetry log for the Non-IID run. |
| `iid_quantized_predictions.csv` | Pi 5 | Edge classification predictions from the IID model. |
| `non_iid_quantized_predictions.csv`| Pi 5 | Edge classification predictions from the Non-IID model. |
| `generate_plots.py` | Workstation | Generates 300 DPI annotated confusion matrix heatmaps. |
| `plot_real_history.py` | Workstation | Plots 50-round comparative federated convergence curves. |
---

## Hardware and Network Prerequisites

* **Central Workstation (Tier 2):** Windows 11 PC, Intel Core i7, 16 GB RAM, dedicated GPU. Static IP: `192.168.1.10`.
* **Physical Edge Node (Tier 3):** Raspberry Pi 5 (ARM Cortex-A76, 2 GB RAM, 64-bit Raspberry Pi OS Debian Bookworm). Static IP: `192.168.1.20`.
* **Network Infrastructure:** Dedicated dual-band router (2.4 GHz / 5.0 GHz) isolating federated socket traffic.

---

## Step-by-Step Reproduction Guide

### Step 1: Environment Setup & Network Binding

#### 1.1 Windows Host Workstation
1. Open **PowerShell as Administrator** and configure the execution policy:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```
2. Navigate to your project root folder, initialise the virtual environment `venv`, and activate it:
   ```powershell
   cd C:\Users\eniayo25\federated_phishing
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install locked project dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install openai pandas scikit-learn onnx onnxruntime matplotlib seaborn
   pip install numpy==1.26.4
   pip install "protobuf>=4.25.2,<5.0.0"
   ```

#### 1.2 Raspberry Pi 5
1. Log into the Raspberry Pi 5 terminal via SSH or Raspberry Pi Connect Remote Shell.
2. Set up the production virtual environment `target_env`:
   ```bash
   cd ~/federated_phishing
   sudo apt-get update && sudo apt-get install -y python3-venv python3-pip tmux
   python3 -m venv target_env
   source target_env/bin/activate
   pip install --upgrade pip
   ```
3. Purge pip cache and install the CPU-only build of PyTorch alongside required libraries:
   ```bash
   pip cache purge
   pip install torch==2.2.1 --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) --no-cache-dir
   pip install transformers==4.38.1 tokenizers==0.15.2 flwr==1.8.0 onnxruntime scikit-learn pandas --no-cache-dir
   pip install "protobuf>=4.25.2,<5.0.0"
   ```

#### 1.3 Bi-Directional Network Ping Check
Verify clean socket connectivity across the subnet before launching scripts:
* **On Windows Workstation:** `ping 192.168.1.20`
* **On Raspberry Pi 5:** `ping -c 4 192.168.1.10`

---

### Step 2: Raspberry Pi 5 Hardware Stabilisation

Run these commands on the Raspberry Pi 5 to eliminate undervoltage shutdowns (`0x50000`), out-of-memory kernel panics, and GUI RAM overhead on standard 15W (5V/3A) adapters:

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
*(Note: Hardware execution limits, including single-core CPU capping via `torch.set_num_threads(1)` and batch pacing delays via `time.sleep(0.1)`, are already pre-programmed inside `IID_federated_client_Pi.py` and `federated_client.py`).*

---

### Step 3: Extract the Independent Validation Baseline

On the **Windows Workstation**, run the ingestion script to download the peer-reviewed UCI SMS Spam Collection and construct the stratified 100-sample validation dataset:
```powershell
python fetch_public_baseline.py
```
Transfer the generated `local_val.csv` to the Raspberry Pi 5:
```powershell
scp local_val.csv eniayo@192.168.1.20:~/federated_phishing/
```

---

### Step 4: Configure OpenAI API Authentication (Tier-1 Cloud Teacher)

On the **Windows Workstation**, bind your developer credentials into the system environment memory variables:

```powershell
# Set for current process session
$env:OPENAI_API_KEY="your_actual_private_api_key_here"

# Set persistently for user environment
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_actual_private_api_key_here", "User")
```

---

### Step 5: Execute 50-Round Federated Training Experiments

#### Experiment 1: Balanced (IID) Baseline Run
Streams uniform 10-sample lesson packages (5 benign / 5 phishing) across all participating client nodes.

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
*Outputs Generated:* `global_model_round_50.pkl` and `iid_baseline_metrics.csv` (Total runtime: ~42,536 seconds / 11h 48m, converging at 90.0% accuracy).

---

#### Experiment 2: Skewed (Non-IID) Stress Test Run
Broadcasts 20-sample lesson packages where clients autonomously slice data locally according to their assigned profile handshake:
* **Virtual Client A (`safe_user`):** Slices indices 0–9 (100% benign).
* **Virtual Client B (`corporate_mix`):** Slices indices 0–4 and 10–14 (50/50 corporate mix).
* **Raspberry Pi 5 (`raspberry_pi_5`):** Slices indices 10–19 (100% aggressive phishing).

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
*Outputs Generated:* `non_iid_model_round_50.pkl` and `non_iid_baseline_metrics.csv` (Total runtime: ~47,519 seconds / 13h 11m, consolidating at 77.0% accuracy).

---

### Step 6: Model Checkpoint Conversion & INT8 Dynamic Quantisation

On the **Windows Workstation**, map the unpickled parameters into PyTorch state dictionaries and compile them into INT8 ONNX computation graphs:

1. **Convert Checkpoint Pickle Files to PyTorch State Dictionaries:**
   ```powershell
   python IID_export_pytorch_model.py
   python export_pytorch_model.py
   ```
   *Outputs:* `iid_global_model.pth` and `non_iid_global_model.pth` (~261.58 MB each).
2. **Execute Graph Tracing & INT8 Dynamic Quantisation:**
   ```powershell
   python compile_and_quantize.py
   ```
   *Outputs:* `iid_baseline_quantized.onnx` and `non_iid_stress_quantized.onnx` (Footprint reduced by 74.84% down to 64.26 MB each).

---

### Step 7: Native Edge Benchmarking on Raspberry Pi 5

1. **Transfer Quantised ONNX Models to Raspberry Pi 5:**
   From Windows PowerShell, push the compiled graphs across the network:
   ```powershell
   scp iid_baseline_quantized.onnx eniayo@192.168.1.20:~/federated_phishing/
   scp non_iid_stress_quantized.onnx eniayo@192.168.1.20:~/federated_phishing/
   ```
2. **Run Inference Speed and Telemetry Evaluation on Pi 5:**
   On the **Raspberry Pi 5** within `target_env`, execute `edge_benchmark.py` to evaluate the 100-sample validation set via `time.perf_counter()` on the ARM processor:
   ```bash
   cd ~/federated_phishing
   source target_env/bin/activate
   python3 edge_benchmark.py
   ```
   *Outputs Generated:* `iid_quantized_predictions.csv` and `non_iid_quantized_predictions.csv`.

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
   * `confusion_matrix_iid_2.png`: 300 DPI annotated heatmap displaying 46 TN, 4 FP, 3 FN, 47 TP.
   * `confusion_matrix_non_iid_2.png`: 300 DPI annotated heatmap displaying 49 TN, 1 FP, 22 FN, 28 TP.
   * `actual_federated_convergence.jpg`: 50-round convergence trajectories comparing balanced training against Non-IID client drift.

---

## Empirical Benchmark Reference (Raspberry Pi 5)

| Metric / Parameter | Balanced IID (Quantised ONNX) | Skewed Non-IID (Quantised ONNX) | Operational Status |
| :--- | :--- | :--- | :--- |
| **Model Storage Footprint** | 64.26 MB | 64.26 MB | 74.84% Footprint Compression |
| **Classification Accuracy** | 93.00% | 77.00% | Target (>90.0%) Achieved under IID|
| **Precision** | 0.9216 | 0.9655 | 1 False Alarm under Non-IID |
| **Recall** | 0.9400 | 0.5600 | 22 False Negatives under Non-IID|
| **F1-Score** | 0.9307 | 0.7089 | Target (>0.85) Achieved under IID|
| **Mean Inference Latency** | 50.31 ms | 50.43 ms | Real-Time ARM Execution |
| **P95 Latency** | 50.82 ms | 51.65 ms | Stable Execution Without Jitter |
