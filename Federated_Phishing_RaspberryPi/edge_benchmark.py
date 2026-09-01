import os
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_edge_benchmark(model_path, model_label, csv_path="local_val.csv"):
    print("\n" + "="*60)
    print(f"  Launching Compliant Edge Benchmark for: {model_label}")
    print(f"  Model File: {model_path}")
    print("="*60)
    
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        print(" Error: Missing vital model or validation assets.")
        return

    # Initialize ONNX Runtime
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    df = pd.read_csv(csv_path)
    
    latencies = []
    true_labels = []
    predicted_labels = []
    
    print("[ Step 1 ] Executing low-latency serial inference loops...")
    for idx, row in df.iterrows():
        text = str(row['text'])
        true_labels.append(int(row['label']))
        
        # Tokenize using high speed pipeline
        inputs = tokenizer(text, return_tensors="np", max_length=128, padding="max_length", truncation=True)
        input_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # High precision timing
        start_time = time.perf_counter()
        outputs = session.run(["logits"], input_feed)
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)
        
        logits = outputs[0]
        predicted_labels.append(int(np.argmax(logits, axis=1)[0]))
            
    
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, zero_division=0)
    rec = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print("\n  COMPLIANT TELEMETRY RESULTS:")
    print(f"   Accuracy    : {acc * 100:.2f}%")
    print(f"   Precision   : {prec:.4f}")
    print(f"   Recall      : {rec:.4f}")
    print(f"   F1-Score    : {f1:.4f}  ")
    print(f"   Mean Latency: {mean_latency:.2f} ms")
    print(f"   P95 Latency : {p95_latency:.2f} ms")
    print("\n  RAW CONFUSION MATRIX ARRAY:")
    print(cm)
    print("="*60)
    
    
    export_df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predicted_labels
    })
    export_df.to_csv(f"{model_label}_predictions.csv", index=False)
    print(f" Saved prediction profile to: '{model_label}_predictions.csv'\n")

def main():
    
    run_edge_benchmark("iid_baseline_quantized.onnx", "iid_quantized")
    run_edge_benchmark("non_iid_stress_quantized.onnx", "non_iid_quantized")

if __name__ == "__main__":
    main()
