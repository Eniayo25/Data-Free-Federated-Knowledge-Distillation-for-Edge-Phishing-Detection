import os
import torch
import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_model(model_name, pth_filename):
    print(f"\n{"="*60}")
    print(f" Starting Optimization Pipeline for: {model_name}")
    print(f"{"="*60}")

    onnx_path = f"{model_name}.onnx"
    quant_path = f"{model_name}_quantized.onnx"

    if not os.path.exists(pth_filename):
        print(f" Error: Cannot find weights file '{pth_filename}'")
        return

    # Initialize structure and load custom weights
    print("[ Step 1 ] Loading PyTorch weights into DistilBERT framework...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(pth_filename, map_location="cpu"))
    model.eval()  

    # Generate standard dummy inputs for short text message tracing
    print("[ Step 2 ] Tracing active computational graph via dummy tokens...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dummy_text = "System alert confirmation scan payload."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]

    # Export the model 
    print("[ Step 3 ] Serializing to universal static ONNX format...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    
    # Verify exported base ONNX 
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f" Success: Base ONNX graph generated at '{onnx_path}' (~{os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")

    # Convert weights from FP32 to INT8 parameters
    print("[ Step 4 ] Applying Post-Training Quantization (PTQ to INT8)...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8
    )
    
    print(f" Optimization Complete for {model_name}!")
    print(f"   Original Weights Size: ~{os.path.getsize(pth_filename)/(1024*1024):.2f} MB")
    print(f"   Quantized ONNX Size:  ~{os.path.getsize(quant_path)/(1024*1024):.2f} MB")

def main():
    # Process both models same time
    optimize_model("iid_baseline", "iid_global_model.pth")
    optimize_model("non_iid_stress", "non_iid_global_model.pth")

if __name__ == "__main__":
    main()