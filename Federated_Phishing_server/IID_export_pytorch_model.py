import pickle
import torch
import os
from transformers import AutoModelForSequenceClassification

def main():
    pkl_path = "global_model_round_50.pkl"
    output_path = "iid_global_model.pth"
    
    if not os.path.exists(pkl_path):
        print(f"[ Error ] Could not find '{pkl_path}' in the current directory.")
        return

    print(f"[ Loader ] Reading federated weights from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        parameters = pickle.load(f)

    # Initialize base, untrained DistilBERT student model
    print("[ Model ] Initializing DistilBERT structure...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    
    print("[ Converter ] Mapping weights to PyTorch tensor format...")
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    
    # Load mapped weights into the model structure
    model.load_state_dict(state_dict, strict=True)

  
    torch.save(model.state_dict(), output_path)
    print("=" * 65)
    print(f" SUCCESS: Baseline weights converted and saved to: '{output_path}'")
    print("=" * 65)

if __name__ == "__main__":
    main()