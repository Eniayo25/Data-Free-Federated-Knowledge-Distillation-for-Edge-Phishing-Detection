import flwr as fl
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[ Client ] Using computing device: {device}")
print("[ Client ] Loading DistilBERT student model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class LessonsDataset(Dataset):
    def __init__(self, lessons):
        self.lessons = lessons

    def __len__(self):
        return len(self.lessons)

    def __getitem__(self, idx):
        item = self.lessons[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

# Dataset loader for local CSV evaluation
class CSVValidationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.DataFrame(columns=["text", "label"])
        if os.path.exists(csv_path):
            self.data = pd.read_csv(csv_path)
            print(f"[ Data Loader ] Loaded {len(self.data)} real validation samples from {csv_path}.")
        else:
            print("[ Warning ] local_val.csv not found! Generating temporary fallback validation set.")
            self.data = pd.DataFrame({
                "text": [
                    "Dear user, please verify your account at this insecure link.",
                    "Let's finalize the meeting schedule for tomorrow morning."
                ],
                "label": [1, 0]
            })
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


val_dataset = CSVValidationDataset("local_val.csv", tokenizer)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

class PhishingDetectionClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        print("\n[ gRPC Link ] Inbound weights received. Loading parameters into layers...")
        
        # Sync local weights with orchestrator
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        current_round = config.get("current_round", 0)
        serialized_lessons = config.get("lessons", b"")
        
        lessons = []
        if serialized_lessons:
            try:
                lessons = json.loads(serialized_lessons.decode('utf-8'))
            except Exception as e:
                print(f"[ Error ] Failed to deserialize gRPC payload: {e}")

# =====================================================================
        # NON-IID DATA PARTITIONING GUARDRAIL: MIXED PROFILE (5 CLEAN / 5 PHISH)
        # =====================================================================
        if lessons and len(lessons) >= 20:
            print(" [Profile: Corporate Mix] Extracting balanced samples (Indices 0-4 + 10-14)...")
            lessons = lessons[0:5] + lessons[10:15]
        elif lessons:
            print(f"[ Data Guard ] Received unexpected payload count ({len(lessons)}). No partitioning applied.")
        # =====================================================================

        if lessons and len(lessons) > 0:
            print(f"[ Success ] Decompressed {len(lessons)} samples. Initializing PyTorch training...")
            
            
            train_dataset = LessonsDataset(lessons)
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            
            # Run local training loop
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"[ Training Complete ] Avg Local Loss: {total_loss / len(train_loader):.4f}")
            num_examples = len(lessons)
        else:
            print(f"[ Data Guard ] Round {current_round} payload empty. Skipping training.")
            num_examples = 1

        return self.get_parameters(config={}), num_examples, {}

    def evaluate(self, parameters, config):
        print("[ Evaluation ] Validating global parameters on local dataset...")
        
        # Sync weights before evaluation
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        correct = 0
        total = 0
        eval_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = eval_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        print(f"[ Evaluation Results ] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        return float(avg_loss), int(total), {"accuracy": float(accuracy)}

    def get_properties(self, config):
        return {"client_id": "virtual_client_b", "profile": "threat_only"}

if __name__ == "__main__":
    print("[ Client ] Connecting to central manager socket over local loopback...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PhishingDetectionClient()
    )