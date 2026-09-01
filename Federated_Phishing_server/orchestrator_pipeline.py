import os
import json
from openai import OpenAI
from transformers import AutoTokenizer

# Initialize cloud interface
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# load tokenizer 
print("[ Orchestrator ] Loading Hugging Face Tokenizer into memory...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def generate_synthetic_lures(round_num, count=10):
    print(f"\n[ Orchestrator ] Adjusting curriculum difficulty for Round {round_num}...")
    
    # Define tier boundaries
    if round_num <= 10:
        difficulty_tier = "LOW"
        narrative_rules = "Use obvious spelling errors, generic greetings, and overt suspicious links."
    elif round_num <= 30:
        difficulty_tier = "MEDIUM"
        narrative_rules = "Use corporate branding contexts, urgent security updates, and convincing domain spoofs."
    else:
        difficulty_tier = "HIGH"
        narrative_rules = "Employ hyper-targeted spear-phishing logic, psychological pressure, and executive authority mimicry."

    # Build Isolated Prompt Layout
    prompt = f"""
    [TASK]
    Generate exactly {count} realistic email text samples. You must generate an even mix of benign corporate communications and synthetic threats matching the difficulty criteria below.

    <NARRATIVE_CONSTRAINTS>
    Difficulty Tier: {difficulty_tier}
    Target Behavior: {narrative_rules}
    </NARRATIVE_CONSTRAINTS>

    <FORMAT_SCHEMA>
    You must output a single JSON object containing a list named 'samples'. 
    Every object inside that list MUST use these exact keys:
    - "text": The raw string content of the email.
    - "label": Integer, exactly 1 for phishing threats, exactly 0 for benign communication.
    
    Example Blueprint:
    {{
        "samples": [
            {{"text": "Actual email content...", "label": 1}},
            {{"text": "Actual email content...", "label": 0}}
        ]
    }}
    </FORMAT_SCHEMA>
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": "You are an adaptive, data-free synthetic threat generator. You communicate exclusively using valid JSON objects matching the user's explicit FORMAT_SCHEMA blueprint. Do not include introductory text or trailing conversational markdown."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    raw_content = response.choices[0].message.content
    return json.loads(raw_content)

def tokenize_payloads(data_payload):
    raw_samples = []
    

    samples_list = []
    if isinstance(data_payload, list):
        samples_list = data_payload
    elif isinstance(data_payload, dict):
        if "samples" in data_payload and isinstance(data_payload["samples"], list):
            samples_list = data_payload["samples"]
        else:
            for value in data_payload.values():
                if isinstance(value, list):
                    samples_list = value
                    break
            if not samples_list:
                samples_list = [data_payload]

    # Key Healing
    for item in samples_list:
        text = None
        label = None
        
        if isinstance(item, dict):
            for text_key in ["text_content", "text", "content", "email", "body"]:
                if text_key in item:
                    text = str(item[text_key])
                    break
            for label_key in ["label", "phishing", "status", "class"]:
                if label_key in item:
                    try:
                        label = int(item[label_key])
                    except (ValueError, TypeError):
                        pass
                    break
                    

        if text is None or label is None:
            continue 
            
        raw_samples.append({"text": text, "label": label})
            
  
    phishing_pool = [s for s in raw_samples if s["label"] == 1]
    benign_pool = [s for s in raw_samples if s["label"] == 0]
    

    min_class_count = min(len(phishing_pool), len(benign_pool))
    
    
    if min_class_count < 2: 
        print("[ Data Guard ] Round payload compromised by heavy drift. Discarding batch to protect gradient stability.")
        return []
        
    balanced_raw_samples = phishing_pool[:min_class_count] + benign_pool[:min_class_count]
    
    # 
    processed_samples = []
    for item in balanced_raw_samples:
        encoded = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=128,  
            return_tensors=None
        )
        processed_samples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": item["label"]
        })
            
    print(f"[ Data Guard ] Pristine Balance Confirmed: Encoded {min_class_count} Phishing and {min_class_count} Benign vectors.")
    return processed_samples

if __name__ == "__main__":
    
    try:
        print("[ Standalone Test ] Verifying engine against Round 45 complex parsing rules...")
        test_payload = generate_synthetic_lures(round_num=45, count=4)
        result = tokenize_payloads(test_payload)
    except Exception as e:
        print(f"[ Standalone Error ] Baseline check failed: {e}")