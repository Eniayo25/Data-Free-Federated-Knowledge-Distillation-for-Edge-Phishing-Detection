import flwr as fl
import json
import os
import pickle
from flwr.common import parameters_to_ndarrays


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if sum(examples) == 0:
        return {"accuracy": 0.0}
    

    return {"accuracy": sum(accuracies) / sum(examples)}


class MetricsPrintingFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # standard aggregation logic
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # save: Serialize active weights
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            checkpoint_path = f"non_iid_model_round_{server_round}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(ndarrays, f)
            print(f"\n [NON-IID CHECKPOINT] Parameters saved to '{checkpoint_path}'")
            
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics:
            print("\n" + "="*60)
            print(f" [SERVER LOG] Round {server_round} Aggregated Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print("="*60 + "\n")
            
        return loss, metrics


from orchestrator_pipeline import generate_synthetic_lures, tokenize_payloads

def fit_config(server_round: int):
    """
    Dynamic Configuration Hook: Executes at the start of EVERY round.
    Fetches 20 synthetic lures (10 clean, 10 phishing), tokenizes them, and ships them via gRPC.
    """
    print(f"\n[ gRPC Link ] Preparing Non-IID transport payload for Round {server_round}...")
    
    try:
        # Generate 20 samples (10 benign, 10 phishing)
        raw_payload = generate_synthetic_lures(round_num=server_round, count=20)
        
        # Convert to numerical tokens
        tokenized_payload = tokenize_payloads(raw_payload)
        
        # Compress the structural JSON array
        serialized_lessons = json.dumps(tokenized_payload).encode('utf-8')
        
        print(f"[ gRPC Link ] Payload compression complete. Streaming 20-sample package over subnet...")
        
        return {
            "current_round": server_round,
            "lessons": serialized_lessons
        }
    except Exception as e:
        print(f"[ Failure ] Server failed to package round metadata: {e}")
        return {"current_round": server_round, "lessons": b""}

def main():
    print("=====================================================")
    print("[ Server ] Initializing Tier 2 Non-IID Workstation Server...")
    print("=====================================================")

    strategy = MetricsPrintingFedAvg(
        fraction_fit=1.0,          
        fraction_evaluate=1.0,     
        min_fit_clients=3,         
        min_evaluate_clients=3,    
        min_available_clients=3,   
        on_fit_config_fn=fit_config, 
        evaluate_metrics_aggregation_fn=weighted_average 
    )

    # Initialize native gRPC engine
    history = fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=50), 
        strategy=strategy
    )

    # ==========================================
    # AUTO-EXPORT TO CSV
    # ==========================================
    print("\n[ Server ] Training complete. Exporting Non-IID run history...")
    try:
        import pandas as pd
        
        rounds = [r for r, _ in history.losses_distributed]
        losses = [loss for _, loss in history.losses_distributed]
        
        acc_dict = dict(history.metrics_distributed.get("accuracy", []))
        accuracies = [acc_dict.get(r, 0.0) for r in rounds]
        
        df = pd.DataFrame({
            "Round": rounds,
            "Loss": losses,
            "Accuracy": accuracies
        })
        
        df.to_csv("non_iid_baseline_metrics.csv", index=False)
        print("=======> SUCCESS: Saved Non-IID metrics to 'non_iid_baseline_metrics.csv'")
        
    except Exception as e:
        print(f"[ Error ] Failed to export training history: {e}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ Warning ] OPENAI_API_KEY environment variable not detected.")
    main()