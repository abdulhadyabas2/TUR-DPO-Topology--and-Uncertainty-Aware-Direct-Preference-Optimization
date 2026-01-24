"""
TUR-DPO Usage Example

This script demonstrates how to use TUR-DPO for training a language model
on preference data.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import TUR-DPO components
from turdpo.trainer import TURDPOTrainer, TURDPOConfig
from turdpo.topology import TopologyExtractor, TopologyScorer, TopologyGraph, Node, Edge
from turdpo.uncertainty import UncertaintyEstimator, PairWeightComputer
from turdpo.rewards import ShapedReward, RewardDifferenceComputer
from turdpo.loss import TURDPOLoss
from turdpo.verifier import NodeVerifier
from turdpo.calibration import CalibrationMetrics


def demo_topology_extraction():
    """Demonstrate topology extraction from a response."""
    print("=" * 60)
    print("Demo: Topology Extraction")
    print("=" * 60)
    
    # Sample response with reasoning
    prompt = "What is 15 + 27?"
    response = """To solve 15 + 27, I'll break it down:
    First, add the ones: 5 + 7 = 12.
    This gives 2 in ones place with 1 carried.
    Then add the tens: 1 + 2 + 1 = 4.
    So the final answer is 42."""
    
    # Extract topology
    extractor = TopologyExtractor()
    graph = extractor.extract(prompt, response)
    
    print(f"\nExtracted graph: {graph}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    
    # Print node contents
    print("\nNodes:")
    for node_id, node in graph.nodes.items():
        print(f"  {node_id}: {node.content[:50]}...")
    
    # Compute structural features
    print(f"\nPath coverage: {graph.compute_path_coverage():.2%}")
    print(f"Cycles: {graph.count_cycles()}")
    print(f"Dangling nodes: {len(graph.get_dangling_nodes())}")
    
    return graph


def demo_topology_scoring():
    """Demonstrate topology scoring."""
    print("\n" + "=" * 60)
    print("Demo: Topology Scoring")
    print("=" * 60)
    
    # Create a simple graph
    graph = TopologyGraph()
    
    # Add nodes
    graph.add_node(Node(id="n0", content="Given: 15 + 27", node_type="premise"))
    graph.add_node(Node(id="n1", content="5 + 7 = 12", node_type="intermediate"))
    graph.add_node(Node(id="n2", content="Carry 1 to tens", node_type="intermediate"))
    graph.add_node(Node(id="n3", content="1 + 2 + 1 = 4", node_type="intermediate"))
    graph.add_node(Node(id="n4", content="Answer: 42", node_type="conclusion"))
    
    # Add edges
    graph.add_edge(Edge(source_id="n0", target_id="n1"))
    graph.add_edge(Edge(source_id="n1", target_id="n2"))
    graph.add_edge(Edge(source_id="n2", target_id="n3"))
    graph.add_edge(Edge(source_id="n3", target_id="n4"))
    
    # Score the graph
    scorer = TopologyScorer(
        alpha_path=1.0,
        alpha_cycle=0.5,
        alpha_dangling=0.3,
        alpha_contradict=0.4
    )
    
    score = scorer.compute_score(graph, contradiction_score=0.0)
    features = scorer.compute_features(graph)
    
    print(f"\nTopology score: {score:.4f}")
    print(f"Features: {features}")
    
    return score


def demo_uncertainty_estimation():
    """Demonstrate uncertainty estimation."""
    print("\n" + "=" * 60)
    print("Demo: Uncertainty Estimation")
    print("=" * 60)
    
    # Create multiple graph samples (simulating re-elicitation)
    graphs = []
    
    for i in range(3):
        graph = TopologyGraph()
        # Vary slightly between samples
        num_nodes = 4 + (i % 2)
        for j in range(num_nodes):
            graph.add_node(Node(
                id=f"n{j}",
                content=f"Step {j}",
                correctness_prob=0.7 + 0.1 * (i % 2)  # Vary correctness
            ))
        for j in range(num_nodes - 1):
            graph.add_edge(Edge(source_id=f"n{j}", target_id=f"n{j+1}"))
        graphs.append(graph)
    
    # Estimate uncertainty
    estimator = UncertaintyEstimator(
        lambda_epi=0.5,
        lambda_ale=0.5,
        tau=0.05
    )
    
    result = estimator.compute(graphs)
    
    print(f"\nUncertainty results:")
    print(f"  Total: {result.total:.4f}")
    print(f"  Epistemic: {result.epistemic:.4f}")
    print(f"  Aleatoric: {result.aleatoric:.4f}")
    
    # Compute pair weight
    weight = estimator.compute_pair_weight(
        u_pos=result.total,
        u_neg=result.total * 1.2,  # Slightly higher uncertainty for negative
        tau_w=1.2,
        w_min=0.05
    )
    print(f"  Pair weight: {weight:.4f}")
    
    return result


def demo_shaped_reward():
    """Demonstrate shaped reward computation."""
    print("\n" + "=" * 60)
    print("Demo: Shaped Reward")
    print("=" * 60)
    
    # Initialize shaped reward
    reward = ShapedReward(
        a=0.6,  # 60% semantic, 40% topology
        lambda_uncertainty=0.5
    )
    
    # Compute reward for a sample
    result = reward.compute(
        semantic_score=0.85,
        topology_score=0.72,
        uncertainty=0.15
    )
    
    print(f"\nShaped reward components:")
    print(f"  Total reward: {result.total_reward:.4f}")
    print(f"  Semantic component: {result.semantic_component:.4f}")
    print(f"  Topology component: {result.topology_component:.4f}")
    print(f"  Uncertainty penalty: {result.uncertainty_penalty:.4f}")
    
    # Compute reward difference for a pair
    reward_pos = reward.compute(0.85, 0.72, 0.15)
    reward_neg = reward.compute(0.60, 0.45, 0.30)
    
    delta_r = reward.compute_reward_difference(reward_pos, reward_neg)
    print(f"\nReward difference (Δr_φ): {delta_r:.4f}")
    
    return delta_r


def demo_loss_computation():
    """Demonstrate TUR-DPO loss computation."""
    print("\n" + "=" * 60)
    print("Demo: TUR-DPO Loss")
    print("=" * 60)
    
    # Initialize loss function
    loss_fn = TURDPOLoss(
        beta=2.0,
        gamma=1.0
    )
    
    # Create sample batch (batch_size=2)
    batch_size = 2
    
    # Simulated log probabilities
    policy_chosen_logps = torch.tensor([-10.5, -12.3])
    policy_rejected_logps = torch.tensor([-15.2, -18.1])
    reference_chosen_logps = torch.tensor([-11.0, -13.0])
    reference_rejected_logps = torch.tensor([-14.5, -17.5])
    
    # Shaped reward differences
    reward_diff = torch.tensor([0.25, 0.18])
    
    # Uncertainty weights
    weights = torch.tensor([0.85, 0.72])
    
    # Compute loss
    loss, metrics = loss_fn(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
        reward_diff=reward_diff,
        weights=weights
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics:")
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    return loss


def demo_calibration():
    """Demonstrate calibration metrics."""
    print("\n" + "=" * 60)
    print("Demo: Calibration Metrics")
    print("=" * 60)
    
    import numpy as np
    
    # Simulated predictions and labels
    np.random.seed(42)
    n = 100
    
    # Well-calibrated model
    confidences = np.random.beta(5, 2, n)  # Skewed toward high confidence
    accuracies = (np.random.random(n) < confidences).astype(float)
    
    # Compute calibration metrics
    metrics = CalibrationMetrics(num_bins=10)
    result = metrics.compute_all(confidences, accuracies)
    
    print(f"\nCalibration metrics:")
    print(f"  ECE: {result.ece:.4f}")
    print(f"  Brier score: {result.brier:.4f}")
    
    print(f"\nReliability by bin:")
    for i, (acc, conf, count) in enumerate(zip(
        result.bin_accuracies,
        result.bin_confidences,
        result.bin_counts
    )):
        if count > 0:
            print(f"  Bin {i}: acc={acc:.2f}, conf={conf:.2f}, n={count}")
    
    return result


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("TUR-DPO DEMONSTRATION")
    print("=" * 60)
    
    # Run demos
    demo_topology_extraction()
    demo_topology_scoring()
    demo_uncertainty_estimation()
    demo_shaped_reward()
    demo_loss_computation()
    demo_calibration()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
