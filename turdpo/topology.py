"""
Topology Module for TUR-DPO

This module implements topology extraction and scoring for reasoning graphs.
Based on Equation (1) from the paper:
    s_topo(G) = α₁ q_path - α₂ c_cycle - α₃ d_dangling - α₄ q_contradict
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import deque
import re


@dataclass
class Node:
    """Represents an atomic subclaim or reasoning step in the topology graph."""
    id: str
    content: str
    node_type: str = "claim"  # "claim", "premise", "conclusion", "intermediate"
    correctness_prob: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False


@dataclass
class Edge:
    """Represents a support or dependency relation between nodes."""
    source_id: str
    target_id: str
    edge_type: str = "supports"  # "supports", "contradicts", "depends"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id))


class TopologyGraph:
    """
    Directed graph representing the reasoning topology of a response.
    
    Nodes represent atomic subclaims/steps, edges represent support relations.
    Provides methods for structural analysis (cycles, paths, dangling nodes).
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[str]] = {}  # outgoing edges
        self.reverse_adjacency: Dict[str, List[str]] = {}  # incoming edges
        
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
        if node.id not in self.reverse_adjacency:
            self.reverse_adjacency[node.id] = []
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # Ensure nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge: {edge.source_id} -> {edge.target_id}")
        
        # Prevent self-loops
        if edge.source_id == edge.target_id:
            return
            
        self.edges.append(edge)
        self.adjacency[edge.source_id].append(edge.target_id)
        self.reverse_adjacency[edge.target_id].append(edge.source_id)
    
    def get_premises(self) -> List[Node]:
        """Get nodes with no incoming edges (premises/starting points)."""
        return [node for node_id, node in self.nodes.items() 
                if len(self.reverse_adjacency.get(node_id, [])) == 0]
    
    def get_conclusions(self) -> List[Node]:
        """Get nodes with no outgoing edges (conclusions/final claims)."""
        return [node for node_id, node in self.nodes.items()
                if len(self.adjacency.get(node_id, [])) == 0]
    
    def get_dangling_nodes(self) -> List[Node]:
        """
        Get dangling nodes - nodes that are neither connected to premises
        nor to conclusions through any path.
        """
        if len(self.nodes) == 0:
            return []
        
        # Find all nodes reachable from premises
        reachable_from_premises = set()
        premises = self.get_premises()
        for premise in premises:
            reachable_from_premises.update(self._bfs_reachable(premise.id, forward=True))
        
        # Find all nodes that can reach conclusions
        can_reach_conclusions = set()
        conclusions = self.get_conclusions()
        for conclusion in conclusions:
            can_reach_conclusions.update(self._bfs_reachable(conclusion.id, forward=False))
        
        # Dangling = not on any valid path from premise to conclusion
        valid_path_nodes = reachable_from_premises & can_reach_conclusions
        dangling = [node for node_id, node in self.nodes.items() 
                   if node_id not in valid_path_nodes]
        
        return dangling
    
    def _bfs_reachable(self, start_id: str, forward: bool = True) -> Set[str]:
        """BFS to find all reachable nodes from start_id."""
        visited = set()
        queue = deque([start_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            neighbors = self.adjacency.get(current, []) if forward else self.reverse_adjacency.get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return visited
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the graph using DFS.
        Returns list of cycles found.
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for neighbor in self.adjacency.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            
            path.pop()
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return cycles
    
    def count_cycles(self) -> int:
        """Count the number of cycles in the graph."""
        return len(self.detect_cycles())
    
    def get_minimal_valid_paths(self) -> List[List[str]]:
        """
        Find all minimal valid paths from premises to conclusions.
        A valid path connects a premise to a conclusion.
        """
        premises = self.get_premises()
        conclusions = self.get_conclusions()
        all_paths = []
        
        for premise in premises:
            for conclusion in conclusions:
                paths = self._find_all_paths(premise.id, conclusion.id)
                all_paths.extend(paths)
        
        return all_paths
    
    def _find_all_paths(self, start_id: str, end_id: str, max_depth: int = 20) -> List[List[str]]:
        """Find all paths from start to end using DFS with depth limit."""
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            if current == end_id:
                paths.append(path.copy())
                return
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start_id, [start_id], {start_id})
        return paths
    
    def compute_path_coverage(self) -> float:
        """
        Compute the fraction of nodes that participate in at least one
        valid path from premises to conclusions.
        """
        if len(self.nodes) == 0:
            return 0.0
        
        paths = self.get_minimal_valid_paths()
        nodes_in_paths = set()
        for path in paths:
            nodes_in_paths.update(path)
        
        return len(nodes_in_paths) / len(self.nodes)
    
    def get_edge_distribution(self) -> Dict[str, float]:
        """Get normalized distribution over edges for JSD computation."""
        if len(self.edges) == 0:
            return {}
        
        dist = {}
        for edge in self.edges:
            key = f"{edge.source_id}->{edge.target_id}"
            dist[key] = dist.get(key, 0) + edge.weight
        
        # Normalize
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}
    
    def get_path_distribution(self) -> Dict[str, float]:
        """Get normalized distribution over paths for JSD computation."""
        paths = self.get_minimal_valid_paths()
        if not paths:
            return {}
        
        dist = {}
        for path in paths:
            key = "->".join(path)
            dist[key] = dist.get(key, 0) + 1
        
        # Normalize
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}
    
    def sanitize(self) -> 'TopologyGraph':
        """
        Sanitize the graph by:
        1. Removing self-loops
        2. Breaking cycles by minimal edge cut
        3. Merging paraphrase nodes (if similarity > threshold)
        """
        # Remove existing cycles by removing back edges
        cycles = self.detect_cycles()
        edges_to_remove = set()
        
        for cycle in cycles:
            if len(cycle) > 1:
                # Remove the edge that creates the cycle (last edge)
                edges_to_remove.add((cycle[-2], cycle[-1]))
        
        # Filter out problematic edges
        self.edges = [e for e in self.edges 
                     if (e.source_id, e.target_id) not in edges_to_remove]
        
        # Rebuild adjacency lists
        self.adjacency = {node_id: [] for node_id in self.nodes}
        self.reverse_adjacency = {node_id: [] for node_id in self.nodes}
        
        for edge in self.edges:
            self.adjacency[edge.source_id].append(edge.target_id)
            self.reverse_adjacency[edge.target_id].append(edge.source_id)
        
        return self
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return f"TopologyGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


class TopologyExtractor:
    """
    Extract reasoning topology from text responses.
    
    Decomposes text into atomic statements and links support relations.
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        extraction_prompt_template: Optional[str] = None,
        max_nodes: int = 10,
        max_edges: int = 20
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        self.extraction_prompt_template = extraction_prompt_template or self._default_prompt()
    
    def _default_prompt(self) -> str:
        return """Analyze the following response and extract its reasoning structure.

Response: {response}

Extract:
1. CLAIMS: List each atomic claim or statement (numbered)
2. SUPPORTS: List support relations as "X supports Y" where X and Y are claim numbers
3. PREMISE: Which claims are starting premises (no support needed)?
4. CONCLUSION: Which claim is the final conclusion?

Format your output as:
CLAIMS:
1. [claim text]
2. [claim text]
...

SUPPORTS:
1 supports 2
2 supports 3
...

PREMISE: [numbers]
CONCLUSION: [number]
"""
    
    def extract(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        perturbation: bool = False
    ) -> TopologyGraph:
        """
        Extract topology graph from a response.
        
        Args:
            prompt: The input prompt
            response: The model's response
            temperature: Sampling temperature (for perturbation)
            perturbation: Whether to add prompt perturbations for uncertainty estimation
            
        Returns:
            TopologyGraph representing the reasoning structure
        """
        if self.model is None:
            # Fallback to rule-based extraction
            return self._rule_based_extract(response)
        
        # Use model-based extraction
        extraction_prompt = self.extraction_prompt_template.format(
            prompt=prompt,
            response=response
        )
        
        if perturbation:
            # Add minor variations for epistemic uncertainty estimation
            extraction_prompt = self._add_perturbation(extraction_prompt)
        
        # Generate extraction output
        # This would use self.model and self.tokenizer
        # For now, use rule-based fallback
        return self._rule_based_extract(response)
    
    def _rule_based_extract(self, response: str) -> TopologyGraph:
        """
        Rule-based topology extraction using sentence segmentation.
        """
        graph = TopologyGraph()
        
        # Split into sentences
        sentences = self._split_sentences(response)
        
        if not sentences:
            return graph
        
        # Create nodes for each sentence
        for i, sentence in enumerate(sentences[:self.max_nodes]):
            node_type = "premise" if i == 0 else ("conclusion" if i == len(sentences) - 1 else "intermediate")
            node = Node(
                id=f"n{i}",
                content=sentence.strip(),
                node_type=node_type
            )
            graph.add_node(node)
        
        # Create sequential edges (simple chain structure)
        for i in range(min(len(sentences) - 1, self.max_nodes - 1)):
            edge = Edge(
                source_id=f"n{i}",
                target_id=f"n{i+1}",
                edge_type="supports"
            )
            graph.add_edge(edge)
        
        return graph.sanitize()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _add_perturbation(self, prompt: str) -> str:
        """Add minor perturbations to prompt for uncertainty estimation."""
        perturbations = [
            "Please analyze carefully: ",
            "Think step by step: ",
            "Consider the reasoning: ",
            "",
        ]
        import random
        return random.choice(perturbations) + prompt
    
    def extract_multiple(
        self,
        prompt: str,
        response: str,
        k: int = 3
    ) -> List[TopologyGraph]:
        """
        Extract K topology graphs with perturbations for epistemic uncertainty.
        """
        graphs = []
        for i in range(k):
            graph = self.extract(
                prompt=prompt,
                response=response,
                temperature=0.3 if i > 0 else 0.0,
                perturbation=(i > 0)
            )
            graphs.append(graph)
        return graphs


class TopologyScorer:
    """
    Compute topology score from graph features.
    
    Based on Equation (1):
        s_topo(G) = α₁ q_path - α₂ c_cycle - α₃ d_dangling - α₄ q_contradict
    """
    
    def __init__(
        self,
        alpha_path: float = 1.0,
        alpha_cycle: float = 0.5,
        alpha_dangling: float = 0.3,
        alpha_contradict: float = 0.4,
        normalize: bool = True,
        score_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize topology scorer.
        
        Args:
            alpha_path: Weight for path coverage (positive contribution)
            alpha_cycle: Weight for cycle count (negative contribution)
            alpha_dangling: Weight for dangling nodes (negative contribution)
            alpha_contradict: Weight for contradiction score (negative contribution)
            normalize: Whether to normalize score to [0, 1]
            score_range: Target range for normalized scores
        """
        self.alpha_path = alpha_path
        self.alpha_cycle = alpha_cycle
        self.alpha_dangling = alpha_dangling
        self.alpha_contradict = alpha_contradict
        self.normalize = normalize
        self.score_range = score_range
        
    def compute_score(
        self,
        graph: TopologyGraph,
        contradiction_score: float = 0.0
    ) -> float:
        """
        Compute topology score for a graph.
        
        Args:
            graph: The topology graph to score
            contradiction_score: Pre-computed contradiction score (from NLI verifier)
            
        Returns:
            Topology score (higher is better)
        """
        if len(graph) == 0:
            return 0.0
        
        # Compute components
        q_path = graph.compute_path_coverage()
        c_cycle = graph.count_cycles() / max(len(graph), 1)  # Normalize by graph size
        d_dangling = len(graph.get_dangling_nodes()) / len(graph)
        q_contradict = contradiction_score
        
        # Apply Equation (1)
        score = (
            self.alpha_path * q_path
            - self.alpha_cycle * c_cycle
            - self.alpha_dangling * d_dangling
            - self.alpha_contradict * q_contradict
        )
        
        if self.normalize:
            # Normalize to score_range
            min_score = -self.alpha_cycle - self.alpha_dangling - self.alpha_contradict
            max_score = self.alpha_path
            
            if max_score > min_score:
                normalized = (score - min_score) / (max_score - min_score)
                score = self.score_range[0] + normalized * (self.score_range[1] - self.score_range[0])
            else:
                score = (self.score_range[0] + self.score_range[1]) / 2
        
        return score
    
    def compute_features(self, graph: TopologyGraph) -> Dict[str, float]:
        """Compute all topology features for analysis."""
        return {
            "path_coverage": graph.compute_path_coverage(),
            "cycle_count": graph.count_cycles(),
            "dangling_count": len(graph.get_dangling_nodes()),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "num_premises": len(graph.get_premises()),
            "num_conclusions": len(graph.get_conclusions()),
        }
