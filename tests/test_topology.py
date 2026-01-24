"""
Tests for TUR-DPO topology module.
"""

import pytest
import numpy as np

from turdpo.topology import (
    Node, Edge, TopologyGraph,
    TopologyExtractor, TopologyScorer
)


class TestNode:
    """Tests for Node class."""
    
    def test_node_creation(self):
        node = Node(id="n1", content="Test claim")
        assert node.id == "n1"
        assert node.content == "Test claim"
        assert node.node_type == "claim"
        assert node.correctness_prob == 0.5
    
    def test_node_equality(self):
        node1 = Node(id="n1", content="Test")
        node2 = Node(id="n1", content="Different")
        node3 = Node(id="n2", content="Test")
        
        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID


class TestTopologyGraph:
    """Tests for TopologyGraph class."""
    
    def test_empty_graph(self):
        graph = TopologyGraph()
        assert len(graph) == 0
        assert len(graph.edges) == 0
    
    def test_add_nodes(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="Test 1"))
        graph.add_node(Node(id="n2", content="Test 2"))
        
        assert len(graph) == 2
        assert "n1" in graph.nodes
        assert "n2" in graph.nodes
    
    def test_add_edges(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="Test 1"))
        graph.add_node(Node(id="n2", content="Test 2"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        
        assert len(graph.edges) == 1
        assert "n2" in graph.adjacency["n1"]
        assert "n1" in graph.reverse_adjacency["n2"]
    
    def test_no_self_loops(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="Test"))
        graph.add_edge(Edge(source_id="n1", target_id="n1"))
        
        assert len(graph.edges) == 0
    
    def test_premises_and_conclusions(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="Premise"))
        graph.add_node(Node(id="n2", content="Middle"))
        graph.add_node(Node(id="n3", content="Conclusion"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n3"))
        
        premises = graph.get_premises()
        conclusions = graph.get_conclusions()
        
        assert len(premises) == 1
        assert premises[0].id == "n1"
        assert len(conclusions) == 1
        assert conclusions[0].id == "n3"
    
    def test_cycle_detection(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_node(Node(id="n3", content="C"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n3"))
        graph.add_edge(Edge(source_id="n3", target_id="n1"))  # Creates cycle
        
        cycles = graph.detect_cycles()
        assert len(cycles) > 0
    
    def test_no_cycles(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_node(Node(id="n3", content="C"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n3"))
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0
    
    def test_path_coverage(self):
        # Linear graph: all nodes on path
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_node(Node(id="n3", content="C"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n3"))
        
        coverage = graph.compute_path_coverage()
        assert coverage == 1.0
    
    def test_dangling_nodes(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_node(Node(id="n3", content="Dangling"))  # Not connected
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        
        dangling = graph.get_dangling_nodes()
        assert len(dangling) == 1
        assert dangling[0].id == "n3"
    
    def test_sanitize(self):
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_node(Node(id="n3", content="C"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n3"))
        graph.add_edge(Edge(source_id="n3", target_id="n1"))  # Creates cycle
        
        # Sanitize should break the cycle
        sanitized = graph.sanitize()
        cycles = sanitized.detect_cycles()
        assert len(cycles) == 0


class TestTopologyScorer:
    """Tests for TopologyScorer class."""
    
    def test_perfect_graph(self):
        """Test scoring a well-structured graph."""
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="Premise"))
        graph.add_node(Node(id="n2", content="Step"))
        graph.add_node(Node(id="n3", content="Conclusion"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n3"))
        
        scorer = TopologyScorer()
        score = scorer.compute_score(graph, contradiction_score=0.0)
        
        # Should have high score (good structure)
        assert score > 0.5
    
    def test_graph_with_cycles(self):
        """Test that cycles decrease score."""
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        graph.add_edge(Edge(source_id="n2", target_id="n1"))  # Cycle
        
        scorer = TopologyScorer()
        score = scorer.compute_score(graph, contradiction_score=0.0)
        
        # Should be lower due to cycle
        assert score < 0.8
    
    def test_empty_graph(self):
        """Test scoring empty graph."""
        graph = TopologyGraph()
        scorer = TopologyScorer()
        score = scorer.compute_score(graph)
        
        assert score == 0.0
    
    def test_feature_computation(self):
        """Test feature extraction."""
        graph = TopologyGraph()
        graph.add_node(Node(id="n1", content="A"))
        graph.add_node(Node(id="n2", content="B"))
        graph.add_edge(Edge(source_id="n1", target_id="n2"))
        
        scorer = TopologyScorer()
        features = scorer.compute_features(graph)
        
        assert "path_coverage" in features
        assert "cycle_count" in features
        assert "dangling_count" in features
        assert "node_count" in features
        assert features["node_count"] == 2


class TestTopologyExtractor:
    """Tests for TopologyExtractor class."""
    
    def test_basic_extraction(self):
        """Test basic graph extraction."""
        extractor = TopologyExtractor()
        
        prompt = "What is 2 + 2?"
        response = "First, we have 2. Then we add 2. The result is 4."
        
        graph = extractor.extract(prompt, response)
        
        assert len(graph) > 0
        assert len(graph.edges) > 0
    
    def test_multiple_extraction(self):
        """Test extracting multiple graphs."""
        extractor = TopologyExtractor()
        
        prompt = "Solve x + 1 = 3"
        response = "Subtract 1 from both sides. x = 2."
        
        graphs = extractor.extract_multiple(prompt, response, k=3)
        
        assert len(graphs) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
