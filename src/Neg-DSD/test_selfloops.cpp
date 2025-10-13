#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

// Simplified Graph class for testing self-loops
class TestGraph {
private:
    int n;
    vector<vector<pair<int, double>>> pos_adj;
    vector<double> pos_degree;
    double current_total_weight;
    
public:
    TestGraph(int num_vertices) : n(num_vertices), current_total_weight(0.0) {
        pos_adj.resize(n);
        pos_degree.resize(n, 0.0);
    }
    
    void addEdge(int u, int v, double weight) {
        pos_adj[u].emplace_back(v, weight);
        if (u != v) {  // Only add the reverse edge if it's not a self-loop
            pos_adj[v].emplace_back(u, weight);
            pos_degree[v] += weight;
        }
        pos_degree[u] += weight;
        current_total_weight += weight;
    }
    
    void printGraph() {
        cout << "Graph adjacency lists:" << endl;
        for (int i = 0; i < n; i++) {
            cout << "Vertex " << i << " (degree=" << pos_degree[i] << "): ";
            for (const auto& edge : pos_adj[i]) {
                cout << "(" << edge.first << "," << edge.second << ") ";
            }
            cout << endl;
        }
        cout << "Total weight: " << current_total_weight << endl;
    }
    
    double getDegree(int v) { return pos_degree[v]; }
    double getTotalWeight() { return current_total_weight; }
    int getAdjListSize(int v) { return pos_adj[v].size(); }
};

int main() {
    cout << "Testing self-loop handling..." << endl;
    
    TestGraph graph(3);
    
    // Add a regular edge between vertices 0 and 1
    graph.addEdge(0, 1, 2.0);
    
    // Add a self-loop on vertex 2
    graph.addEdge(2, 2, 3.0);
    
    graph.printGraph();
    
    // Test assertions
    cout << "\nRunning tests..." << endl;
    
    // For regular edge (0,1): both vertices should have degree 2.0
    assert(graph.getDegree(0) == 2.0);
    assert(graph.getDegree(1) == 2.0);
    
    // For self-loop (2,2): vertex 2 should have degree 3.0 (counted once)
    assert(graph.getDegree(2) == 3.0);
    
    // Total weight should be 2.0 + 3.0 = 5.0
    assert(graph.getTotalWeight() == 5.0);
    
    // Adjacency list checks
    assert(graph.getAdjListSize(0) == 1);  // vertex 0 has 1 neighbor (vertex 1)
    assert(graph.getAdjListSize(1) == 1);  // vertex 1 has 1 neighbor (vertex 0)
    assert(graph.getAdjListSize(2) == 1);  // vertex 2 has 1 self-loop (not duplicated)
    
    cout << "All tests passed! Self-loops are handled correctly." << endl;
    
    return 0;
} 