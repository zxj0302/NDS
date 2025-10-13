#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

struct Edge {
    int u, v;
    double weight;
    
    Edge(int u, int v, double weight) : u(u), v(v), weight(weight) {}
};

class Graph {
private:
    int n; // number of vertices
    std::vector<std::vector<std::pair<int, double>>> pos_adj; // positive adjacency list
    std::vector<std::vector<std::pair<int, double>>> neg_adj; // negative adjacency list
    std::vector<double> pos_degree; // positive degree of each vertex
    std::vector<double> neg_degree; // negative degree of each vertex
    std::vector<bool> removed; // track removed vertices
    
public:
    Graph(int num_vertices) : n(num_vertices) {
        pos_adj.resize(n);
        neg_adj.resize(n);
        pos_degree.resize(n, 0.0);
        neg_degree.resize(n, 0.0);
        removed.resize(n, false);
    }
    
    void addEdge(int u, int v, double weight) {
        if (weight >= 0) {
            pos_adj[u].push_back({v, weight});
            pos_adj[v].push_back({u, weight});
            pos_degree[u] += weight;
            pos_degree[v] += weight;
        } else {
            neg_adj[u].push_back({v, -weight}); // store absolute value
            neg_adj[v].push_back({u, -weight});
            neg_degree[u] += (-weight);
            neg_degree[v] += (-weight);
        }
    }
    
    double getScore(int v, double C) const {
        if (removed[v]) return std::numeric_limits<double>::lowest();
        return C * pos_degree[v] - neg_degree[v];
    }
    
    double getAverageDegree(const std::vector<int>& vertices) const {
        if (vertices.empty()) return 0.0;
        
        double total_weight = 0.0;
        std::unordered_set<int> vertex_set(vertices.begin(), vertices.end());
        
        // Calculate total induced weight
        for (int u : vertices) {
            if (removed[u]) continue;
            
            // Add positive edges within the subgraph
            for (const auto& edge : pos_adj[u]) {
                int v = edge.first;
                double weight = edge.second;
                if (vertex_set.count(v) && u < v) { // avoid double counting
                    total_weight += weight;
                }
            }
            
            // Subtract negative edges within the subgraph
            for (const auto& edge : neg_adj[u]) {
                int v = edge.first;
                double weight = edge.second;
                if (vertex_set.count(v) && u < v) { // avoid double counting
                    total_weight -= weight;
                }
            }
        }
        
        return total_weight / vertices.size();
    }
    
    void removeVertex(int v) {
        removed[v] = true;
        
        // Update degrees of neighbors
        for (const auto& edge : pos_adj[v]) {
            int neighbor = edge.first;
            double weight = edge.second;
            if (!removed[neighbor]) {
                pos_degree[neighbor] -= weight;
            }
        }
        
        for (const auto& edge : neg_adj[v]) {
            int neighbor = edge.first;
            double weight = edge.second;
            if (!removed[neighbor]) {
                neg_degree[neighbor] -= weight;
            }
        }
    }
    
    void reset() {
        std::fill(removed.begin(), removed.end(), false);
        
        // Recalculate degrees
        std::fill(pos_degree.begin(), pos_degree.end(), 0.0);
        std::fill(neg_degree.begin(), neg_degree.end(), 0.0);
        
        for (int u = 0; u < n; u++) {
            for (const auto& edge : pos_adj[u]) {
                pos_degree[u] += edge.second;
            }
            for (const auto& edge : neg_adj[u]) {
                neg_degree[u] += edge.second;
            }
        }
    }
    
    int getNumVertices() const { return n; }
    bool isRemoved(int v) const { return removed[v]; }
};

std::vector<int> heuristicPeeling(Graph& graph, double C) {
    int n = graph.getNumVertices();
    graph.reset(); // Reset the graph to initial state
    
    std::vector<std::vector<int>> H(n + 1); // H[i] stores vertices remaining at iteration i
    
    // Initialize H[n] with all vertices
    for (int i = 0; i < n; i++) {
        H[n].push_back(i);
    }
    
    // Peeling process
    for (int i = n; i >= 2; i--) {
        // Find vertex with minimum score C*deg+(v) - deg-(v)
        int min_vertex = -1;
        double min_score = std::numeric_limits<double>::max();
        
        for (int v = 0; v < graph.getNumVertices(); v++) {
            if (!graph.isRemoved(v)) {
                double score = graph.getScore(v, C);
                if (score < min_score) {
                    min_score = score;
                    min_vertex = v;
                }
            }
        }
        
        // Remove the vertex with minimum score
        if (min_vertex != -1) {
            graph.removeVertex(min_vertex);
        }
        
        // Store remaining vertices in H[i-1]
        for (int v = 0; v < graph.getNumVertices(); v++) {
            if (!graph.isRemoved(v)) {
                H[i-1].push_back(v);
            }
        }
    }
    
    // Find the subgraph with maximum average degree
    std::vector<int> best_subgraph;
    double best_density = std::numeric_limits<double>::lowest();
    
    for (int i = 1; i <= n; i++) {
        if (!H[i].empty()) {
            double density = graph.getAverageDegree(H[i]);
            if (density > best_density) {
                best_density = density;
                best_subgraph = H[i];
            }
        }
    }
    
    return best_subgraph;
}

Graph readGraphFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    int n, m;
    file >> n >> m;
    
    Graph graph(n);
    
    std::string line;
    std::getline(file, line); // consume the rest of the first line
    
    for (int i = 0; i < m; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 7) {
            int u = std::stoi(tokens[0]);      // first value
            int v = std::stoi(tokens[3]);      // fourth value  
            double weight = std::stod(tokens[6]); // seventh value
            
            graph.addEdge(u, v, weight);
        }
    }
    
    file.close();
    return graph;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    
    // Read graph from file
    Graph graph = readGraphFromFile(filename);
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run heuristic peeling with C = 1.0
    double C = 1.0;
    std::vector<int> densest_subgraph = heuristicPeeling(graph, C);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_seconds = duration.count() / 1000000.0;
    
    // Output the time (6 decimal places)
    std::cout << std::fixed << std::setprecision(6) << time_seconds << std::endl;
    
    // Output the node IDs
    for (size_t i = 0; i < densest_subgraph.size(); i++) {
        if (i > 0) std::cout << " ";
        std::cout << densest_subgraph[i];
    }
    std::cout << std::endl;
    
    return 0;
}