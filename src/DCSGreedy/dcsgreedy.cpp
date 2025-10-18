#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>

using namespace std;
using namespace boost::heap;

struct HeapNode {
    double degree;
    int vertex;

    HeapNode(double d, int v) : degree(d), vertex(v) {}

    bool operator<(const HeapNode& other) const {
        return degree > other.degree; // Min-heap
    }
};

class Graph {
private:
    int n;
    vector<vector<pair<int, double>>> adj;
    
public:
    Graph(int num_vertices) : n(num_vertices) {
        adj.resize(n);
    }
    
    void addEdge(int u, int v, double weight) {
        adj[u].emplace_back(v, weight);
        if (u != v) {
            adj[v].emplace_back(u, weight);
        }
    }
    
    int getVertexCount() const { return n; }
    
    const vector<vector<pair<int, double>>>& getAdj() const { return adj; }
    
    // Get max weighted edge
    pair<pair<int, int>, double> getMaxWeightEdge() const {
        int max_u = 0, max_v = 0;
        double max_weight = -1e100;
        
        for (int u = 0; u < n; u++) {
            for (const auto& edge : adj[u]) {
                int v = edge.first;
                double w = edge.second;
                if (u <= v && w > max_weight) {
                    max_weight = w;
                    max_u = u;
                    max_v = v;
                }
            }
        }
        
        return make_pair(make_pair(max_u, max_v), max_weight);
    }
};

// Optimized Greedy algorithm using Fibonacci heap
class GreedySolver {
private:
    const Graph& G;
    int n;
    vector<bool> removed;
    vector<int> removal_order;
    vector<double> current_degree;
    fibonacci_heap<HeapNode> pq;
    vector<fibonacci_heap<HeapNode>::handle_type> handles;
    double current_total_weight;
    int current_vertex_count;
    
    void updateDegree(int v) {
        if (removed[v]) return;
        
        double new_degree = 0.0;
        for (const auto& edge : G.getAdj()[v]) {
            int neighbor = edge.first;
            double weight = edge.second;
            if (!removed[neighbor]) {
                new_degree += weight;
            }
        }
        
        current_degree[v] = new_degree;
        pq.update(handles[v], HeapNode(new_degree, v));
    }
    
public:
    GreedySolver(const Graph& graph) : G(graph), n(graph.getVertexCount()) {
        removed.resize(n, false);
        current_degree.resize(n, 0.0);
        handles.resize(n);
        current_total_weight = 0.0;
        current_vertex_count = n;
    }
    
    vector<int> solve() {
        // Initialize degrees and heap
        for (int u = 0; u < n; u++) {
            double deg = 0.0;
            for (const auto& edge : G.getAdj()[u]) {
                int v = edge.first;
                double w = edge.second;
                deg += w;
                if (u <= v) {
                    current_total_weight += w;
                }
            }
            current_degree[u] = deg;
            handles[u] = pq.push(HeapNode(deg, u));
        }
        
        double best_density = current_vertex_count > 0 ? current_total_weight / current_vertex_count : 0.0;
        int best_step = 0;
        
        // Greedy removal
        for (int step = 1; current_vertex_count > 0 && !pq.empty(); step++) {
            HeapNode min_node = pq.top();
            pq.pop();
            int v = min_node.vertex;
            
            if (removed[v]) continue;
            
            removed[v] = true;
            removal_order.push_back(v);
            current_vertex_count--;
            
            // Update total weight and neighbors
            for (const auto& edge : G.getAdj()[v]) {
                int neighbor = edge.first;
                double weight = edge.second;
                
                if (neighbor == v) {
                    current_total_weight -= weight;
                } else if (!removed[neighbor]) {
                    current_total_weight -= weight;
                    updateDegree(neighbor);
                }
            }
            
            double current_density = current_vertex_count > 0 ? current_total_weight / current_vertex_count : 0.0;
            
            if (current_density > best_density) {
                best_density = current_density;
                best_step = step;
            }
        }
        
        // Reconstruct best vertex set
        vector<int> result;
        set<int> removed_set(removal_order.begin(), removal_order.begin() + best_step);
        
        for (int i = 0; i < n; i++) {
            if (removed_set.find(i) == removed_set.end()) {
                result.push_back(i);
            }
        }
        
        return result;
    }
};

// Fast connected components using Union-Find
class UnionFind {
private:
    vector<int> parent;
    vector<int> rank;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x);
        int py = find(y);
        
        if (px == py) return;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }
};

vector<vector<int>> getConnectedComponents(const Graph& G, const vector<int>& vertices) {
    int n = G.getVertexCount();
    set<int> vertex_set(vertices.begin(), vertices.end());
    UnionFind uf(n);
    
    for (int u : vertices) {
        for (const auto& edge : G.getAdj()[u]) {
            int v = edge.first;
            if (vertex_set.count(v) && edge.second != 0) {
                uf.unite(u, v);
            }
        }
    }
    
    map<int, vector<int>> components_map;
    for (int v : vertices) {
        components_map[uf.find(v)].push_back(v);
    }
    
    vector<vector<int>> components;
    for (const auto& pair : components_map) {
        components.push_back(pair.second);
    }
    
    return components;
}

double calculateDensity(const Graph& G, const vector<int>& vertices) {
    if (vertices.empty()) return -1e100;
    
    set<int> vertex_set(vertices.begin(), vertices.end());
    double total_weight = 0.0;
    
    for (int u : vertices) {
        for (const auto& edge : G.getAdj()[u]) {
            int v = edge.first;
            double w = edge.second;
            if (vertex_set.count(v) && u <= v) {
                total_weight += w;
            }
        }
    }
    
    return total_weight / vertices.size();
}

// DCSGreedy Algorithm (Algorithm 2 from paper)
pair<vector<int>, double> DCSGreedy(const Graph& GD, const Graph& GD_plus) {
    int n = GD.getVertexCount();
    
    // Find edge with maximum weight in GD and use as initial S
    auto max_edge_info = GD.getMaxWeightEdge();
    int max_u = max_edge_info.first.first;
    int max_v = max_edge_info.first.second;
    
    // S = {u, v} with maximum edge weight
    vector<int> S = {max_u, max_v};
    
    // S1 = Greedy(GD)
    GreedySolver solver1(GD);
    vector<int> S1 = solver1.solve();
    
    // S2 = Greedy(GD+)
    GreedySolver solver2(GD_plus);
    vector<int> S2 = solver2.solve();
    
    // Pick best among S, S1, S2
    double density_S = calculateDensity(GD, S);
    double density_S1 = calculateDensity(GD, S1);
    double density_S2 = calculateDensity(GD, S2);
    
    vector<int> best_S = S;
    double best_density = density_S;
    
    if (density_S1 > best_density) {
        best_S = S1;
        best_density = density_S1;
    }
    if (density_S2 > best_density) {
        best_S = S2;
        best_density = density_S2;
    }
    
    // Check if GD(S) is connected, if not pick best connected component
    vector<vector<int>> components = getConnectedComponents(GD, best_S);
    
    if (components.size() > 1) {
        double best_comp_density = -1e100;
        vector<int> best_comp;
        
        for (const auto& comp : components) {
            double comp_density = calculateDensity(GD, comp);
            if (comp_density > best_comp_density) {
                best_comp_density = comp_density;
                best_comp = comp;
            }
        }
        
        best_S = best_comp;
        best_density = best_comp_density;
    }
    
    return make_pair(best_S, best_density);
}

Graph readGraphFromFile(const string& filename, bool reverse_weight) {
    ifstream infile(filename);
    int n, m;
    infile >> n >> m;
    
    Graph graph(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        double weight;
        infile >> u >> v >> weight;
        graph.addEdge(u, v, reverse_weight ? -weight : weight);
    }
    infile.close();
    
    return graph;
}

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    if (argc < 4 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <graph_file> <output_json> <reverse_weight> [num_iterations]" << endl;
        return 1;
    }
    
    string filename = argv[1];
    string output_filename = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    int num_iterations = (argc >= 5) ? stoi(argv[4]) : 1;

    // Read graph (GD is the difference graph with all edges)
    Graph GD = readGraphFromFile(filename, reverse_weight);
    
    // Build GD_plus (only positive weighted edges)
    int n = GD.getVertexCount();
    Graph GD_plus(n);
    for (int u = 0; u < n; u++) {
        for (const auto& edge : GD.getAdj()[u]) {
            int v = edge.first;
            double w = edge.second;
            if (w > 0 && u <= v) {
                GD_plus.addEdge(u, v, w);
            }
        }
    }
    
    vector<int> first_result;
    double first_density = 0.0;
    double total_time = 0.0;
    
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        auto start_time = chrono::high_resolution_clock::now();
        auto result = DCSGreedy(GD, GD_plus);
        auto end_time = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e9;
        total_time += time_seconds;
        
        if (iteration == 0) {
            first_result = result.first;
            first_density = result.second;
        }
    }
    
    double avg_time = total_time / num_iterations;
    
    // Write results to JSON file
    ofstream json_file(output_filename);
    json_file << fixed << setprecision(6);
    json_file << "{\n";
    json_file << "  \"time\": " << avg_time << ",\n";
    json_file << "  \"nodes\": [";
    for (size_t i = 0; i < first_result.size(); i++) {
        if (i > 0) json_file << ", ";
        json_file << first_result[i];
    }
    json_file << "],\n";
    json_file << "  \"size\": " << first_result.size() << ",\n";
    json_file << "  \"density\": " << first_density << "\n";
    json_file << "}\n";
    json_file.close();
    
    return 0;
}