#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <boost/heap/fibonacci_heap.hpp>

using namespace std;
using namespace boost::heap;

struct HeapNode {
    double score;
    int vertex;

    HeapNode(double s, int v) : score(s), vertex(v) {}

    bool operator<(const HeapNode& other) const {
        return score > other.score; // Min-heap (reverse comparison)
    }
};

class Graph {
private:
    int n;
    vector<vector<pair<int, double>>> pos_adj;
    vector<vector<pair<int, double>>> neg_adj;
    vector<double> pos_degree;
    vector<double> neg_degree;
    vector<bool> removed;
    vector<int> removal_step;

    double current_total_weight;
    int current_vertex_count;

    // Fibonacci heap and handles
    fibonacci_heap<HeapNode> pq;
    vector<fibonacci_heap<HeapNode>::handle_type> handles;

public:
    Graph(int num_vertices) : n(num_vertices), current_total_weight(0.0), current_vertex_count(n) {
        pos_adj.resize(n);
        neg_adj.resize(n);
        pos_degree.resize(n, 0.0);
        neg_degree.resize(n, 0.0);
        removed.resize(n, false);
        removal_step.resize(n, -1);
        handles.resize(n);
    }

    // Copy constructor for deep copying
    Graph(const Graph& other) : n(other.n), current_total_weight(other.current_total_weight),
                                current_vertex_count(other.current_vertex_count) {
        pos_adj = other.pos_adj;
        neg_adj = other.neg_adj;
        pos_degree = other.pos_degree;
        neg_degree = other.neg_degree;
        removed = other.removed;
        removal_step = other.removal_step;
        handles.resize(n);
        // Note: pq and handles will be reinitialized when needed
    }

    // Assignment operator
    Graph& operator=(const Graph& other) {
        if (this != &other) {
            n = other.n;
            current_total_weight = other.current_total_weight;
            current_vertex_count = other.current_vertex_count;
            pos_adj = other.pos_adj;
            neg_adj = other.neg_adj;
            pos_degree = other.pos_degree;
            neg_degree = other.neg_degree;
            removed = other.removed;
            removal_step = other.removal_step;
            handles.resize(n);
            pq.clear();
        }
        return *this;
    }

    void addEdge(int u, int v, double weight) {
        if (weight >= 0) {
            pos_adj[u].emplace_back(v, weight);
            if (u != v) {  // Only add the reverse edge if it's not a self-loop
                pos_adj[v].emplace_back(u, weight);
                pos_degree[v] += weight;
            }
            pos_degree[u] += weight;
            current_total_weight += weight;
        } else {
            double abs_weight = -weight;
            neg_adj[u].emplace_back(v, abs_weight);
            if (u != v) {  // Only add the reverse edge if it's not a self-loop
                neg_adj[v].emplace_back(u, abs_weight);
                neg_degree[v] += abs_weight;
            }
            neg_degree[u] += abs_weight;
            current_total_weight -= abs_weight;
        }
    }

    void initializePriorityQueue(double C) {
        pq.clear(); // Clear any existing heap
        for (int i = 0; i < n; i++) {
            double score = C * pos_degree[i] - neg_degree[i];
            handles[i] = pq.push(HeapNode(score, i));
        }
    }

    int getMinVertex() {
        if (pq.empty()) return -1;

        HeapNode min_node = pq.top();
        pq.pop();
        return min_node.vertex;
    }

    void removeVertex(int v, int step, double C) {
        removed[v] = true;
        removal_step[v] = step;
        current_vertex_count--;

        double weight_change = 0.0;

        for (const auto& edge : pos_adj[v]) {
            int neighbor = edge.first;
            double weight = edge.second;
            if (neighbor == v) {
                weight_change -= weight;
                continue;
            }
            if (!removed[neighbor]) {
                pos_degree[neighbor] -= weight;
                double new_score = C * pos_degree[neighbor] - neg_degree[neighbor];
                pq.update(handles[neighbor], HeapNode(new_score, neighbor));
                weight_change -= weight;
            }
        }

        for (const auto& edge : neg_adj[v]) {
            int neighbor = edge.first;
            double weight = edge.second;
            if (neighbor == v) {
                weight_change += weight;
                continue;
            }
            if (!removed[neighbor]) {
                neg_degree[neighbor] -= weight;
                double new_score = C * pos_degree[neighbor] - neg_degree[neighbor];
                pq.update(handles[neighbor], HeapNode(new_score, neighbor));
                weight_change += weight;
            }
        }

        current_total_weight += weight_change;
    }

    double getCurrentDensity() const {
        return current_vertex_count > 0 ? current_total_weight / current_vertex_count : 0.0;
    }

    vector<int> getVerticesAtStep(int target_step) const {
        vector<int> result;
        for (int i = 0; i < n; i++) {
            if (removal_step[i] == -1 || removal_step[i] > target_step) {
                result.push_back(i);
            }
        }
        return result;
    }

    int getVertexCount() const {
        return current_vertex_count;
    }
};

pair<vector<int>, double> heuristicPeeling(Graph& graph, double C) {
    graph.initializePriorityQueue(C);

    double best_density = graph.getCurrentDensity();
    int best_step = 0;

    for (int step = 1; graph.getVertexCount() > 1; step++) {
        int min_vertex = graph.getMinVertex();
        if (min_vertex == -1) break;

        graph.removeVertex(min_vertex, step, C);

        double current_density = graph.getCurrentDensity();
        // cout << "Step: " << step << " Current Density: " << current_density << " Best Density: " << best_density << endl;
        if (current_density > best_density) {
            best_density = current_density;
            best_step = step;
        }
    }

    return make_pair(graph.getVerticesAtStep(best_step), best_density);
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

    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " <graph_file> <output_json> <reverse_weight> <C_value> [num_iterations]" << endl;
        return 1;
    }

    string filename = argv[1];
    string output_filename = argv[2];
    bool reverse_weight = (string(argv[3]) == "1"); // Default to false if not provided
    double C = stod(argv[4]);
    int num_iterations = (argc >= 6) ? stoi(argv[5]) : 1; // Default to 1 iteration

    // Read graph once
    Graph original_graph = readGraphFromFile(filename, reverse_weight);

    // Variables to store results from first iteration (for output consistency)
    vector<int> first_densest_subgraph;
    double first_best_density = 0.0;

    // Variable to accumulate timing results
    double total_time = 0.0;

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        // Create fresh copy of the graph for this iteration
        Graph graph_copy = original_graph;

        auto start_time = chrono::high_resolution_clock::now();
        auto result = heuristicPeeling(graph_copy, C);
        auto end_time = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e9;
        total_time += time_seconds;

        // Store first iteration results for output
        if (iteration == 0) {
            first_densest_subgraph = result.first;
            first_best_density = result.second;
        }
    }

    // Calculate average time
    double avg_time = total_time / num_iterations;

    // Write results to JSON file
    ofstream json_file(output_filename);
    if (!json_file.is_open()) {
        cerr << "Error: Could not open output file " << output_filename << endl;
        return 1;
    }

    json_file << fixed << setprecision(6);
    json_file << "{\n";
    json_file << "  \"time\": " << avg_time << ",\n";
    json_file << "  \"nodes\": [";
    for (size_t i = 0; i < first_densest_subgraph.size(); i++) {
        if (i > 0) json_file << ", ";
        json_file << first_densest_subgraph[i];
    }
    json_file << "],\n";
    json_file << "  \"size\": " << first_densest_subgraph.size() << ",\n";
    json_file << "  \"density\": " << first_best_density << "\n";
    json_file << "}\n";

    json_file.close();

    cout << "Results written to " << output_filename << endl;

    return 0;
}