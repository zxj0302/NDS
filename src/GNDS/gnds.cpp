#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>

using namespace std;
using namespace boost;

/*
 * =========================================================
 * Node and Edge Structures
 * =========================================================
 */
// Node status
enum class Status {
    Out,
    Fringe,
    In
};
// Now includes self-loop flags
struct NodeProperty {
    // self-loop info
    bool has_self_loop = false;
    double self_loop_polarity = 0.0;

    // status, priority_key, and other bookkeeping
    Status status = Status::Out;   // "out", "fringe", or "in"
    double priority_key = 0.0;
    unsigned in_neighbor_count = 0;
};

struct EdgeProperty {
    double edge_polarity = 0.0;
};

// Define the Graph using adjacency_list with bundled properties
using Graph = adjacency_list<vecS, vecS, undirectedS, NodeProperty, EdgeProperty>;
using Vertex = graph_traits<Graph>::vertex_descriptor;
using Edge = graph_traits<Graph>::edge_descriptor;
using Traits = graph_traits<Graph>;

/*
 * =========================================================
 * Priority Structure and Fibonacci Heap
 * =========================================================
 */
struct PriorityTuple {
    double priority_key;
    Vertex vertex;

    // We use "less than" so that the largest priority_key is on top.
    bool operator<(const PriorityTuple& other) const {
        return
            (priority_key < other.priority_key) ||
            (priority_key == other.priority_key && vertex < other.vertex);
    }
};

using FibHeap = heap::fibonacci_heap<PriorityTuple>;

/*
 * =========================================================
 * Read Edge List
 * =========================================================
 *
 * File format:
 *   First line: <num_nodes> <num_edges>
 *   Next <num_edges> lines:
 *     <u> <polarity_u> <polarity_label_u>
 *     <v> <polarity_v> <polarity_label_v>
 *     <edge_polarity>
 */
Graph read_graph(const string& filename, bool reverse_weight = false) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    // Read number of nodes and edges
    size_t num_nodes = 0, num_edges = 0;
    {
        string first_line;
        if (!getline(infile, first_line)) {
            throw runtime_error("Failed to read the first line for node and edge counts.");
        }
        istringstream iss_first(first_line);
        if (!(iss_first >> num_nodes >> num_edges)) {
            throw runtime_error("Failed to parse the number of nodes and edges.");
        }
    }

    Graph G(num_nodes);
    size_t edge_count = 0;
    string line;
    while (edge_count < num_edges && getline(infile, line)) {
        istringstream iss(line);
        unsigned u, v;
        double edge_polarity;

        if (!(iss >> u >> v >> edge_polarity)) {
            throw runtime_error("Failed to parse edge data on line: " + line);
        }

        // Check if this is a self-loop
        if (u == v) {
            // Store self-loop info in the node
            G[u].has_self_loop = true;
            G[u].self_loop_polarity = edge_polarity * (reverse_weight ? -1.0 : 1.0);
        } else {
            add_edge(u, v, EdgeProperty{edge_polarity * (reverse_weight ? -1.0 : 1.0)}, G);
        }

        edge_count++;
    }

    infile.close();
    if (edge_count != num_edges) {
        throw runtime_error(
            "Number of edges read (" + to_string(edge_count) +
            ") does not match specified (" + to_string(num_edges) + ")");
    }

    return G;
}

/*
 * =========================================================
 * Helper function to compute positive edge sum for each node
 * =========================================================
 */
void compute_positive_edge_sums(const Graph& G, const vector<bool>& is_removed, vector<double>& pos_weights) {
    fill(pos_weights.begin(), pos_weights.end(), 0.0);
    
    // Add contributions from edges
    for (auto e_it = edges(G); e_it.first != e_it.second; ++e_it.first) {
        Vertex u = source(*e_it.first, G);
        Vertex v = target(*e_it.first, G);
        
        // Skip if either endpoint is removed
        if (is_removed[u] || is_removed[v]) {
            continue;
        }
        
        if (G[*e_it.first].edge_polarity > 0) {
            pos_weights[u] += G[*e_it.first].edge_polarity;
            pos_weights[v] += G[*e_it.first].edge_polarity;
        }
    }
    
    // Add contributions from self-loops
    for (Vertex v = 0; v < num_vertices(G); ++v) {
        if (!is_removed[v] && G[v].has_self_loop && G[v].self_loop_polarity > 0) {
            pos_weights[v] += G[v].self_loop_polarity;
        }
    }
}

/*
 * =========================================================
 * Reset node properties for a fresh run
 * =========================================================
 */
inline void reset_node_properties(Graph& G, const vector<bool>& is_removed) {
    for (Vertex v = 0; v < num_vertices(G); ++v) {
        if (!is_removed[v]) {
            G[v].status = Status::Out;
            G[v].priority_key = 0.0;
            G[v].in_neighbor_count = 0;
        }
    }
}

/*
 * =========================================================
 * Eccentricity-based Greedy Algorithm (Modified)
 * =========================================================
 * Now takes is_removed to avoid removed nodes without creating a subgraph
 */
pair<FibHeap, double> ecc_greedy(Graph& G, const vector<bool>& is_removed, unsigned max_neg_count = 100) {
    // Reset node properties
    reset_node_properties(G, is_removed);
    
    // Define null vertex
    Vertex null_v = Traits::null_vertex();

    // Find the node id (index in the graph) with the maximum positive weight sum
    vector<double> pos_weights(num_vertices(G));
    compute_positive_edge_sums(G, is_removed, pos_weights);
    
    Vertex node_promising = null_v;
    double max_weight = -numeric_limits<double>::infinity();
    for (Vertex v = 0; v < num_vertices(G); ++v) {
        if (!is_removed[v] && pos_weights[v] > max_weight) {
            max_weight = pos_weights[v];
            node_promising = v;
        }
    }
    
    if (node_promising == null_v) {
        // No valid starting node
        return {FibHeap(), -numeric_limits<double>::infinity()};
    }

    // ============== Basic structures for the iteration ==============

    double polarity_sum = 0.0;
    FibHeap selected_heap, to_select_heap;
    // We'll store handles for each vertex in a single vector
    vector<FibHeap::handle_type> handles(num_vertices(G));

    // Initialize node_promising
    G[node_promising].status = Status::Fringe;
    // Use pre-stored self-loop polarity
    G[node_promising].priority_key = G[node_promising].has_self_loop
                                       ? G[node_promising].self_loop_polarity
                                       : 0.0;

    handles[node_promising] = to_select_heap.push(
        {G[node_promising].priority_key, node_promising}
    );

    // ============== Main loop variables ==============
    Vertex next_node = node_promising;
    double max_f = -numeric_limits<double>::infinity();
    unsigned neg_count = 0;
    // Best-known configuration if we find a better objective
    FibHeap best_selected_heap = selected_heap;

    // Continue while we have a valid next node and haven't exceeded max_neg_count
    while (next_node != null_v && neg_count < max_neg_count) {
        auto status = G[next_node].status;

        // ============== If node is "fringe" → move it to "in" ==============
        if (status == Status::Fringe) {
            G[next_node].status = Status::In;

            // Pop from the "to_select" heap
            auto item = to_select_heap.top();
            to_select_heap.pop();

            // Push into the "selected" heap (flip sign for internal tracking)
            handles[next_node] = selected_heap.push(
                {-item.priority_key, item.vertex}
            );
            G[next_node].priority_key = -item.priority_key;

            polarity_sum += item.priority_key;

            // Update neighbors
            for (auto oe = out_edges(next_node, G); oe.first != oe.second; ++oe.first) {
                Vertex neighbor = target(*oe.first, G);
                
                // Skip if neighbor is removed or is a self-loop
                if (is_removed[neighbor] || neighbor == next_node) continue;

                double edge_polarity = G[*oe.first].edge_polarity;
                G[neighbor].in_neighbor_count += 1;

                if (G[neighbor].status == Status::Out) {
                    // Move out → fringe
                    G[neighbor].status = Status::Fringe;
                    // Combine edge polarity + possible neighbor self-loop
                    double extra = G[neighbor].has_self_loop ? G[neighbor].self_loop_polarity : 0.0;
                    G[neighbor].priority_key = edge_polarity + extra;

                    handles[neighbor] = to_select_heap.push(
                        {G[neighbor].priority_key, neighbor}
                    );
                }
                else if (G[neighbor].status == Status::Fringe) {
                    // Increase priority
                    G[neighbor].priority_key += edge_polarity;
                    to_select_heap.update(
                        handles[neighbor],
                        {G[neighbor].priority_key, neighbor}
                    );
                }
                else if (G[neighbor].status == Status::In) {
                    // Decrease priority
                    G[neighbor].priority_key -= edge_polarity;
                    selected_heap.update(
                        handles[neighbor],
                        {G[neighbor].priority_key, neighbor}
                    );
                }
            }
        }
        // ============== If node is "in" → move it to "fringe" ==============
        else if (status == Status::In) {
            G[next_node].status = Status::Fringe;

            // Pop from "selected" heap
            auto item = selected_heap.top();
            selected_heap.pop();

            // Move it to "to_select" (flip sign)
            handles[next_node] = to_select_heap.push(
                {-item.priority_key, item.vertex}
            );
            G[next_node].priority_key = -item.priority_key;

            polarity_sum += item.priority_key;

            // Update neighbors
            for (auto oe = out_edges(next_node, G); oe.first != oe.second; ++oe.first) {
                Vertex neighbor = target(*oe.first, G);
                
                // Skip if neighbor is removed or is a self-loop
                if (is_removed[neighbor] || neighbor == next_node) continue;

                double edge_polarity = G[*oe.first].edge_polarity;
                G[neighbor].in_neighbor_count -= 1;

                if (G[neighbor].status == Status::Fringe) {
                    // Possibly move fringe → out if in_neighbor_count == 0
                    if (G[neighbor].in_neighbor_count == 0) {
                        G[neighbor].status = Status::Out;
                        G[neighbor].priority_key = 0.0;
                        to_select_heap.erase(handles[neighbor]);
                        handles[neighbor] = FibHeap::handle_type();
                    } else {
                        // Decrease priority
                        G[neighbor].priority_key -= edge_polarity;
                        to_select_heap.update(
                            handles[neighbor],
                            {G[neighbor].priority_key, neighbor}
                        );
                    }
                }
                else if (G[neighbor].status == Status::In) {
                    // Increase priority
                    G[neighbor].priority_key += edge_polarity;
                    selected_heap.update(
                        handles[neighbor],
                        {G[neighbor].priority_key, neighbor}
                    );
                }
            }
        }
        else {
            // We only expect "out", "fringe", or "in" statuses
            throw invalid_argument("Invalid node status encountered.");
        }

        // ============== Compute the objective function ==============
        unsigned num_selected_now = selected_heap.size();

        double value_old = 0.0;
        if (num_selected_now > 0) {
            value_old = polarity_sum / static_cast<double>(num_selected_now);
        }
        if (value_old >= max_f) {
            max_f = value_old;
        }

        // ============== Compute marginal gains for top of each heap ==============
        vector<pair<double, Vertex>> marginal_gains;
        vector<unsigned> addition_idx;
        marginal_gains.reserve(2);

        // Evaluate removing from selected_heap
        if (!selected_heap.empty()) {
            auto top_item = selected_heap.top();
            unsigned total_minus_1 = num_selected_now - 1;
            double new_sum = (num_selected_now > 1)
                            ? (polarity_sum + top_item.priority_key) / static_cast<double>(total_minus_1)
                            : 0.0;

            double mg = new_sum - value_old;
            marginal_gains.emplace_back(mg, top_item.vertex);
        }

        // Evaluate adding from to_select_heap
        if (!to_select_heap.empty()) {
            auto top_item = to_select_heap.top();
            unsigned total_plus_1 = num_selected_now + 1;
            double new_sum = (polarity_sum + top_item.priority_key) / static_cast<double>(total_plus_1);

            double mg = new_sum - value_old;
            marginal_gains.emplace_back(mg, top_item.vertex);
            addition_idx.push_back(marginal_gains.size() - 1);
        }

        if (marginal_gains.empty()) {
            next_node = null_v;
        } else {
            // Find the node with the maximum marginal gain
            auto max_mg_it = max_element(
                marginal_gains.begin(), marginal_gains.end(),
                [](auto& a, auto& b) { return a.first < b.first; }
            );
            double max_mg = max_mg_it->first;
            Vertex max_mg_node = max_mg_it->second;

            // If the best improvement cannot exceed current best, increment counter
            if ((value_old + max_mg) <= max_f) {
                neg_count++;
                if (addition_idx.empty()) {
                    next_node = null_v;
                } else {
                    // Among additions, pick the best one
                    double best_add = -numeric_limits<double>::infinity();
                    Vertex candidate_node = null_v;
                    for (auto idx : addition_idx) {
                        if (marginal_gains[idx].first > best_add) {
                            best_add = marginal_gains[idx].first;
                            candidate_node = marginal_gains[idx].second;
                        }
                    }
                    next_node = candidate_node;
                }
            } else {
                // We can still improve upon best_f
                neg_count = 0;
                next_node = max_mg_node;
            }

            // Check if we have a new best
            if (value_old >= max_f) {
                if (max_mg <= 0 || next_node == null_v) {
                    best_selected_heap = selected_heap; // deep copy the heap
                }
            }
        }
    }


    // ============== Peeling phase: remove nodes to find maximum density ==============
    // Check initial density of current selected_heap
    if (!selected_heap.empty()) {
        double initial_density = polarity_sum / static_cast<double>(selected_heap.size());
        if (initial_density > max_f) {
            max_f = initial_density;
            best_selected_heap = selected_heap;
        }
    }
    
    // Peel nodes one by one
    while (!selected_heap.empty()) {
        // Remove the node with smallest priority (most negative impact)
        auto top_item = selected_heap.top();
        selected_heap.pop();
        
        // Update polarity sum (remember priority is negated)
        polarity_sum += top_item.priority_key;
        
        // Calculate new density
        if (!selected_heap.empty()) {
            double current_density = polarity_sum / static_cast<double>(selected_heap.size());
            
            // Check if this is the best density found
            if (current_density > max_f) {
                max_f = current_density;
                best_selected_heap = selected_heap;
            }
        }
    }
    
    // Return the configuration with maximum density found during peeling
    return {best_selected_heap, max_f};
}

/*
 * =========================================================
 * Multi Local Optima Search
 * =========================================================
 * Finds multiple local optima and returns the one with largest density
 */
struct SubgraphResult {
    vector<Vertex> nodes;
    double density;
};

SubgraphResult find_multi_local_optima(Graph& G, unsigned max_neg_count = 100, unsigned max_iterations = 10) {
    double global_max_density = -numeric_limits<double>::infinity();
    vector<Vertex> best_subgraph;
    
    // Track which nodes have been removed across all iterations
    vector<bool> is_removed(num_vertices(G), false);
    
    // Reusable buffer for positive weights
    vector<double> pos_weights(num_vertices(G));
    
    for (unsigned iter = 0; iter < max_iterations; ++iter) {
        // Compute positive edge sums for current active nodes
        compute_positive_edge_sums(G, is_removed, pos_weights);
        
        // Find maximum positive edge sum among non-removed nodes
        double max_pos_weight = -numeric_limits<double>::infinity();
        for (Vertex v = 0; v < num_vertices(G); ++v) {
            if (!is_removed[v] && pos_weights[v] > max_pos_weight) {
                max_pos_weight = pos_weights[v];
            }
        }
        
        // Stop if no nodes have positive edge sum greater than current best density
        if (max_pos_weight <= global_max_density) {
            break;
        }
        
        // Find a local densest subgraph (pass is_removed to avoid creating subgraph)
        auto result = ecc_greedy(G, is_removed, max_neg_count);
        double current_density = result.second;
        
        // If no valid result, break
        if (current_density == -numeric_limits<double>::infinity()) {
            break;
        }
        
        // Extract nodes from the heap
        set<Vertex> current_subgraph_set;
        vector<Vertex> current_subgraph_vec;
        for (auto it = result.first.ordered_begin(); it != result.first.ordered_end(); ++it) {
            current_subgraph_set.insert(it->vertex);
            current_subgraph_vec.push_back(it->vertex);
        }
        
        // Update global best if this is better
        if (current_density > global_max_density) {
            global_max_density = current_density;
            best_subgraph = current_subgraph_vec;
        }
        
        // Mark nodes for removal: those in current subgraph + those with pos_weight < current_density
        size_t nodes_marked = 0;
        for (Vertex v = 0; v < num_vertices(G); ++v) {
            if (!is_removed[v]) {
                if (current_subgraph_set.find(v) != current_subgraph_set.end() || 
                    pos_weights[v] < current_density) {
                    is_removed[v] = true;
                    nodes_marked++;
                }
            }
        }
        
        // If no nodes marked, we're done
        if (nodes_marked == 0) {
            break;
        }
    }
    
    return {best_subgraph, global_max_density};
}

/*
 * =========================================================
 * Main Function
 * =========================================================
 */
int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 7) {
        cerr << "Usage: " << argv[0] << " <filename> <output_filename> <reverse_weight> <max #neg-steps> [num_its] [max_local_optima]" << endl;
        return EXIT_FAILURE;
    }
    string filename = argv[1];
    string output_filename = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    unsigned max_neg = stoul(argv[4], nullptr);
    unsigned max_local_optima = (argc >= 6) ? stoul(argv[5], nullptr) : 10;
    unsigned num_its = (argc >= 7) ? stoul(argv[6], nullptr) : 1;

    try {
        // Read the graph from the edge list file (only once)
        Graph G = read_graph(filename, reverse_weight);

        // Variables to store results from first iteration (for output consistency)
        vector<Vertex> first_selected;
        double first_density;
        // Variables to accumulate timing results
        double total_elapsed = 0.0;

        for (unsigned iteration = 0; iteration < num_its; ++iteration) {
            // Time the multi-local optima search
            auto start = chrono::high_resolution_clock::now();
            auto result = find_multi_local_optima(G, max_neg, max_local_optima);
            auto end = chrono::high_resolution_clock::now();
            double elapsed = static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(end - start).count()) / 1e9;
            total_elapsed += elapsed;

            // Store first iteration results for output
            if (iteration == 0) {
                first_selected = result.nodes;
                first_density = result.density;
            }
        }

        // Calculate and output averages
        double avg_time = total_elapsed / num_its;

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
        for (size_t i = 0; i < first_selected.size(); i++) {
            if (i > 0) json_file << ", ";
            json_file << first_selected[i];
        }
        json_file << "],\n";
        json_file << "  \"size\": " << first_selected.size() << ",\n";
        json_file << "  \"density\": " << first_density << "\n";
        json_file << "}\n";

        json_file.close();
        // cout << "Results written to " << output_filename << endl;

    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}