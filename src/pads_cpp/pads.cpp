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
    double polarity = 0.0;
    unsigned polarity_label = 0;
    double promising_value = 0.0;
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
    double promising_value;
    Vertex vertex;

    // We use "less than" so that the largest priority_key is on top.
    bool operator<(const PriorityTuple& other) const {
        return
            (priority_key < other.priority_key) ||
            (priority_key == other.priority_key && promising_value < other.promising_value) ||
            (priority_key == other.priority_key && promising_value == other.promising_value && vertex < other.vertex);
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
Graph read_graph(const string& filename) {
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
    vector node_properties_set(num_nodes, false);

    size_t edge_count = 0;
    string line;
    while (edge_count < num_edges && getline(infile, line)) {
        istringstream iss(line);
        unsigned u, v;
        double polarity_u, polarity_v, edge_polarity;
        unsigned polarity_label_u, polarity_label_v;

        if (!(iss >> u >> polarity_u >> polarity_label_u
                  >> v >> polarity_v >> polarity_label_v
                  >> edge_polarity))
        {
            throw runtime_error("Failed to parse edge data on line: " + line);
        }

        // Set node properties if not already set
        if (!node_properties_set[u]) {
            G[u].polarity       = polarity_u;
            G[u].polarity_label = polarity_label_u;
            G[u].promising_value = 0.0;
            node_properties_set[u] = true;
        } else {
            // Optional: consistency check
            if (fabs(G[u].polarity - polarity_u) > 1e-9 ||
                G[u].polarity_label != polarity_label_u)
            {
                throw runtime_error("Inconsistent node properties for node " + to_string(u));
            }
        }

        if (!node_properties_set[v]) {
            G[v].polarity       = polarity_v;
            G[v].polarity_label = polarity_label_v;
            G[v].promising_value = 0.0;
            node_properties_set[v] = true;
        } else {
            if (fabs(G[v].polarity - polarity_v) > 1e-9 ||
                G[v].polarity_label != polarity_label_v)
            {
                throw runtime_error("Inconsistent node properties for node " + to_string(v));
            }
        }

        // Check if this is a self-loop
        if (u == v) {
            // Store self-loop info in the node
            G[u].has_self_loop = true;
            G[u].self_loop_polarity = edge_polarity;
        } else {
            add_edge(u, v, EdgeProperty{edge_polarity}, G);
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
 * Eccentricity-based Greedy Algorithm
 * =========================================================
 *
 * pos = false inverts polarities to simulate negative polarity scenario.
 * theta is a threshold that penalizes the entropy portion in the objective.
 * max_neg_count is a cap on how many times we fail to improve the best found
 * objective before exiting.
 */
vector<Vertex> ecc_greedy(Graph& G, double theta, bool pos = true, unsigned max_neg_count = 100, unsigned num_labels = 5) {
    // Define null vertex
    Vertex null_v = Traits::null_vertex();

    // Find the node_promising among the highest label
    Vertex node_promising = null_v;
    double max_promising = -numeric_limits<double>::infinity();
    for (auto v_it = vertices(G); v_it.first != v_it.second; ++v_it.first) {
        if (G[*v_it.first].polarity > 0 &&
            G[*v_it.first].promising_value > max_promising)
        {
            max_promising = G[*v_it.first].promising_value;
            node_promising = *v_it.first;
        }
    }

    // ============== Basic structures for the iteration ==============

    double polarity_sum = 0.0;
    vector<unsigned> num_selected_by_label(num_labels, 0);

    vector<FibHeap> selected_heaps(num_labels), to_select_heaps(num_labels);

    // We'll store handles for each vertex in a single vector
    vector<FibHeap::handle_type> handles(num_vertices(G));

    // Initialize node_promising
    G[node_promising].status = Status::Fringe;
    // Use pre-stored self-loop polarity
    G[node_promising].priority_key = G[node_promising].has_self_loop
                                       ? G[node_promising].self_loop_polarity
                                       : 0.0;

    unsigned label_prom = G[node_promising].polarity_label;
    handles[node_promising] = to_select_heaps[label_prom].push(
        {G[node_promising].priority_key, G[node_promising].promising_value, node_promising}
    );

    // ============== Main loop variables ==============
    Vertex next_node = node_promising;
    double max_f = -numeric_limits<double>::infinity();
    unsigned neg_count = 0;

    // Best-known configuration if we find a better objective
    vector<FibHeap> best_selected_heaps = selected_heaps;

    // Continue while we have a valid next node and haven't exceeded max_neg_count
    while (next_node != null_v && neg_count < max_neg_count) {
        auto status = G[next_node].status;
        unsigned label = G[next_node].polarity_label;

        // ============== If node is "fringe" → move it to "in" ==============
        if (status == Status::Fringe) {
            G[next_node].status = Status::In;

            // Pop from the "to_select" heap
            auto item = to_select_heaps[label].top();
            to_select_heaps[label].pop();

            // Push into the "selected" heap (flip sign for internal tracking)
            handles[next_node] = selected_heaps[label].push(
                {-item.priority_key, -item.promising_value, item.vertex}
            );
            G[next_node].priority_key = -item.priority_key;

            polarity_sum += item.priority_key;
            num_selected_by_label[label]++;

            // Update neighbors
            for (auto oe = out_edges(next_node, G); oe.first != oe.second; ++oe.first) {
                Vertex neighbor = target(*oe.first, G);
                if (neighbor == next_node) continue; // skip self-loop in adjacency

                double edge_polarity = G[*oe.first].edge_polarity;
                G[neighbor].in_neighbor_count += 1;

                if (G[neighbor].status == Status::Out) {
                    // Move out → fringe
                    G[neighbor].status = Status::Fringe;
                    // Combine edge polarity + possible neighbor self-loop
                    double extra = G[neighbor].has_self_loop ? G[neighbor].self_loop_polarity : 0.0;
                    G[neighbor].priority_key = edge_polarity + extra;

                    handles[neighbor] = to_select_heaps[G[neighbor].polarity_label].push(
                        {G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );
                }
                else if (G[neighbor].status == Status::Fringe) {
                    // Increase priority
                    G[neighbor].priority_key += edge_polarity;
                    to_select_heaps[G[neighbor].polarity_label].update(
                        handles[neighbor],
                        {G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );
                }
                else if (G[neighbor].status == Status::In) {
                    // Decrease priority
                    G[neighbor].priority_key -= edge_polarity;
                    selected_heaps[G[neighbor].polarity_label].update(
                        handles[neighbor],
                        {G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );
                }
                else {
                    throw invalid_argument("Invalid neighbor status encountered.");
                }
            }
        }
        // ============== If node is "in" → move it to "fringe" ==============
        else if (status == Status::In) {
            G[next_node].status = Status::Fringe;

            // Pop from "selected" heap
            auto item = selected_heaps[label].top();
            selected_heaps[label].pop();

            // Move it to "to_select" (flip sign)
            handles[next_node] = to_select_heaps[label].push(
                {-item.priority_key, -item.promising_value, item.vertex}
            );
            G[next_node].priority_key = -item.priority_key;

            polarity_sum += item.priority_key;
            num_selected_by_label[label]--;

            // Update neighbors
            for (auto oe = out_edges(next_node, G); oe.first != oe.second; ++oe.first) {
                Vertex neighbor = target(*oe.first, G);
                if (neighbor == next_node) continue; // skip self-loop

                double edge_polarity = G[*oe.first].edge_polarity;
                G[neighbor].in_neighbor_count -= 1;

                if (G[neighbor].status == Status::Fringe) {
                    // Possibly move fringe → out if in_neighbor_count == 0
                    if (G[neighbor].in_neighbor_count == 0) {
                        G[neighbor].status = Status::Out;
                        G[neighbor].priority_key = 0.0;
                        to_select_heaps[G[neighbor].polarity_label].erase(handles[neighbor]);
                        handles[neighbor] = FibHeap::handle_type();
                    } else {
                        // Decrease priority
                        G[neighbor].priority_key -= edge_polarity;
                        to_select_heaps[G[neighbor].polarity_label].update(
                            handles[neighbor],
                            {G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                        );
                    }
                }
                else if (G[neighbor].status == Status::In) {
                    // Increase priority
                    G[neighbor].priority_key += edge_polarity;
                    selected_heaps[G[neighbor].polarity_label].update(
                        handles[neighbor],
                        {G[neighbor].priority_key, G[neighbor].promising_value, neighbor}
                    );
                }
                else {
                    throw invalid_argument("Invalid neighbor status encountered.");
                }
            }
        }
        else {
            // We only expect "out", "fringe", or "in" statuses
            throw invalid_argument("Invalid node status encountered.");
        }

        // ============== Compute the objective function ==============
        unsigned num_selected_now = 0;
        for (auto c : num_selected_by_label) {
            num_selected_now += c;
        }

        double value_old = 0.0;
        if (num_selected_now > 0) {
            // Compute distribution for labels with nonzero counts
            vector<double> label_dist;
            label_dist.reserve(num_labels);
            for (auto c : num_selected_by_label) {
                if (c > 0) {
                    label_dist.push_back(static_cast<double>(c) / static_cast<double>(num_selected_now));
                }
            }

            // Compute entropy
            double entropy = 0.0;
            for (double p : label_dist) {
                entropy -= p * log2(p);
            }
            value_old = (polarity_sum / static_cast<double>(num_selected_now)) - theta * entropy;
        }
        if (value_old >= max_f) {
            max_f = value_old;
        }

        // ============== Compute marginal gains for top of each heap ==============
        vector<pair<double, Vertex>> marginal_gains;
        vector<unsigned> addition_idx;
        marginal_gains.reserve(2 * num_labels); // num_labels × 2 sets each

        // Evaluate removing from each selected_heap, and adding from each to_select_heap
        for (unsigned lbl = 0; lbl < num_labels; ++lbl) {
            // If there's something in selected_heaps[lbl]
            if (!selected_heaps[lbl].empty()) {
                auto top_item = selected_heaps[lbl].top();
                unsigned total_minus_1 = num_selected_now - 1;
                double new_sum = (num_selected_now > 1)
                               ? (polarity_sum + top_item.priority_key) / static_cast<double>(total_minus_1)
                               : 0.0;

                // Recompute label distribution
                vector<unsigned> temp_dist = num_selected_by_label;
                temp_dist[lbl]--;

                vector<double> temp_p;
                for (auto c : temp_dist) {
                    if (c > 0) {
                        temp_p.push_back(static_cast<double>(c) / static_cast<double>(total_minus_1));
                    }
                }
                double temp_entropy = 0.0;
                for (double p : temp_p) {
                    temp_entropy -= p * log2(p);
                }

                double mg = (new_sum - theta * temp_entropy) - value_old;
                marginal_gains.emplace_back(mg, top_item.vertex);
            }

            // If there's something in to_select_heaps[lbl]
            if (!to_select_heaps[lbl].empty()) {
                auto top_item = to_select_heaps[lbl].top();
                unsigned total_plus_1 = num_selected_now + 1;
                double new_sum = (polarity_sum + top_item.priority_key) / static_cast<double>(total_plus_1);

                vector<unsigned> temp_dist = num_selected_by_label;
                temp_dist[lbl]++;

                vector<double> temp_p;
                for (auto c : temp_dist) {
                    if (c > 0) {
                        temp_p.push_back(static_cast<double>(c) / static_cast<double>(total_plus_1));
                    }
                }
                double temp_entropy = 0.0;
                for (double p : temp_p) {
                    temp_entropy -= p * log2(p);
                }

                double mg = (new_sum - theta * temp_entropy) - value_old;
                marginal_gains.emplace_back(mg, top_item.vertex);
                // We'll track which of these are "additions"
                addition_idx.push_back(marginal_gains.size() - 1);
            }
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
                    best_selected_heaps = selected_heaps; // deep copy the heaps
                }
            }
        }
    }

    // ============== Gather final best selection ==============
    set<Vertex> final_selected;
    for (auto& fib : best_selected_heaps) {
        // Each fib item has form { <priority_key>, <promising_val>, <vertex> }
        for (auto it = fib.ordered_begin(); it != fib.ordered_end(); ++it) {
            final_selected.insert(it->vertex);
        }
    }

    // Return the selected vertices
    return {final_selected.begin(), final_selected.end()};
}

/*
 * =========================================================
 * Main Function
 * =========================================================
 */
int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 9) {
        cerr << "Usage: " << argv[0] << " <filename> <theta> <max number of negative steps> [deg_threshold] "
                                        "[num_labels] [prom_skip] [num_its] [sim_aug]" << endl;
        return EXIT_FAILURE;
    }
    string filename = argv[1];
    double theta = strtod(argv[2], nullptr);
    unsigned max_neg = stoul(argv[3], nullptr);
    unsigned deg_threshold = (argc >= 5) ? stoul(argv[4], nullptr) : 0;
    unsigned num_labels = (argc >= 6) ? stoul(argv[5], nullptr) : 5;
    bool prom_skip = (argc >= 7) ? static_cast<bool>(stoul(argv[6]), nullptr) : false;
    unsigned num_its = (argc >= 8) ? stoul(argv[7], nullptr) : 1;
    bool sim_aug = (argc >= 9) ? static_cast<bool>(stoul(argv[8], nullptr)) : false;

    try {
        // Read the graph from the edge list file (only once)
        Graph G = read_graph(filename);

        // Variables to store results from first iteration (for output consistency)
        vector<Vertex> first_selected_pos, first_selected_neg;

        // Variables to accumulate timing results
        double total_elapsed_promising = 0.0;
        double total_elapsed_pos = 0.0;
        double total_elapsed_neg = 0.0;

        for (unsigned iteration = 0; iteration < num_its; ++iteration) {
            // Create fresh copies of the graph for this iteration
            auto G_iteration = G;  // Deep copy - original G is not affected

            auto start_promising = chrono::high_resolution_clock::now();
            // Compute promising_value for each node by iterating over all edges once
            for (auto e_it = edges(G_iteration); e_it.first != e_it.second; ++e_it.first) {
                Vertex s = source(*e_it.first, G_iteration);
                Vertex t = target(*e_it.first, G_iteration);

                // If it's not a self-loop, compute similarity
                if (s != t) {
                    double sim = (2.0 - (sim_aug ? 2 : 1) * fabs(G_iteration[s].polarity - G_iteration[t].polarity)) / 2.0;
                    // if the nodes have opposite opinions, skip them
                    if (prom_skip && G_iteration[s].polarity * G_iteration[t].polarity < 0) {
                        continue;
                    }
                    // add sim only when the other node has degree > deg_threshold
                    if (out_degree(t, G_iteration) > deg_threshold) {
                        G_iteration[s].promising_value += sim;
                    }
                    if (out_degree(s, G_iteration) > deg_threshold) {
                        G_iteration[t].promising_value += sim;
                    }
                }
            }

            // If a node has a self-loop, add self-loop polarity to promising_value
            for (auto v_it = vertices(G_iteration); v_it.first != v_it.second; ++v_it.first) {
                if (G_iteration[*v_it.first].has_self_loop) {
                    G_iteration[*v_it.first].promising_value += 1;//G_iteration[*v_it.first].self_loop_polarity;
                }
            }
            auto end_promising = chrono::high_resolution_clock::now();
            double elapsed_promising = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end_promising - start_promising).count()) / 1e6;
            total_elapsed_promising += elapsed_promising;

            auto G_pos = G_iteration;  // Another deep copy
            auto G_neg = G_iteration;  // Another deep copy

            // Invert node polarities and labels for negative case
            for (auto v_it = vertices(G_neg); v_it.first != v_it.second; ++v_it.first) {
                G_neg[*v_it.first].polarity = -G_neg[*v_it.first].polarity;
                G_neg[*v_it.first].polarity_label = (num_labels - 1) - G_neg[*v_it.first].polarity_label;
            }
            // Invert edge polarities
            for (auto e_it = edges(G_neg); e_it.first != e_it.second; ++e_it.first) {
                G_neg[*e_it.first].edge_polarity = -G_neg[*e_it.first].edge_polarity;
            }
            // Invert stored self-loop polarities
            for (auto v_it = vertices(G_neg); v_it.first != v_it.second; ++v_it.first) {
                if (G_neg[*v_it.first].has_self_loop) {
                    G_neg[*v_it.first].self_loop_polarity = -G_neg[*v_it.first].self_loop_polarity;
                }
            }

            // Time positive case
            auto start_pos = chrono::high_resolution_clock::now();
            vector<Vertex> selected_pos = ecc_greedy(G_pos, theta, true, max_neg, num_labels);
            auto end_pos = chrono::high_resolution_clock::now();
            double elapsed_pos = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end_pos - start_pos).count()) / 1e6;
            total_elapsed_pos += elapsed_pos;

            // Time negative case
            auto start_neg = chrono::high_resolution_clock::now();
            vector<Vertex> selected_neg = ecc_greedy(G_neg, theta, true, max_neg, num_labels);
            auto end_neg = chrono::high_resolution_clock::now();
            double elapsed_neg = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end_neg - start_neg).count()) / 1e6;
            total_elapsed_neg += elapsed_neg;

            // Store first iteration results for output
            if (iteration == 0) {
                first_selected_pos = selected_pos;
                first_selected_neg = selected_neg;
            }
        }

        // Calculate and output averages
        double avg_elapsed_promising = total_elapsed_promising / num_its;
        double avg_elapsed_pos = total_elapsed_pos / num_its;
        double avg_elapsed_neg = total_elapsed_neg / num_its;

        cout << "Time_Pos: " << avg_elapsed_pos << " seconds (averaged over " << num_its << " iterations)" << endl;
        cout << "Nodes_Pos(" << first_selected_pos.size() << "):";
        for (auto n : first_selected_pos) {
            cout << " " << n;
        }
        cout << endl;

        cout << "Time_Neg: " << avg_elapsed_neg << " seconds (averaged over " << num_its << " iterations)" << endl;
        cout << "Nodes_Neg(" << first_selected_neg.size() << "):";
        for (auto n : first_selected_neg) {
            cout << " " << n;
        }
        cout << endl;

        cout << "Total Elapsed Time: " << avg_elapsed_promising + avg_elapsed_pos + avg_elapsed_neg
             << " seconds (averaged over " << num_its << " iterations)" << endl;

    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}