// Combined Dinkelbach-QPBO-MIP Strategy with GNDS initialization
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <set>
#include <boost/graph/adjacency_list.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <gurobi_c++.h>
#include "QPBO/QPBO.h"

using namespace std;
using namespace boost;

// =========================================================
// Graph type definitions (unified from both files)
// =========================================================
enum class Status {
    Out,
    Fringe,
    In
};

struct VertexProperty {
    // Self-loop info
    bool has_self_loop = false;
    double self_loop_polarity = 0.0;
    
    // GNDS-specific properties
    Status status = Status::Out;
    double priority_key = 0.0;
    unsigned in_neighbor_count = 0;
};

struct EdgeProperty {
    double polarity;
};

typedef adjacency_list<vecS, vecS, undirectedS, VertexProperty, EdgeProperty> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph> Traits;

struct DensestSubgraphResult {
    vector<Vertex> nodes;
    double density;
};

// =========================================================
// Graph reading function
// =========================================================
Graph read_graph(const string& filename, bool reverse_weight = false) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

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

        if (u == v) {
            G[u].has_self_loop = true;
            G[u].self_loop_polarity = edge_polarity * (reverse_weight ? -1.0 : 1.0);
        } else {
            auto e = add_edge(u, v, G).first;
            G[e].polarity = edge_polarity * (reverse_weight ? -1.0 : 1.0);
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

// =========================================================
// Density computation
// =========================================================
double compute_density(const Graph& G, const vector<Vertex>& nodes) {
    if (nodes.empty()) return 0;
    
    vector<bool> in_set(num_vertices(G), false);
    for (auto v : nodes) {
        in_set[v] = true;
    }
    
    double total_weight = 0.0;
    
    // Add edge weights
    auto edge_range = edges(G);
    for (auto it = edge_range.first; it != edge_range.second; ++it) {
        auto u = source(*it, G);
        auto v = target(*it, G);
        if (in_set[u] && in_set[v]) {
            total_weight += G[*it].polarity;
        }
    }
    
    // Add self-loop weights
    for (auto v : nodes) {
        if (G[v].has_self_loop) {
            total_weight += G[v].self_loop_polarity;
        }
    }
    
    return total_weight / nodes.size();
}

// =========================================================
// GNDS Helper Functions
// =========================================================
struct PriorityTuple {
    double priority_key;
    Vertex vertex;

    bool operator<(const PriorityTuple& other) const {
        return
            (priority_key < other.priority_key) ||
            (priority_key == other.priority_key && vertex < other.vertex);
    }
};

using FibHeap = heap::fibonacci_heap<PriorityTuple>;

void compute_positive_edge_sums(const Graph& G, const vector<bool>& is_removed, vector<double>& pos_weights) {
    fill(pos_weights.begin(), pos_weights.end(), 0.0);
    
    for (auto e_it = edges(G); e_it.first != e_it.second; ++e_it.first) {
        Vertex u = source(*e_it.first, G);
        Vertex v = target(*e_it.first, G);
        
        if (is_removed[u] || is_removed[v]) {
            continue;
        }
        
        if (G[*e_it.first].polarity > 0) {
            pos_weights[u] += G[*e_it.first].polarity;
            pos_weights[v] += G[*e_it.first].polarity;
        }
    }
    
    for (Vertex v = 0; v < num_vertices(G); ++v) {
        if (!is_removed[v] && G[v].has_self_loop) {
            pos_weights[v] += G[v].self_loop_polarity;
        }
    }
}

inline void reset_node_properties(Graph& G, const vector<bool>& is_removed) {
    for (Vertex v = 0; v < num_vertices(G); ++v) {
        if (!is_removed[v]) {
            G[v].status = Status::Out;
            G[v].priority_key = 0.0;
            G[v].in_neighbor_count = 0;
        }
    }
}

// =========================================================
// Eccentricity-based Greedy Algorithm
// =========================================================
pair<FibHeap, double> ecc_greedy(Graph& G, const vector<bool>& is_removed, unsigned max_neg_count = 100) {
    reset_node_properties(G, is_removed);
    
    Vertex null_v = Traits::null_vertex();

    vector<double> pos_weights(num_vertices(G));
    compute_positive_edge_sums(G, is_removed, pos_weights);
    
    Vertex node_promising = null_v;
    double max_weight = -numeric_limits<double>::infinity();
    for (Vertex v = 0; v < num_vertices(G); ++v) {
        if (!is_removed[v] && pos_weights[v] > max_weight && pos_weights[v] > 0.0) {
            max_weight = pos_weights[v];
            node_promising = v;
        }
    }
    
    if (node_promising == null_v) {
        return {FibHeap(), -numeric_limits<double>::infinity()};
    }

    double polarity_sum = 0.0;
    FibHeap selected_heap, to_select_heap;
    vector<FibHeap::handle_type> handles(num_vertices(G));

    G[node_promising].status = Status::Fringe;
    G[node_promising].priority_key = G[node_promising].has_self_loop
                                       ? G[node_promising].self_loop_polarity
                                       : 0.0;

    handles[node_promising] = to_select_heap.push(
        {G[node_promising].priority_key, node_promising}
    );

    Vertex next_node = node_promising;
    double max_f = -numeric_limits<double>::infinity();
    unsigned neg_count = 0;
    FibHeap best_selected_heap = selected_heap;

    while (next_node != null_v && neg_count < max_neg_count) {
        auto status = G[next_node].status;

        if (status == Status::Fringe) {
            G[next_node].status = Status::In;

            auto item = to_select_heap.top();
            to_select_heap.pop();

            handles[next_node] = selected_heap.push(
                {-item.priority_key, item.vertex}
            );
            G[next_node].priority_key = -item.priority_key;

            polarity_sum += item.priority_key;

            for (auto oe = out_edges(next_node, G); oe.first != oe.second; ++oe.first) {
                Vertex neighbor = target(*oe.first, G);
                if (is_removed[neighbor] || neighbor == next_node) continue;
                
                double edge_polarity = G[*oe.first].polarity;
                G[neighbor].in_neighbor_count += 1;

                if (G[neighbor].status == Status::Out) {
                    G[neighbor].status = Status::Fringe;
                    double extra = G[neighbor].has_self_loop ? G[neighbor].self_loop_polarity : 0.0;
                    G[neighbor].priority_key = edge_polarity + extra;

                    handles[neighbor] = to_select_heap.push(
                        {G[neighbor].priority_key, neighbor}
                    );  
                }
                else if (G[neighbor].status == Status::Fringe) {
                    G[neighbor].priority_key += edge_polarity;
                    to_select_heap.update(
                        handles[neighbor],
                        {G[neighbor].priority_key, neighbor}
                    );
                }
                else if (G[neighbor].status == Status::In) {
                    G[neighbor].priority_key -= edge_polarity;
                    selected_heap.update(
                        handles[neighbor],
                        {G[neighbor].priority_key, neighbor}
                    );
                }
            }
        }
        else if (status == Status::In) {
            G[next_node].status = Status::Fringe;

            auto item = selected_heap.top();
            selected_heap.pop();

            handles[next_node] = to_select_heap.push(
                {-item.priority_key, item.vertex}
            );
            G[next_node].priority_key = -item.priority_key;

            polarity_sum += item.priority_key;

            for (auto oe = out_edges(next_node, G); oe.first != oe.second; ++oe.first) {
                Vertex neighbor = target(*oe.first, G);
                if (is_removed[neighbor] || neighbor == next_node) continue;

                double edge_polarity = G[*oe.first].polarity;
                G[neighbor].in_neighbor_count -= 1;

                if (G[neighbor].status == Status::Fringe) {
                    if (G[neighbor].in_neighbor_count == 0) {
                        G[neighbor].status = Status::Out;
                        G[neighbor].priority_key = 0.0;
                        to_select_heap.erase(handles[neighbor]);
                        handles[neighbor] = FibHeap::handle_type();
                    } else {
                        G[neighbor].priority_key -= edge_polarity;
                        to_select_heap.update(
                            handles[neighbor],
                            {G[neighbor].priority_key, neighbor}
                        );
                    }
                }
                else if (G[neighbor].status == Status::In) {
                    G[neighbor].priority_key += edge_polarity;
                    selected_heap.update(
                        handles[neighbor],
                        {G[neighbor].priority_key, neighbor}
                    );
                }
            }
        }

        unsigned num_selected_now = selected_heap.size();

        double value_old = 0.0;
        if (num_selected_now > 0) {
            value_old = polarity_sum / static_cast<double>(num_selected_now);
        }
        if (value_old >= max_f) {
            max_f = value_old;
        }

        double best_mg = -numeric_limits<double>::infinity();
        Vertex best_node = null_v;
        bool best_is_addition = false;

        if (!selected_heap.empty()) {
            auto top_item = selected_heap.top();
            double new_sum = (num_selected_now > 1)
                            ? (polarity_sum + top_item.priority_key) / static_cast<double>(num_selected_now - 1)
                            : 0.0;
            double mg = new_sum - value_old;
            if (mg > best_mg) {
                best_mg = mg;
                best_node = top_item.vertex;
            }
        }

        if (!to_select_heap.empty()) {
            auto top_item = to_select_heap.top();
            double new_sum = (polarity_sum + top_item.priority_key) / static_cast<double>(num_selected_now + 1);
            double mg = new_sum - value_old;
            if (mg > best_mg) {
                best_mg = mg;
                best_node = top_item.vertex;
                best_is_addition = true;
            }
        }

        if (best_node == null_v) {
            next_node = null_v;
        } else {
            if ((value_old + best_mg) <= max_f || next_node == best_node) {
                neg_count++;
                next_node = best_is_addition ? best_node : (to_select_heap.empty() ? null_v : to_select_heap.top().vertex);
            } else {
                neg_count = 0;
                next_node = best_node;
            }

            if (value_old >= max_f) {
                if (best_mg <= 0 || next_node == null_v) {
                    best_selected_heap = selected_heap;
                }
            }
        }
    }

    // Peeling phase
    while (!selected_heap.empty()) {
        auto top_item = selected_heap.top();
        selected_heap.pop();
        G[top_item.vertex].status = Status::Out;

        for (auto e = out_edges(top_item.vertex, G); e.first != e.second; ++e.first) {
            Vertex neighbor = target(*e.first, G);
            if (is_removed[neighbor] || G[neighbor].status != Status::In) continue;

            double edge_polarity = G[*e.first].polarity;
            G[neighbor].priority_key += edge_polarity;
            selected_heap.update(
                handles[neighbor],
                {G[neighbor].priority_key, neighbor}
            );
        }
        
        polarity_sum += top_item.priority_key;
        
        if (!selected_heap.empty()) {
            double current_density = polarity_sum / static_cast<double>(selected_heap.size());
            
            if (current_density > max_f) {
                max_f = current_density;
                best_selected_heap = selected_heap;
            }
        }
    }
    
    return {best_selected_heap, max_f};
}

// =========================================================
// Multi Local Optima Search (GNDS initialization)
// =========================================================
DensestSubgraphResult find_multi_local_optima(Graph& G, unsigned max_neg_count = 100, unsigned max_local_optima = 10) {
    double global_max_density = -numeric_limits<double>::infinity();
    vector<Vertex> best_subgraph;
    
    vector<bool> is_removed(num_vertices(G), false);
    vector<double> pos_weights(num_vertices(G));
    
    for (unsigned iter = 0; iter < max_local_optima; ++iter) {
        compute_positive_edge_sums(G, is_removed, pos_weights);
        
        double max_pos_weight = -numeric_limits<double>::infinity();
        for (Vertex v = 0; v < num_vertices(G); ++v) {
            if (!is_removed[v] && pos_weights[v] > max_pos_weight) {
                max_pos_weight = pos_weights[v];
            }
        }
        
        if (max_pos_weight <= global_max_density) {
            break;
        }
        
        auto result = ecc_greedy(G, is_removed, max_neg_count);

        double current_density = result.second;
        
        if (current_density == -numeric_limits<double>::infinity()) {
            break;
        }
        
        set<Vertex> current_subgraph_set;
        vector<Vertex> current_subgraph_vec;
        for (auto it = result.first.ordered_begin(); it != result.first.ordered_end(); ++it) {
            current_subgraph_set.insert(it->vertex);
            current_subgraph_vec.push_back(it->vertex);
        }
        
        if (current_density > global_max_density) {
            global_max_density = current_density;
            best_subgraph = current_subgraph_vec;
        }
        
        size_t nodes_marked = 0;
        for (Vertex v = 0; v < num_vertices(G); ++v) {
            if (!is_removed[v]) {
                if (current_subgraph_set.find(v) != current_subgraph_set.end() || 
                    pos_weights[v] < global_max_density) {
                    is_removed[v] = true;
                    nodes_marked++;
                }
            }
        }
        
        if (nodes_marked == 0) {
            break;
        }
    }
    
    return {best_subgraph, global_max_density};
}

// =========================================================
// QPBO Functions
// =========================================================
struct QPBOResult {
    vector<int> labels;
    vector<Vertex> fixed_in;
    vector<Vertex> fixed_out;
    vector<Vertex> undecided;
};

QPBOResult run_qpbo(const Graph& G, double lambda, bool improve = false) {
    size_t n = num_vertices(G);
    
    typedef double REAL;
    std::unique_ptr<QPBO<REAL>> qpbo(new QPBO<REAL>(n, num_edges(G)));
    qpbo->AddNode(n);
    
    for (size_t i = 0; i < n; i++) {
        REAL E0 = 0.0;
        REAL E1 = lambda;
        
        if (G[i].has_self_loop) {
            E1 -= G[i].self_loop_polarity;
        }
        
        qpbo->AddUnaryTerm(i, E0, E1);
    }
    
    auto edge_range = edges(G);
    for (auto it = edge_range.first; it != edge_range.second; ++it) {
        auto u = source(*it, G);
        auto v = target(*it, G);
        double w = G[*it].polarity;
        
        qpbo->AddPairwiseTerm(u, v, 0, 0, 0, -w);
    }
    
    qpbo->Solve();
    qpbo->ComputeWeakPersistencies();

    if (improve) {
        qpbo->Improve();
    }
    
    QPBOResult result;
    result.labels.resize(n);
    
    for (size_t i = 0; i < n; i++) {
        int label = qpbo->GetLabel(i);
        if (label == 0) {
            result.labels[i] = 0;
            result.fixed_out.push_back(i);
        } else if (label == 1) {
            result.labels[i] = 1;
            result.fixed_in.push_back(i);
        } else {
            result.labels[i] = -1;
            result.undecided.push_back(i);
        }
    }
    
    return result;
}

// =========================================================
// MIP Solver
// =========================================================
vector<Vertex> solve_mip_on_undecided(const Graph& G, const QPBOResult& qpbo_result, double lambda) {
    try {
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 0);
        env.start();
        GRBModel model = GRBModel(env);
        
        size_t n = num_vertices(G);
        std::unordered_map<size_t, GRBVar> undecided_vars;

        for (size_t i = 0; i < n; i++) {
            if (qpbo_result.labels[i] == -1) {
                undecided_vars[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
            }
        }

        GRBQuadExpr obj = 0.0;

        for (size_t i = 0; i < n; i++) {
            double coeff = lambda;
            if (G[i].has_self_loop) {
                coeff -= G[i].self_loop_polarity;
            }
            
            if (qpbo_result.labels[i] == 1) {
                obj += coeff * 1.0;
            } else if (qpbo_result.labels[i] == -1) {
                obj += coeff * undecided_vars[i];
            }
        }

        auto edge_range = edges(G);
        for (auto it = edge_range.first; it != edge_range.second; ++it) {
            auto u = source(*it, G);
            auto v = target(*it, G);
            
            double w = G[*it].polarity;
            
            int label_u = qpbo_result.labels[u];
            int label_v = qpbo_result.labels[v];
            
            if (label_u == -1 && label_v == -1) {
                obj += -w * undecided_vars[u] * undecided_vars[v];
            } else if (label_u == -1 && label_v != -1) {
                obj += -w * label_v * undecided_vars[u];
            } else if (label_v == -1 && label_u != -1) {
                obj += -w * label_u * undecided_vars[v];
            } else {
                obj += -w * label_u * label_v;
            }
        }
        
        model.setObjective(obj, GRB_MINIMIZE);
        model.set(GRB_DoubleParam_TimeLimit, 300.0);
        model.optimize();
        
        vector<Vertex> selected;
        if (model.get(GRB_IntAttr_SolCount) > 0) {
            for (size_t i = 0; i < n; i++) {
                if (qpbo_result.labels[i] == 1) {
                    selected.push_back(i);
                } else if (qpbo_result.labels[i] == -1) {
                    if (undecided_vars.count(i) && undecided_vars[i].get(GRB_DoubleAttr_X) > 0.5) {
                        selected.push_back(i);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                if (qpbo_result.labels[i] == 1) {
                    selected.push_back(i);
                }
            }
        }
        
        return selected;
        
    } catch (GRBException& e) {
        cerr << "Gurobi exception: " << e.getMessage() << endl;
        vector<Vertex> selected;
        for (size_t i = 0; i < num_vertices(G); i++) {
            if (qpbo_result.labels[i] == 1) {
                selected.push_back(i);
            }
        }
        return selected;
    }
}

// =========================================================
// Dinkelbach algorithm with GNDS initialization
// =========================================================
DensestSubgraphResult gnds_qpbo_mip_ud(Graph& G, double step_size = 1.1, unsigned max_iterations = 10, double epsilon = 1e-3, 
                               unsigned gnds_max_neg = 100, unsigned gnds_max_optima = 10) {
    size_t n = num_vertices(G);
    
    // Use GNDS to compute initial lambda and best solution
    auto gnds_result = find_multi_local_optima(G, gnds_max_neg, gnds_max_optima);
    
    vector<Vertex> best_solution = gnds_result.nodes;
    double best_density = gnds_result.density;

    // if qpbo_result.undecided is empty and fixed_in is empty, we can directly return result found by GNDS as Optimal
    QPBOResult qpbo_result_pre = run_qpbo(G, best_density, false);
    if (qpbo_result_pre.undecided.empty() && qpbo_result_pre.fixed_in.empty()) {
        return {best_solution, best_density};
    }

    // otherwise, find an upper bound which always makes the QPBO label all nodes to fixed_out
    double lambda_ub = best_density * step_size;
    while (true) {
        QPBOResult qpbo_result_step = run_qpbo(G, lambda_ub, false);
        if (qpbo_result_step.undecided.empty() && qpbo_result_step.fixed_in.empty()) {
            break;
        }
        lambda_ub *= step_size;
    }

    // find optimal solution with binary search on lambda
    double lambda = (best_density + lambda_ub) / 2.0;
    for (int iter = 0; iter < max_iterations; iter++) {
        QPBOResult qpbo_result = run_qpbo(G, lambda, false);
        vector<Vertex> current_solution = qpbo_result.undecided.empty() ? qpbo_result.fixed_in : solve_mip_on_undecided(G, qpbo_result, lambda);

        if (current_solution.empty()) {
            lambda_ub = lambda;
        } else {
            double density = compute_density(G, current_solution);
            if (density >= best_density) {
                best_density = density;
                best_solution = current_solution;
            } else {
                cout << "Should not reach here!" << endl;
            }
        }
        lambda = (best_density + lambda_ub) / 2.0;
        cout << "Dinkelbach iteration " << iter + 1 << ": best density = " << best_density 
             << ", lambda_ub = " << lambda_ub << endl;
        if (abs(best_density - lambda_ub) < epsilon) {
            break;
        }
    }
    
    return {best_solution, best_density};
}

// =========================================================
// Main Function
// =========================================================
int main(int argc, char* argv[]) {
    if (argc < 9 || argc > 10) {
        cerr << "Usage: " << argv[0] << " <filename> <output_filename> <reverse_weight> <step_size> "
             << "<dinkelbach_iterations> <epsilon> <gnds_max_neg> <gnds_max_optima> [num_its]" << endl;
        return EXIT_FAILURE;
    }
    string filename = argv[1];
    string output_filename = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    double step_size = stod(argv[4]);
    unsigned dinkelbach_iterations = stoul(argv[5], nullptr);
    double epsilon = stod(argv[6]);
    unsigned gnds_max_neg = stoul(argv[7], nullptr);
    unsigned gnds_max_optima = stoul(argv[8], nullptr);
    unsigned num_its = (argc >= 10) ? stoul(argv[9], nullptr) : 1;

    try {
        Graph G = read_graph(filename, reverse_weight);

        vector<Vertex> first_selected;
        double first_density;
        double total_elapsed = 0.0;

        for (unsigned iteration = 0; iteration < num_its; ++iteration) {
            auto start = chrono::high_resolution_clock::now();
            auto result = gnds_qpbo_mip_ud(G, step_size, dinkelbach_iterations, epsilon, gnds_max_neg, gnds_max_optima);
            auto end = chrono::high_resolution_clock::now();
            double elapsed = static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(end - start).count()) / 1e9;
            total_elapsed += elapsed;

            if (iteration == 0) {
                first_selected = result.nodes;
                first_density = result.density;
            }
        }

        double avg_time = total_elapsed / num_its;

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

    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}