// Dinkelbach-QPBO-MIP Strategy
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
#include <boost/graph/adjacency_list.hpp>
#include <gurobi_c++.h>
#include "QPBO/QPBO.h"

using namespace std;
using namespace boost;

// Graph type definitions
struct EdgeProperty {
    double polarity;
};

struct VertexProperty {
    bool has_self_loop = false;
    double self_loop_polarity = 0.0;
};

typedef adjacency_list<vecS, vecS, undirectedS, VertexProperty, EdgeProperty> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

struct DensestSubgraphResult {
    vector<Vertex> nodes;
    double density;
};

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

// Compute subgraph density including self-loops
double compute_density(const Graph& G, const vector<Vertex>& nodes) {
    if (nodes.empty()) return 0;
    
    vector<bool> in_set(num_vertices(G), false);
    for (auto v : nodes) {
        in_set[v] = true;
    }
    
    double total_weight = 0.0;
    
    // Add edge weights (count each undirected edge once)
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

// Run QPBO to get partial assignment for Dinkelbach subproblem
// Dinkelbach transforms: max sum w_ij x_i x_j / sum x_i
// into: max sum w_ij x_i x_j - lambda * sum x_i
// which becomes: min sum_i lambda * x_i - sum_{i,j} w_ij x_i x_j
// 
// Note: For i=j (self-loops), w_ii * x_i * x_i = w_ii * x_i (since x_i is binary)
// So self-loops are already included in the quadratic formulation!
struct QPBOResult {
    vector<int> labels; // -1 = undecided, 0 = out, 1 = in
    vector<Vertex> fixed_in;
    vector<Vertex> fixed_out;
    vector<Vertex> undecided;
};

QPBOResult run_qpbo(const Graph& G, double lambda) {
    size_t n = num_vertices(G);
    
    typedef double REAL;
    std::unique_ptr<QPBO<REAL>> qpbo(new QPBO<REAL>(n, num_edges(G)));
    qpbo->AddNode(n);
    
    // Dinkelbach subproblem: minimize sum_i lambda * x_i - sum_{i,j} w_ij * x_i * x_j
    
    // Add unary terms: lambda * xi for each node
    for (size_t i = 0; i < n; i++) {
        REAL E0 = 0.0;      // cost if xi = 0
        REAL E1 = lambda;   // cost if xi = 1
        
        // Self-loops: w_ii * x_i * x_i = w_ii * x_i (treated as unary)
        // We want to minimize -w_ii * x_i, so add -w_ii to E1
        if (G[i].has_self_loop) {
            E1 -= G[i].self_loop_polarity;
        }
        
        qpbo->AddUnaryTerm(i, E0, E1);
    }
    
    // Add pairwise terms: -wij * xi * xj (for i != j)
    auto edge_range = edges(G);
    for (auto it = edge_range.first; it != edge_range.second; ++it) {
        auto u = source(*it, G);
        auto v = target(*it, G);
        double w = G[*it].polarity;
        
        // We want to minimize -w * xi * xj
        // QPBO minimizes E00*(1-xi)*(1-xj) + E01*(1-xi)*xj + E10*xi*(1-xj) + E11*xi*xj
        // For -w * xi * xj: E00=0, E01=0, E10=0, E11=-w
        qpbo->AddPairwiseTerm(u, v, 0, 0, 0, -w);
    }
    
    qpbo->Solve();
    qpbo->ComputeWeakPersistencies();
    
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

// Solve MIP on undecided nodes for Dinkelbach subproblem
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

        // Unary terms
        for (size_t i = 0; i < n; i++) {
            double coeff = lambda;
            if (G[i].has_self_loop) {
                coeff -= G[i].self_loop_polarity;
            }
            
            if (qpbo_result.labels[i] == 1) {
                obj += coeff * 1.0;  // Constant
            } else if (qpbo_result.labels[i] == 0) {
                // obj += coeff * 0.0;  // Zero, skip
            } else {
                obj += coeff * undecided_vars[i];
            }
        }

        // Pairwise terms
        auto edge_range = edges(G);
        for (auto it = edge_range.first; it != edge_range.second; ++it) {
            auto u = source(*it, G);
            auto v = target(*it, G);
            
            double w = G[*it].polarity;
            
            int label_u = qpbo_result.labels[u];
            int label_v = qpbo_result.labels[v];
            
            if (label_u == -1 && label_v == -1) {
                // Both undecided: quadratic term
                obj += -w * undecided_vars[u] * undecided_vars[v];
            } else if (label_u == -1 && label_v != -1) {
                // Only u is undecided: linear term
                obj += -w * label_v * undecided_vars[u];
            } else if (label_v == -1 && label_u != -1) {
                // Only v is undecided: linear term
                obj += -w * label_u * undecided_vars[v];
            } else {
                // Both fixed: constant
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
                    // Fixed to 1 by QPBO
                    selected.push_back(i);
                } else if (qpbo_result.labels[i] == -1) {
                    // Undecided, check MIP solution
                    if (undecided_vars.count(i) && undecided_vars[i].get(GRB_DoubleAttr_X) > 0.5) {
                        selected.push_back(i);
                    }
                }
                // qpbo_result.labels[i] == 0: not selected
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

// Dinkelbach algorithm with QPBO + MIP hybrid
DensestSubgraphResult qpbo_mip(const Graph& G, unsigned max_iterations = 10, double epsilon = 1e-3) {
    size_t n = num_vertices(G);    
    vector<Vertex> best_solution;
    double best_density = -numeric_limits<double>::infinity();
    double lambda = 0.0;
    // set initial lambda
    double init_density = 0.0;
    for (auto e : make_iterator_range(edges(G))) {
        double new_density = (G[e].polarity + G[source(e, G)].self_loop_polarity + G[target(e, G)].self_loop_polarity) / 2.0;
        init_density = max(init_density, new_density);
    }
    for (size_t i = 0; i < num_vertices(G); i++) {
        if (G[i].has_self_loop) {
            init_density = max(init_density, G[i].self_loop_polarity);
        }
    }
    lambda = init_density;

    for (int iter = 0; iter < max_iterations; iter++) {
        // Step 1: Run QPBO with current lambda
        QPBOResult qpbo_result = run_qpbo(G, lambda);
        
        vector<Vertex> current_solution;
        
        // Step 2: Check if QPBO solved everything
        if (qpbo_result.undecided.empty()) {
            current_solution = qpbo_result.fixed_in;
        } else {            
            // Step 4: Solve MIP on undecided nodes (seeded with heuristic)
            current_solution = solve_mip_on_undecided(G, qpbo_result, lambda);
        }

        // Handle empty solutions properly
        double density;
        if (current_solution.empty()) {
            density = 0.0;
            // If lambda is already close to 0, we've converged to empty set
            if (abs(lambda) < epsilon) {
                break;
            }
        } else {
            density = compute_density(G, current_solution);
            
            // Update best solution
            if (density > best_density) {
                best_density = density;
                best_solution = current_solution;
            }
        }
        
        // Dinkelbach convergence check
        if (abs(density - lambda) < epsilon) {
            break;
        }
        
        // Update lambda for next iteration
        lambda = density;
    }

    if (best_solution.empty()) {
        best_density = 0.0;
    }
    
    return {best_solution, best_density};
}

int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 7) {
        cerr << "Usage: " << argv[0] << " <filename> <output_filename> <reverse_weight> <dinkelbach_iterations> <epsilon> [num_its]" << endl;
        return EXIT_FAILURE;
    }
    string filename = argv[1];
    string output_filename = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    unsigned dinkelbach_iterations = stoul(argv[4], nullptr);
    double epsilon = stod(argv[5]);
    unsigned num_its = (argc >= 7) ? stoul(argv[6], nullptr) : 1;

    try {
        Graph G = read_graph(filename, reverse_weight);

        vector<Vertex> first_selected;
        double first_density;
        double total_elapsed = 0.0;

        for (unsigned iteration = 0; iteration < num_its; ++iteration) {
            auto start = chrono::high_resolution_clock::now();
            auto result = qpbo_mip(G, dinkelbach_iterations, epsilon);
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