#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <queue>
#include <chrono>
#include "gurobi_c++.h"
#include "QPBO.h"

using namespace std;

// ============================================================================
// Graph Structure
// ============================================================================

struct Graph {
    size_t num_nodes;
    size_t num_edges;  // Count of regular edges (i != j)
    vector<double> node_self_loops;  // Self-loop weights w_ii
    vector<vector<pair<int, double>>> adj_list;  // Adjacency list: (neighbor, weight) for i != j
    
    Graph(size_t n) : num_nodes(n), num_edges(0), node_self_loops(n, 0.0), adj_list(n) {}
    
    void add_edge(int i, int j, double weight) {
        if (i == j) {
            // Self-loop
            node_self_loops[i] += weight;
        } else {
            // Regular edge
            adj_list[i].push_back({j, weight});
            adj_list[j].push_back({i, weight});
            num_edges++;
        }
    }
};

// ============================================================================
// QPBO Result Structure
// ============================================================================

struct QPBOResult {
    vector<int> assignment;      // -1 = undecided, 0 = out, 1 = in
    vector<int> fixed_in;        // Nodes persistently labeled as IN
    vector<int> fixed_out;       // Nodes persistently labeled as OUT
    vector<int> undecided;       // Nodes that QPBO couldn't decide
    bool fully_labeled;          // True if all nodes were decided
    
    QPBOResult(size_t n) : assignment(n, -1), fully_labeled(false) {}
};

// ============================================================================
// Utility Functions
// ============================================================================

double compute_total_weight(const Graph& G, const vector<int>& x) {
    double W = 0.0;
    
    // Self-loops
    for (size_t i = 0; i < G.num_nodes; i++) {
        if (x[i] == 1) {
            W += G.node_self_loops[i];
        }
    }
    
    // Regular edges (count each once)
    vector<bool> counted(G.num_nodes, false);
    for (size_t i = 0; i < G.num_nodes; i++) {
        counted[i] = true;
        if (x[i] == 1) {
            for (const auto& [j, weight] : G.adj_list[i]) {
                if (x[j] == 1 && !counted[j]) {
                    W += weight;
                }
            }
        }
    }
    
    return W;
}

int count_selected(const vector<int>& x) {
    int count = 0;
    for (int val : x) {
        if (val == 1) count++;
    }
    return count;
}

double compute_density(const Graph& G, const vector<int>& x) {
    int size = count_selected(x);
    if (size == 0) return 0.0;
    return compute_total_weight(G, x) / size;
}

// ============================================================================
// QPBO Solver
// ============================================================================

QPBOResult run_qpbo(const Graph& G, double lambda) {
    typedef double REAL;
    QPBOResult result(G.num_nodes);
    
    // Create QPBO instance
    QPBO<REAL>* qpbo = new QPBO<REAL>(G.num_nodes, G.num_edges * 2);
    qpbo->AddNode(G.num_nodes);
    
    // Add unary terms: (lambda - w_ii) * x_i
    // Energy to minimize: lambda * x_i - w_ii * x_i
    for (size_t i = 0; i < G.num_nodes; i++) {
        double unary_coeff = lambda - G.node_self_loops[i];
        qpbo->AddUnaryTerm(i, 0, unary_coeff);
    }
    
    // Add pairwise terms: -w_ij * x_i * x_j
    // Need to avoid adding each edge twice
    vector<bool> edge_added(G.num_nodes, false);
    for (size_t i = 0; i < G.num_nodes; i++) {
        edge_added[i] = true;
        for (const auto& [j, weight] : G.adj_list[i]) {
            if (!edge_added[j]) {
                // E00, E01, E10, E11
                // We want to subtract w_ij when both are selected
                qpbo->AddPairwiseTerm(i, j, 0, 0, 0, -weight);
            }
        }
    }
    
    // Solve QPBO
    qpbo->Solve();
    qpbo->ComputeWeakPersistencies();
    
    // Extract labels
    for (size_t i = 0; i < G.num_nodes; i++) {
        int label = qpbo->GetLabel(i);
        if (label == 0) {
            result.assignment[i] = 0;
            result.fixed_out.push_back(i);
        } else if (label == 1) {
            result.assignment[i] = 1;
            result.fixed_in.push_back(i);
        } else {
            result.assignment[i] = -1;
            result.undecided.push_back(i);
        }
    }
    
    result.fully_labeled = result.undecided.empty();
    
    delete qpbo;
    return result;
}

// ============================================================================
// MIP Solver (Gurobi)
// ============================================================================

vector<int> solve_mip_gurobi(const Graph& G, const QPBOResult& qpbo_result, 
                              double lambda, int min_size = 1, double time_limit = 300.0) {
    vector<int> solution = qpbo_result.assignment;
    
    if (qpbo_result.fully_labeled) {
        cout << "QPBO fully labeled all nodes!" << endl;
        return solution;
    }
    
    try {
        GRBEnv env = GRBEnv();
        env.set(GRB_IntParam_OutputFlag, 1);
        GRBModel model = GRBModel(env);
        model.set(GRB_DoubleParam_TimeLimit, time_limit);
        
        // Create variables for undecided nodes
        int num_undecided = qpbo_result.undecided.size();
        vector<GRBVar> x_vars(num_undecided);
        vector<int> idx_map(G.num_nodes, -1);  // Maps node ID to variable index
        
        for (int k = 0; k < num_undecided; k++) {
            int i = qpbo_result.undecided[k];
            x_vars[k] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
            idx_map[i] = k;
        }
        
        // Build objective
        GRBQuadExpr obj = 0;
        
        // Unary terms: (w_ii - lambda) * x_i for undecided nodes
        for (int k = 0; k < num_undecided; k++) {
            int i = qpbo_result.undecided[k];
            double coeff = G.node_self_loops[i] - lambda;
            obj += coeff * x_vars[k];
        }
        
        // Edge terms
        vector<bool> edge_counted(G.num_nodes, false);
        
        for (size_t i = 0; i < G.num_nodes; i++) {
            edge_counted[i] = true;
            int idx_i = idx_map[i];
            
            for (const auto& [j, weight] : G.adj_list[i]) {
                if (edge_counted[j]) continue;  // Each edge only once
                
                int idx_j = idx_map[j];
                
                if (idx_i >= 0 && idx_j >= 0) {
                    // Both nodes undecided: create auxiliary variable y_ij
                    GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                    
                    // Constraints: y_ij <= x_i, y_ij <= x_j
                    model.addConstr(y <= x_vars[idx_i]);
                    model.addConstr(y <= x_vars[idx_j]);
                    
                    // For negative edges: y_ij >= x_i + x_j - 1
                    if (weight < 0) {
                        model.addConstr(y >= x_vars[idx_i] + x_vars[idx_j] - 1);
                    }
                    
                    obj += weight * y;
                    
                } else if (idx_i >= 0 && solution[j] == 1) {
                    // Node i undecided, node j fixed to 1
                    obj += weight * x_vars[idx_i];
                    
                } else if (idx_j >= 0 && solution[i] == 1) {
                    // Node j undecided, node i fixed to 1
                    obj += weight * x_vars[idx_j];
                }
            }
        }
        
        // Cardinality constraint
        int num_fixed_in = qpbo_result.fixed_in.size();
        GRBLinExpr card_expr = 0;
        for (int k = 0; k < num_undecided; k++) {
            card_expr += x_vars[k];
        }
        model.addConstr(card_expr + num_fixed_in >= min_size);
        
        // Set objective and optimize
        model.setObjective(obj, GRB_MAXIMIZE);
        model.optimize();
        
        // Extract solution
        int status = model.get(GRB_IntAttr_Status);
        if (status == GRB_OPTIMAL || status == GRB_TIME_LIMIT) {
            for (int k = 0; k < num_undecided; k++) {
                int i = qpbo_result.undecided[k];
                solution[i] = (int)round(x_vars[k].get(GRB_DoubleAttr_X));
            }
            
            double obj_val = model.get(GRB_DoubleAttr_ObjVal);
            cout << "MIP objective: " << obj_val << " (status: " << status << ")" << endl;
        } else {
            cerr << "MIP solver failed with status: " << status << endl;
        }
        
    } catch (GRBException& e) {
        cerr << "Gurobi exception: " << e.getMessage() << endl;
    }
    
    return solution;
}

// ============================================================================
// Heuristic: Signed Peeling + Hill Climbing
// ============================================================================

vector<int> heuristic_peeling(const Graph& G, int min_size = 1) {
    // Start from all nodes
    vector<int> S(G.num_nodes, 1);
    vector<double> deg(G.num_nodes, 0.0);
    
    // Compute initial internal degrees
    for (size_t i = 0; i < G.num_nodes; i++) {
        deg[i] = G.node_self_loops[i];
        for (const auto& [j, weight] : G.adj_list[i]) {
            if (S[j] == 1) {
                deg[i] += weight;
            }
        }
    }
    
    int current_size = G.num_nodes;
    bool improved = true;
    
    while (improved && current_size > min_size) {
        improved = false;
        double current_density = compute_density(G, S);
        
        // Try to remove node with smallest degree < density
        int best_remove = -1;
        double min_deg = numeric_limits<double>::max();
        
        for (size_t i = 0; i < G.num_nodes; i++) {
            if (S[i] == 1 && deg[i] < current_density) {
                if (deg[i] < min_deg) {
                    min_deg = deg[i];
                    best_remove = i;
                }
            }
        }
        
        if (best_remove >= 0 && current_size > min_size) {
            // Remove node
            S[best_remove] = 0;
            current_size--;
            
            // Update neighbors' degrees
            for (const auto& [j, weight] : G.adj_list[best_remove]) {
                if (S[j] == 1) {
                    deg[j] -= weight;
                }
            }
            deg[best_remove] = 0;
            
            improved = true;
            continue;
        }
        
        // Try to add node with largest degree > density
        int best_add = -1;
        double max_deg = -numeric_limits<double>::max();
        
        for (size_t i = 0; i < G.num_nodes; i++) {
            if (S[i] == 0) {
                // Compute degree if added
                double potential_deg = G.node_self_loops[i];
                for (const auto& [j, weight] : G.adj_list[i]) {
                    if (S[j] == 1) {
                        potential_deg += weight;
                    }
                }
                
                if (potential_deg > current_density && potential_deg > max_deg) {
                    max_deg = potential_deg;
                    best_add = i;
                }
            }
        }
        
        if (best_add >= 0) {
            // Add node
            S[best_add] = 1;
            current_size++;
            
            // Update degree
            deg[best_add] = G.node_self_loops[best_add];
            for (const auto& [j, weight] : G.adj_list[best_add]) {
                if (S[j] == 1) {
                    deg[best_add] += weight;
                    deg[j] += weight;
                }
            }
            
            improved = true;
        }
    }
    
    return S;
}

// ============================================================================
// Dinkelbach Main Solver
// ============================================================================

struct DensestSubgraphResult {
    vector<int> solution;
    double density;
    int num_iterations;
    double total_time;
};

DensestSubgraphResult solve_densest_subgraph(const Graph& G, int min_size = 1, 
                                               double epsilon = 1e-6, 
                                               double mip_time_limit = 60.0,
                                               int max_iterations = 50) {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Get initial solution from heuristic
    cout << "Running heuristic..." << endl;
    vector<int> best_solution = heuristic_peeling(G, min_size);
    double best_density = compute_density(G, best_solution);
    cout << "Heuristic density: " << best_density << endl;
    
    // Initialize lambda
    double lambda = best_density;
    
    int iteration = 0;
    while (iteration < max_iterations) {
        iteration++;
        cout << "\n=== Dinkelbach Iteration " << iteration << " (lambda = " << lambda << ") ===" << endl;
        
        // Run QPBO
        cout << "Running QPBO..." << endl;
        QPBOResult qpbo_result = run_qpbo(G, lambda);
        cout << "QPBO: fixed_in=" << qpbo_result.fixed_in.size() 
             << ", fixed_out=" << qpbo_result.fixed_out.size()
             << ", undecided=" << qpbo_result.undecided.size() << endl;
        
        // Solve remaining with MIP
        vector<int> current_solution;
        if (qpbo_result.fully_labeled) {
            current_solution = qpbo_result.assignment;
        } else {
            cout << "Running MIP on " << qpbo_result.undecided.size() << " undecided nodes..." << endl;
            current_solution = solve_mip_gurobi(G, qpbo_result, lambda, min_size, mip_time_limit);
        }
        
        // Compute metrics
        double W = compute_total_weight(G, current_solution);
        int S_size = count_selected(current_solution);
        double current_density = (S_size > 0) ? W / S_size : 0.0;
        double F = W - lambda * S_size;
        
        cout << "Solution: |S| = " << S_size << ", W(S) = " << W 
             << ", density = " << current_density << ", F(S) = " << F << endl;
        
        // Update best solution
        if (current_density > best_density) {
            best_density = current_density;
            best_solution = current_solution;
            cout << "*** New best density: " << best_density << " ***" << endl;
        }
        
        // Check convergence
        if (abs(F) < epsilon) {
            cout << "Converged!" << endl;
            break;
        }
        
        // Update lambda
        lambda = current_density;
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end_time - start_time).count();
    
    DensestSubgraphResult result;
    result.solution = best_solution;
    result.density = best_density;
    result.num_iterations = iteration;
    result.total_time = total_time;
    
    return result;
}

// ============================================================================
// Example Usage
// ============================================================================

int main() {
    // Create a small example graph
    int n = 6;
    Graph G(n);
    
    // Add self-loops
    G.node_self_loops[0] = 2.0;
    G.node_self_loops[1] = 1.5;
    G.node_self_loops[2] = -0.5;
    G.node_self_loops[3] = 1.0;
    G.node_self_loops[4] = 0.5;
    G.node_self_loops[5] = -1.0;
    
    // Add edges (including some negative weights)
    G.add_edge(0, 1, 3.0);
    G.add_edge(0, 2, 2.0);
    G.add_edge(1, 2, 2.5);
    G.add_edge(1, 3, -1.0);  // Negative edge
    G.add_edge(2, 3, 1.5);
    G.add_edge(3, 4, 2.0);
    G.add_edge(3, 5, -0.5);  // Negative edge
    G.add_edge(4, 5, 1.0);
    
    cout << "Graph: " << G.num_nodes << " nodes, " << G.num_edges << " edges" << endl;
    
    // Solve
    DensestSubgraphResult result = solve_densest_subgraph(G, 2, 1e-6, 60.0, 50);
    
    // Print results
    cout << "\n=== Final Result ===" << endl;
    cout << "Optimal density: " << result.density << endl;
    cout << "Number of iterations: " << result.num_iterations << endl;
    cout << "Total time: " << result.total_time << " seconds" << endl;
    cout << "Selected nodes: ";
    for (size_t i = 0; i < result.solution.size(); i++) {
        if (result.solution[i] == 1) {
            cout << i << " ";
        }
    }
    cout << endl;
    
    return 0;
}