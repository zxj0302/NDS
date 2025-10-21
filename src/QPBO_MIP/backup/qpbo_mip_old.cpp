#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// QPBO library - you need to download from: https://pub.ista.ac.at/~vnk/software.html
#include "QPBO/QPBO.h"

// Choose one of these MIP solvers
#ifdef USE_GUROBI
#include "gurobi_c++.h"
#endif

#ifdef USE_CBC
#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"
#include "coin/CoinPackedMatrix.hpp"
#include "coin/CoinPackedVector.hpp"
#endif

using namespace std;

/*
 * =========================================================
 * Graph Structure (Simple adjacency representation)
 * =========================================================
 */
struct Graph {
    size_t num_nodes;
    size_t num_edges;
    vector<double> node_self_loops;  // self-loop weights
    vector<vector<pair<int, double>>> adj_list;  // adjacency list: (neighbor, weight)
    
    Graph(size_t n) : num_nodes(n), num_edges(0) {
        node_self_loops.resize(n, 0.0);
        adj_list.resize(n);
    }
    
    void add_edge(int u, int v, double weight) {
        if (u == v) {
            node_self_loops[u] = weight;
        } else {
            adj_list[u].push_back({v, weight});
            adj_list[v].push_back({u, weight});
        }
        num_edges++;
    }
};

/*
 * =========================================================
 * Read Graph from File
 * =========================================================
 */
Graph read_graph(const string& filename, bool reverse_weight = false) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    size_t num_nodes = 0, num_edges = 0;
    string first_line;
    if (!getline(infile, first_line)) {
        throw runtime_error("Failed to read the first line.");
    }
    
    istringstream iss_first(first_line);
    if (!(iss_first >> num_nodes >> num_edges)) {
        throw runtime_error("Failed to parse number of nodes and edges.");
    }

    Graph G(num_nodes);
    
    size_t edge_count = 0;
    string line;
    while (edge_count < num_edges && getline(infile, line)) {
        istringstream iss(line);
        unsigned u, v;
        double edge_weight;

        if (!(iss >> u >> v >> edge_weight)) {
            throw runtime_error("Failed to parse edge data: " + line);
        }

        double final_weight = reverse_weight ? -edge_weight : edge_weight;
        G.add_edge(u, v, final_weight);
        edge_count++;
    }

    infile.close();
    if (edge_count != num_edges) {
        throw runtime_error("Edge count mismatch");
    }

    return G;
}

/*
 * =========================================================
 * QPBO + MIP Solver
 * =========================================================
 */
class QPBOMIPSolver {
private:
    const Graph& G;
    
    struct QPBOResult {
        vector<int> assignment;  // 0, 1, or -1 (undecided)
        vector<int> fixed_in;
        vector<int> fixed_out;
        vector<int> undecided;
        bool fully_labeled;
    };
    
    // Compute objective value for a given solution
    double compute_objective(const vector<int>& x) {
        double obj = 0.0;
        
        // Self-loop contributions
        for (size_t i = 0; i < G.num_nodes; i++) {
            if (x[i] == 1) {
                obj += G.node_self_loops[i];
            }
        }
        
        // Edge contributions (count each edge once)
        vector<bool> counted(G.num_nodes, false);
        for (size_t i = 0; i < G.num_nodes; i++) {
            if (x[i] == 1) {
                counted[i] = true;
                for (const auto& [neighbor, weight] : G.adj_list[i]) {
                    if (x[neighbor] == 1 && !counted[neighbor]) {
                        obj += weight;
                    }
                }
            }
        }
        
        return obj;
    }
    
    // Run QPBO to get partial assignment
    QPBOResult run_qpbo() {
        QPBOResult result;
        result.assignment.resize(G.num_nodes, -1);
        result.fully_labeled = false;
        
        typedef double REAL;
        QPBO<REAL>* qpbo = new QPBO<REAL>(G.num_nodes, G.num_edges * 2);
        qpbo->AddNode(G.num_nodes);
        
        // Add unary terms (self-loops)
        // We want to MAXIMIZE total weight, so negate for minimization
        for (size_t i = 0; i < G.num_nodes; i++) {
            qpbo->AddUnaryTerm(i, 0, -G.node_self_loops[i]);
        }
        
        // Add pairwise terms (edges - count each once)
        vector<bool> edge_added(G.num_nodes, false);
        for (size_t i = 0; i < G.num_nodes; i++) {
            edge_added[i] = true;
            for (const auto& [j, weight] : G.adj_list[i]) {
                if (!edge_added[j]) {
                    // Maximize weight when both are selected
                    // E00=0, E01=0, E10=0, E11=-weight
                    qpbo->AddPairwiseTerm(i, j, 0, 0, 0, -weight);
                }
            }
        }
        
        qpbo->Solve();
        qpbo->ComputeWeakPersistencies();
        
        int num_undecided = 0;
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
                num_undecided++;
            }
        }
        
        result.fully_labeled = (num_undecided == 0);
        delete qpbo;
        return result;
    }
    
#ifdef USE_GUROBI
    vector<int> solve_mip_gurobi(const QPBOResult& qpbo_result) {
        vector<int> solution = qpbo_result.assignment;
        
        if (qpbo_result.fully_labeled) {
            return solution;
        }
        
        try {
            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            env.start();
            GRBModel model = GRBModel(env);
            
            int num_undecided = qpbo_result.undecided.size();
            vector<GRBVar> vars(num_undecided);
            vector<int> idx_map(G.num_nodes, -1);
            
            for (int k = 0; k < num_undecided; k++) {
                int i = qpbo_result.undecided[k];
                vars[k] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                idx_map[i] = k;
            }
            
            GRBQuadExpr obj = 0;
            
            // Self-loop terms (maximize, so negate)
            for (int k = 0; k < num_undecided; k++) {
                int i = qpbo_result.undecided[k];
                obj -= G.node_self_loops[i] * vars[k];
            }
            
            // Edge terms
            vector<bool> edge_counted(G.num_nodes, false);
            for (size_t i = 0; i < G.num_nodes; i++) {
                edge_counted[i] = true;
                int idx_i = idx_map[i];
                
                for (const auto& [j, weight] : G.adj_list[i]) {
                    if (edge_counted[j]) continue;
                    
                    int idx_j = idx_map[j];
                    
                    if (idx_i >= 0 && idx_j >= 0) {
                        obj -= weight * vars[idx_i] * vars[idx_j];
                    } else if (idx_i >= 0 && solution[j] == 1) {
                        obj -= weight * vars[idx_i];
                    } else if (idx_j >= 0 && solution[i] == 1) {
                        obj -= weight * vars[idx_j];
                    }
                }
            }
            
            model.setObjective(obj, GRB_MINIMIZE);
            model.optimize();
            
            for (int k = 0; k < num_undecided; k++) {
                int i = qpbo_result.undecided[k];
                solution[i] = (int)round(vars[k].get(GRB_DoubleAttr_X));
            }
            
        } catch (GRBException& e) {
            cerr << "Gurobi error: " << e.getMessage() << endl;
        }
        
        return solution;
    }
#endif

#ifdef USE_CBC
    vector<int> solve_mip_cbc(const QPBOResult& qpbo_result) {
        vector<int> solution = qpbo_result.assignment;
        
        if (qpbo_result.fully_labeled) {
            return solution;
        }
        
        int num_undecided = qpbo_result.undecided.size();
        vector<int> idx_map(G.num_nodes, -1);
        
        for (int k = 0; k < num_undecided; k++) {
            idx_map[qpbo_result.undecided[k]] = k;
        }
        
        // Count quadratic terms to linearize
        int num_quad = 0;
        vector<bool> edge_counted(G.num_nodes, false);
        for (size_t i = 0; i < G.num_nodes; i++) {
            edge_counted[i] = true;
            if (idx_map[i] >= 0) {
                for (const auto& [j, weight] : G.adj_list[i]) {
                    if (!edge_counted[j] && idx_map[j] >= 0) {
                        num_quad++;
                    }
                }
            }
        }
        
        int total_vars = num_undecided + num_quad;
        
        OsiClpSolverInterface solver;
        solver.messageHandler()->setLogLevel(0);
        
        vector<double> col_lb(total_vars, 0.0);
        vector<double> col_ub(total_vars, 1.0);
        vector<double> objective(total_vars, 0.0);
        
        // Unary terms (negate for maximization)
        for (int k = 0; k < num_undecided; k++) {
            objective[k] = -G.node_self_loops[qpbo_result.undecided[k]];
        }
        
        // Quadratic terms - create auxiliary variables
        int aux_idx = num_undecided;
        vector<tuple<int, int, int>> quad_map;
        
        fill(edge_counted.begin(), edge_counted.end(), false);
        for (size_t i = 0; i < G.num_nodes; i++) {
            edge_counted[i] = true;
            int idx_i = idx_map[i];
            if (idx_i >= 0) {
                for (const auto& [j, weight] : G.adj_list[i]) {
                    if (!edge_counted[j]) {
                        int idx_j = idx_map[j];
                        if (idx_j >= 0) {
                            objective[aux_idx] = -weight;
                            quad_map.push_back({idx_i, idx_j, aux_idx});
                            aux_idx++;
                        } else if (solution[j] == 1) {
                            objective[idx_i] -= weight;
                        }
                    }
                }
            }
        }
        
        solver.loadProblem(0, total_vars, nullptr, nullptr, col_lb.data(),
                          col_ub.data(), objective.data(), nullptr, nullptr);
        
        // McCormick constraints: z = x_i * x_j
        for (const auto& [idx_i, idx_j, z_idx] : quad_map) {
            CoinPackedVector row1, row2, row3;
            row1.insert(z_idx, 1.0);
            row1.insert(idx_i, -1.0);
            row2.insert(z_idx, 1.0);
            row2.insert(idx_j, -1.0);
            row3.insert(z_idx, -1.0);
            row3.insert(idx_i, 1.0);
            row3.insert(idx_j, 1.0);
            
            solver.addRow(row1, -solver.getInfinity(), 0.0);
            solver.addRow(row2, -solver.getInfinity(), 0.0);
            solver.addRow(row3, -solver.getInfinity(), 1.0);
        }
        
        for (int i = 0; i < total_vars; i++) {
            solver.setInteger(i);
        }
        
        CbcModel model(solver);
        model.setLogLevel(0);
        model.branchAndBound();
        
        const double* sol = model.bestSolution();
        if (sol) {
            for (int k = 0; k < num_undecided; k++) {
                solution[qpbo_result.undecided[k]] = (int)round(sol[k]);
            }
        }
        
        return solution;
    }
#endif

public:
    QPBOMIPSolver(const Graph& graph) : G(graph) {}
    
    pair<vector<int>, double> solve() {
        auto qpbo_result = run_qpbo();
        
        vector<int> solution;
#ifdef USE_GUROBI
        solution = solve_mip_gurobi(qpbo_result);
#elif defined(USE_CBC)
        solution = solve_mip_cbc(qpbo_result);
#else
        solution = qpbo_result.assignment;
        // If undecided, use simple heuristic
        for (int node : qpbo_result.undecided) {
            solution[node] = (G.node_self_loops[node] > 0) ? 1 : 0;
        }
#endif
        
        double obj = compute_objective(solution);
        return {solution, obj};
    }
};

/*
 * =========================================================
 * Main Function
 * =========================================================
 */
int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " <filename> <output_filename> <reverse_weight> <unused> [num_its]" << endl;
        return EXIT_FAILURE;
    }
    
    string filename = argv[1];
    string output_filename = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    unsigned num_its = (argc >= 6) ? stoul(argv[5], nullptr) : 1;

    try {
        Graph G = read_graph(filename, reverse_weight);
        
        vector<int> first_selected;
        double first_density;
        double total_elapsed = 0.0;

        for (unsigned iteration = 0; iteration < num_its; ++iteration) {
            auto start = chrono::high_resolution_clock::now();
            
            QPBOMIPSolver solver(G);
            auto [solution, objective] = solver.solve();
            
            auto end = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration_cast<chrono::nanoseconds>(end - start).count() / 1e9;
            total_elapsed += elapsed;

            if (iteration == 0) {
                for (size_t i = 0; i < solution.size(); i++) {
                    if (solution[i] == 1) {
                        first_selected.push_back(i);
                    }
                }
                
                // Compute density
                if (first_selected.size() > 0) {
                    first_density = objective / first_selected.size();
                } else {
                    first_density = 0.0;
                }
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

    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}