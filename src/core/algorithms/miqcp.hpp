#pragma once

#include "../graph.hpp"
#include <gurobi_c++.h>

class MIQCP : public PGraph {
public:
    struct MIQCP_Config : public PGraph_Config {
        double time_limit = 3600.0;  // Time limit in seconds (default 1 hour)
        double mip_gap = 1e-4;       // MIP gap tolerance
        int threads = 0;             // Number of threads (0 = automatic)
        bool verbose = true;         // Show Gurobi output
        
        void load_from_json(const string& filename) override {
            PGraph_Config::load_from_json(filename);
            
            ifstream file(filename);
            string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
            auto json = json::parse(content).as_object();
            
            if (json.contains("time_limit")) time_limit = json.at("time_limit").to_number<double>();
            if (json.contains("mip_gap")) mip_gap = json.at("mip_gap").to_number<double>();
            if (json.contains("threads")) threads = json.at("threads").to_number<int>();
            if (json.contains("verbose")) verbose = json.at("verbose").as_bool();
        }
        
        void add_to_json(json::object& cfg) const override {
            PGraph_Config::add_to_json(cfg);
            cfg["time_limit"] = time_limit;
            cfg["mip_gap"] = mip_gap;
            cfg["threads"] = threads;
            cfg["verbose"] = verbose;
        }
    };

private:
    size_t n;  // Number of vertices
    
public:
    MIQCP(const MIQCP_Config& cfg) : PGraph(cfg) {
        n = num_vertices(G);
        info = "| MIQCP Baseline | ";
    }

    SubgraphResult Run(const MIQCP_Config& cfg) {
        try {
            // Create Gurobi environment and model
            GRBEnv env = GRBEnv(true);
            if (!cfg.verbose) {
                env.set(GRB_IntParam_OutputFlag, 0);
            }
            env.start();
            
            GRBModel model = GRBModel(env);
            model.set(GRB_StringAttr_ModelName, "Densest_Subgraph_MIQCP");
            
            // Set parameters
            if (cfg.time_limit > 0) {
                model.set(GRB_DoubleParam_TimeLimit, cfg.time_limit);
            }
            model.set(GRB_DoubleParam_MIPGap, cfg.mip_gap);
            if (cfg.threads > 0) {
                model.set(GRB_IntParam_Threads, cfg.threads);
            }
            
            // Decision variables
            // x[i]: binary variable indicating if vertex i is in the subgraph
            vector<GRBVar> x(n);
            for (size_t i = 0; i < n; ++i) {
                x[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + to_string(i));
            }
            
            // t: auxiliary variable for the objective (density)
            GRBVar t = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "t");
            
            // Objective: maximize t (density)
            GRBLinExpr obj_expr = t;
            model.setObjective(obj_expr, GRB_MAXIMIZE);
            
            // Constraint: at least one vertex must be selected
            GRBLinExpr sum_x = 0;
            for (size_t i = 0; i < n; ++i) {
                sum_x += x[i];
            }
            model.addConstr(sum_x >= 1, "at_least_one_vertex");
            
            // Build the edge weight sum expression
            GRBQuadExpr edge_weight_sum = 0;
            
            // Add loop weights (self-loops)
            for (size_t i = 0; i < n; ++i) {
                if (loop_weight[i] != 0.0) {
                    edge_weight_sum += loop_weight[i] * x[i];
                }
            }
            
            // Add edge weights
            // For undirected graph, each edge (u,v) contributes w_uv * x_u * x_v
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                auto u = source(*ei, G);
                auto v = target(*ei, G);
                double weight = G[*ei].weight;
                
                if (weight != 0.0) {
                    // Add quadratic term: w * x[u] * x[v]
                    edge_weight_sum += weight * x[u] * x[v];
                }
            }
            
            // Constraint: edge_weight_sum >= t * sum_x
            // Rearranged: edge_weight_sum - t * sum_x >= 0
            model.addQConstr(edge_weight_sum - t * sum_x >= 0, "density_constraint");
            
            // Solve the model
            model.optimize();
            
            // Extract solution
            SubgraphResult result;
            result.nodes.clear();
            result.density = 0.0;
            
            int status = model.get(GRB_IntAttr_Status);
            
            if (status == GRB_OPTIMAL || status == GRB_TIME_LIMIT || status == GRB_INTERRUPTED) {
                // Check if we have a feasible solution
                int sol_count = model.get(GRB_IntAttr_SolCount);
                if (sol_count > 0) {
                    // Extract vertices in the solution
                    for (size_t i = 0; i < n; ++i) {
                        if (x[i].get(GRB_DoubleAttr_X) > 0.5) {
                            result.nodes.push_back(i);
                        }
                    }
                    
                    // Compute actual density
                    if (!result.nodes.empty()) {
                        double weight_sum = 0.0;
                        vector<bool> selected(n, false);
                        for (auto node : result.nodes) {
                            selected[node] = true;
                            weight_sum += loop_weight[node];
                        }
                        
                        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                            auto u = source(*ei, G);
                            auto v = target(*ei, G);
                            if (selected[u] && selected[v]) {
                                weight_sum += G[*ei].weight;
                            }
                        }
                        
                        result.density = weight_sum / result.nodes.size();
                    }
                    
                    if (cfg.verbose) {
                        cout << "Gurobi status: ";
                        if (status == GRB_OPTIMAL) {
                            cout << "OPTIMAL" << endl;
                        } else if (status == GRB_TIME_LIMIT) {
                            cout << "TIME_LIMIT (found feasible solution)" << endl;
                        } else if (status == GRB_INTERRUPTED) {
                            cout << "INTERRUPTED (found feasible solution)" << endl;
                        }
                        cout << "Solution found with " << result.nodes.size() 
                             << " vertices and density " << result.density << endl;
                        
                        // Show bound information
                        try {
                            double obj_val = model.get(GRB_DoubleAttr_ObjVal);
                            double obj_bound = model.get(GRB_DoubleAttr_ObjBound);
                            cout << "Objective value: " << obj_val << endl;
                            cout << "Best bound: " << obj_bound << endl;
                            if (status != GRB_OPTIMAL && obj_bound > obj_val) {
                                cout << "Gap: " << (obj_bound - obj_val) / max(abs(obj_val), 1.0) * 100 << "%" << endl;
                            }
                        } catch (...) {
                            // Bound might not be available in some cases
                        }
                    }
                } else {
                    cerr << "Warning: Gurobi finished but no feasible solution found" << endl;
                    result.nodes = {};
                    result.density = 0.0;
                }
            } else if (status == GRB_INFEASIBLE) {
                cerr << "Error: Model is infeasible" << endl;
                result.nodes = {};
                result.density = 0.0;
            } else if (status == GRB_UNBOUNDED) {
                cerr << "Error: Model is unbounded" << endl;
                result.nodes = {};
                result.density = 0.0;
            } else {
                cerr << "Error: Optimization ended with status " << status << endl;
                result.nodes = {};
                result.density = 0.0;
            }
            
            return result;
            
        } catch (GRBException& e) {
            cerr << "Gurobi exception: " << e.getMessage() << endl;
            cerr << "Error code: " << e.getErrorCode() << endl;
            return {{}, 0.0};
        } catch (std::exception& e) {
            cerr << "Standard exception: " << e.what() << endl;
            return {{}, 0.0};
        }
    }
    
    void Reset(const MIQCP_Config& cfg) {
        // No state to reset for MIQCP (stateless solver)
    }
};
