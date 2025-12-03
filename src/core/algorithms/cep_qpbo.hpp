#include "cep.hpp"
#include "../../external/QPBO/QPBO.h"
#include <gurobi_c++.h>

class CEP_QPBO : public CEP {
public:
    struct CEP_QPBO_Config : public CEP_Config {
        bool use_binary = true;
        double step_size = 1.05;
        bool use_probe = false;
        unsigned findub_mip_bound = 100;
        unsigned dinkelbach_iterations = 30;
        double epsilon = -0.00001;
        double mip_time_limit = 300.0;
        
        
        void load_from_json(const string& filename) override {
            CEP_Config::load_from_json(filename);
            
            ifstream file(filename);
            string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
            auto json = json::parse(content).as_object();
            
            if (json.contains("use_binary")) use_binary = json.at("use_binary").as_bool();
            if (json.contains("step_size")) step_size = json.at("step_size").to_number<double>();
            if (json.contains("use_probe")) use_probe = json.at("use_probe").as_bool();
            if (json.contains("findub_mip_bound")) findub_mip_bound = json.at("findub_mip_bound").to_number<unsigned>();
            if (json.contains("dinkelbach_iterations")) dinkelbach_iterations = json.at("dinkelbach_iterations").to_number<unsigned>();
            if (json.contains("epsilon")) epsilon = json.at("epsilon").to_number<double>();
            if (json.contains("mip_time_limit")) mip_time_limit = json.at("mip_time_limit").to_number<double>();
        }
        
        void add_to_json(json::object& cfg) const override {
            CEP_Config::add_to_json(cfg);
            cfg["use_binary"] = use_binary;
            cfg["step_size"] = step_size;
            cfg["use_probe"] = use_probe;
            cfg["findub_mip_bound"] = findub_mip_bound;
            cfg["dinkelbach_iterations"] = dinkelbach_iterations;
            cfg["epsilon"] = epsilon;
            cfg["mip_time_limit"] = mip_time_limit;
        }
    };

private:
    struct QPBOResult {
        vector<int> labels;
        vector<Vertex> fixed_in;
        vector<Vertex> fixed_out;
        vector<Vertex> undecided;
    };
    using REAL = double;
    size_t vertex_lower_bound, vertex_upper_bound;
    const bool set_vertex_lb = true;

public:
    CEP_QPBO(const CEP_QPBO_Config& cfg) : CEP(cfg) {
        vertex_lower_bound = 3;
        vertex_upper_bound = num_vertices(G);
    }

    QPBOResult RunQPBO(const CEP_QPBO_Config& cfg, double lambda) {
        size_t n = num_vertices(G);
        unique_ptr<QPBO<REAL>> qpbo(new QPBO<REAL>(n, 2 * num_edges(G)));
        qpbo->AddNode(n);
        
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            qpbo->AddUnaryTerm(*vi, 0.0, lambda-loop_weight[*vi]);
        }
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            qpbo->AddPairwiseTerm(source(*ei, G), target(*ei, G), 0, 0, 0, -G[*ei].weight);
        }
        
        QPBOResult result;
        result.labels.resize(n);
        
        if (cfg.use_probe) {
            // Use Probe to potentially fix more nodes
            vector<int> mapping(n);
            QPBO<REAL>::ProbeOptions options;
            options.weak_persistencies = 1;
            options.directed_constraints = 2;
            
            qpbo->Probe(mapping.data(), options);
            // // Probe() internally calls Solve() at the end, so we can call ComputeWeakPersistencies()
            // qpbo->ComputeWeakPersistencies();
            
            // After Probe, the problem is transformed. Use mapping to interpret results.
            for (size_t i = 0; i < n; i++) {
                if (mapping[i] < 2) {
                    // Node i was fixed to label mapping[i]
                    result.labels[i] = mapping[i];
                    if (mapping[i] == 0) {
                        result.fixed_out.push_back(i);
                    } else {
                        result.fixed_in.push_back(i);
                    }
                } else {
                    // Node i maps to new node (mapping[i]/2) with possible inversion
                    int new_node = mapping[i] / 2;
                    int inversion = mapping[i] % 2;
                    int new_label = qpbo->GetLabel(new_node);
                    
                    if (new_label >= 0) {
                        result.labels[i] = (new_label + inversion) % 2;
                        if (result.labels[i] == 0) {
                            result.fixed_out.push_back(i);
                        } else {
                            result.fixed_in.push_back(i);
                        }
                    } else {
                        result.labels[i] = -1;
                        result.undecided.push_back(i);
                    }
                }
            }
        } else {
            // Standard QPBO without Probe
            qpbo->Solve();
            qpbo->ComputeWeakPersistencies();
            
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
        }

        return result;
    }

    SubgraphResult FindLowerBound(const CEP_QPBO_Config& cfg) {
        auto result = CEP::Run(cfg);
        Reset();
        if (set_vertex_lb) {
            if (result.density < 0) {
                result = {{}, 0.0};
            }
            auto max_loop_weight_it = max_element(loop_weight.begin(), loop_weight.end());
            if (*max_loop_weight_it > result.density) {
                result = {{static_cast<size_t>(max_loop_weight_it - loop_weight.begin())}, *max_loop_weight_it};
            }
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                if (G[*ei].weight > 0) {
                    auto u = source(*ei, G);
                    auto v = target(*ei, G);
                    auto density = (loop_weight[u] + loop_weight[v] + G[*ei].weight) / 2.0;
                    if (density > result.density) {
                        result = {{u, v}, density};
                    }
                }
            }
        }
        return result;
    }

    double FindUpperBoundOld(const CEP_QPBO_Config& cfg, SubgraphResult& result_lb) {
        auto pos_weight_ub = *max_element(pos_weight.begin(), pos_weight.end());
        double lambda_ub = result_lb.density * cfg.step_size;
        while (lambda_ub < pos_weight_ub) {
            auto result = RunQPBO(cfg, lambda_ub);
            if (result.undecided.empty() && result.fixed_in.empty()) {
                break;
            }
            lambda_ub *= cfg.step_size;
        }
        return min(lambda_ub, pos_weight_ub);
    }

    double FindUpperBound(const CEP_QPBO_Config& cfg, SubgraphResult& result_lb) {
        auto pos_weight_ub = *max_element(pos_weight.begin(), pos_weight.end());
        double lambda_ub = result_lb.density * cfg.step_size;
        while (lambda_ub < pos_weight_ub) {
            auto result = RunQPBO(cfg, lambda_ub);
            if (result.undecided.size() <= cfg.findub_mip_bound) {
                auto [solution, success] = result.undecided.empty() ? make_pair(result.fixed_in, true) : RunMIP(cfg, result, lambda_ub);
                if (solution.empty() && success) {
                    // the density of any subgraph must be <= lambda_ub
                    return lambda_ub;
                } else if (!solution.empty() && success) {
                    // the density of solution must be larger than lambda_ub, update lb and lambda_ub
                    double density = ComputeDensity(solution);
                    result_lb = {solution, density};
                    vertex_upper_bound = solution.size() - 1;
                    if (set_vertex_lb) {
                        pos_weight_ub = min(pos_weight_ub, lambda_ub + solution.size() * (density - lambda_ub) / vertex_lower_bound);
                    }
                    lambda_ub = density;
                } else if (!solution.empty() && !success) {
                    // MIP hit time limit, get some solution but cannot guarantee optimality, check density to update lb
                    double density = ComputeDensity(solution);
                    if (density > result_lb.density) {
                        result_lb = {solution, density};
                        lambda_ub = max(lambda_ub, density);
                    }
                } else {
                    // MIP hit time limit and solution empty, do nothing, try next lambda_ub
                }
            }
            lambda_ub *= cfg.step_size;
        }
        return pos_weight_ub;
    }

    pair<vector<Vertex>, bool> RunMIP(const CEP_QPBO_Config& cfg, QPBOResult& qpbo_result, double lambda) {
        try {
            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            env.start();
            GRBModel model = GRBModel(env);
            
            auto n = num_vertices(G);
            std::unordered_map<size_t, GRBVar> undecided_vars;
            for (auto i : qpbo_result.undecided) {
                undecided_vars[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
            }
            GRBQuadExpr obj = 0.0;
            for (auto i = 0; i < n; i++) {
                double coeff = lambda - loop_weight[i];
                if (qpbo_result.labels[i] == 1) {
                    obj += coeff * 1.0;
                } else if (qpbo_result.labels[i] == -1) {
                    obj += coeff * undecided_vars[i];
                }
            }

            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                auto u = source(*ei, G);
                auto v = target(*ei, G);
                auto w = G[*ei].weight;
                auto label_u = qpbo_result.labels[u];
                auto label_v = qpbo_result.labels[v];
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

            // Note: add vertex count constraint
            GRBLinExpr vertex_sum = 0.0;
            for (auto i : qpbo_result.undecided) {
                vertex_sum += undecided_vars[i];
            }
            model.addConstr(vertex_sum <= (vertex_upper_bound - qpbo_result.fixed_in.size()));
            // Warn: Don't set it, because it will return non-empty result even if the lambda >= largest density
            // The non-empty result cannot reflect correct upper bound for Dinkelbach, will make the Dinkelbach break.
            // if (set_vertex_lb) {
            //     model.addConstr(vertex_sum >= max(static_cast<size_t>(0), vertex_lower_bound - qpbo_result.fixed_in.size()));
            // }
            
            model.setObjective(obj, GRB_MINIMIZE);
            model.set(GRB_DoubleParam_TimeLimit, cfg.mip_time_limit);
            model.optimize();
            vector<Vertex> selected = qpbo_result.fixed_in;
            if (model.get(GRB_IntAttr_SolCount) > 0) {
                for (auto i : qpbo_result.undecided) {
                    if (undecided_vars.count(i) && undecided_vars[i].get(GRB_DoubleAttr_X) > 0.5) {
                        selected.push_back(i);
                    }
                }
            }
            return {selected, model.get(GRB_IntAttr_Status) == GRB_OPTIMAL};

        } catch (GRBException& e) {
            throw runtime_error("Gurobi exception: " + string(e.getMessage()));
            return {qpbo_result.fixed_in, false};
        }
    }

    SubgraphResult DinkelbachBinary(const CEP_QPBO_Config& cfg, const SubgraphResult& cep_result, double upper_bound) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < cfg.dinkelbach_iterations; iter++) {
            if ((cfg.epsilon > 0 ? (lower_bound / upper_bound) : (lower_bound - upper_bound)) >= cfg.epsilon) {
                break;
            }
            double lambda = (lower_bound + upper_bound) / 2.0;
            QPBOResult qpbo_result = RunQPBO(cfg, lambda);
            auto [solution, success] = qpbo_result.undecided.empty() ? make_pair(qpbo_result.fixed_in, true) : RunMIP(cfg, qpbo_result, lambda);
            // FIX: there is possibility that solution given by MIP is empty because of time limit, which cannot reflect correct upper bound
            // FIX: Just assume MIP can give exact solution here for simplicity, should fix it later
            if (solution.empty() && success) {
                upper_bound = lambda;
            } else {
                double density = ComputeDensity(solution);
                if (density > lower_bound) {
                    lower_bound = density;
                    best_solution = solution;
                    vertex_upper_bound = solution.size() - 1;
                    if (set_vertex_lb && success) {
                        upper_bound = min(upper_bound, lambda + solution.size() * (density - lambda) / vertex_lower_bound);
                    }
                } else {
                    // throw runtime_error("Dinkelbach: computed density is less than best density");
                    // Cannot find better solution under current lambda within time limit
                    cerr << "DinkelbachBinary: cannot find better solution under current lambda within time limit." << endl;
                    break;
                }
            }
        }
        return {best_solution, lower_bound};
    }

    SubgraphResult Dinkelbach(const CEP_QPBO_Config& cfg, const SubgraphResult& cep_result) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < cfg.dinkelbach_iterations; iter++) {
            QPBOResult qpbo_result = RunQPBO(cfg, lower_bound);
            auto solution = qpbo_result.undecided.empty() ? qpbo_result.fixed_in : RunMIP(cfg, qpbo_result, lower_bound).first;
                double density = ComputeDensity(solution);
                if (density > lower_bound) {
                    lower_bound = density;
                    best_solution = solution;
                    vertex_upper_bound = solution.size() - 1;
                } else {
                    break;
            }
        }
        return {best_solution, lower_bound};
    }

    SubgraphResult Run(const CEP_QPBO_Config& cfg) {
        // Step 1. Result found by CEP as initial solution
        auto result_lb = FindLowerBound(cfg);
        // if qpbo_result.undecided is empty and fixed_in is empty, we can directly return result found by CEP as Optimal
        auto pre_qpbo = RunQPBO(cfg, result_lb.density);
        if (pre_qpbo.undecided.empty() && pre_qpbo.fixed_in.empty()) {
            return result_lb;
        } else if (pre_qpbo.undecided.empty()) {
            double density = ComputeDensity(pre_qpbo.fixed_in);
            if (density > result_lb.density) {
                result_lb = {pre_qpbo.fixed_in, density};
            }
        }
        
        if (cfg.use_binary) {
            // Step 2. Find an upper bound for QPBO
            auto upper_bound = FindUpperBound(cfg, result_lb);
            // Step 3. Refine the solution by Dinkelbach
            return DinkelbachBinary(cfg, result_lb, upper_bound);
        } else {
            return Dinkelbach(cfg, result_lb);
        }      
    }
};