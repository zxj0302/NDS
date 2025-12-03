#include "cep.hpp"
#include "../../external/QPBO/QPBO.h"
#include <gurobi_c++.h>

class CEP_QPBO : public CEP {
public:
    struct Config : public CEP::Config {
        double step_size = 1.05;
        unsigned ub_mip_bound = 100;
        unsigned dinkelbach_iterations = 30;
        double epsilon = -0.00001;
        double mip_time_limit = 300.0;
        bool use_binary = true;
        
        void load_from_json(const string& filename) {
            CEP::Config::load_from_json(filename);
            
            std::ifstream file(filename);
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            auto json = boost::json::parse(content).as_object();
            
            if (json.contains("step_size")) step_size = json.at("step_size").to_number<double>();
            if (json.contains("ub_mip_bound")) ub_mip_bound = json.at("ub_mip_bound").to_number<unsigned>();
            if (json.contains("dinkelbach_iterations")) dinkelbach_iterations = json.at("dinkelbach_iterations").to_number<unsigned>();
            if (json.contains("epsilon")) epsilon = json.at("epsilon").to_number<double>();
            if (json.contains("mip_time_limit")) mip_time_limit = json.at("mip_time_limit").to_number<double>();
            if (json.contains("use_binary")) use_binary = json.at("use_binary").as_bool();
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
    CEP_QPBO(const string& config_file) : CEP(config_file) {
        static_cast<Config&>(config).load_from_json(config_file);
        InitializePositiveWeights();
        vertex_lower_bound = 3; // initialized, but not used in RunMIP (explained there).
            vertex_upper_bound = num_vertices(G);
        }

    QPBOResult RunQPBO(double lambda, bool probe = false) {
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
        
        if (probe) {
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

    SubgraphResult FindLowerBound() {
        auto result = CEP::Run();
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

    double FindUpperBoundOld(SubgraphResult& result_lb, double step_size, unsigned ub_mip_bound, unsigned mip_time_limit) {
        auto pos_weight_ub = *max_element(pos_weight.begin(), pos_weight.end());
        double lambda_ub = result_lb.density * step_size;
        while (lambda_ub < pos_weight_ub) {
            auto result = RunQPBO(lambda_ub, false);
            if (result.undecided.empty() && result.fixed_in.empty()) {
                break;
            }
            lambda_ub *= step_size;
        }
        return min(lambda_ub, pos_weight_ub);
    }

    double FindUpperBound(SubgraphResult& result_lb, double step_size, unsigned ub_mip_bound, unsigned mip_time_limit) {
        auto pos_weight_ub = *max_element(pos_weight.begin(), pos_weight.end());
        double lambda_ub = result_lb.density * step_size;
        while (lambda_ub < pos_weight_ub) {
            auto result = RunQPBO(lambda_ub, false);
            if (result.undecided.size() <= ub_mip_bound) {
                auto [solution, success] = result.undecided.empty() ? make_pair(result.fixed_in, true) : RunMIP(result, lambda_ub, mip_time_limit);
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
            lambda_ub *= step_size;
        }
        return pos_weight_ub;
    }

    pair<vector<Vertex>, bool> RunMIP(QPBOResult& qpbo_result, double lambda, double mip_time_limit) {
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
            model.set(GRB_DoubleParam_TimeLimit, mip_time_limit);
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

    SubgraphResult DinkelbachBinary(const SubgraphResult& cep_result, double upper_bound, unsigned iterations, double epsilon, double mip_time_limit) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < iterations; iter++) {
            if ((epsilon > 0 ? (lower_bound / upper_bound) : (lower_bound - upper_bound)) >= epsilon) {
                break;
            }
            double lambda = (lower_bound + upper_bound) / 2.0;
            QPBOResult qpbo_result = RunQPBO(lambda, false);
            auto [solution, success] = qpbo_result.undecided.empty() ? make_pair(qpbo_result.fixed_in, true) : RunMIP(qpbo_result, lambda, mip_time_limit);
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

    SubgraphResult Dinkelbach(const SubgraphResult& cep_result, unsigned iterations, double mip_time_limit) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < iterations; iter++) {
            QPBOResult qpbo_result = RunQPBO(lower_bound, false);
            auto solution = qpbo_result.undecided.empty() ? qpbo_result.fixed_in : RunMIP(qpbo_result, lower_bound, mip_time_limit).first;
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

    SubgraphResult Run() {
        auto& qpbo_config = static_cast<Config&>(config);
        // Step 1. Result found by CEP as initial solution
        auto result_lb = FindLowerBound();
        // if qpbo_result.undecided is empty and fixed_in is empty, we can directly return result found by CEP as Optimal
        auto pre_qpbo = RunQPBO(result_lb.density, false);
        if (pre_qpbo.undecided.empty() && pre_qpbo.fixed_in.empty()) {
            return result_lb;
        } else if (pre_qpbo.undecided.empty()) {
            double density = ComputeDensity(pre_qpbo.fixed_in);
            if (density > result_lb.density) {
                result_lb = {pre_qpbo.fixed_in, density};
            }
        }
        
        if (qpbo_config.use_binary) {
            // Step 2. Find an upper bound for QPBO
            auto upper_bound = FindUpperBound(result_lb, qpbo_config.step_size, qpbo_config.ub_mip_bound, qpbo_config.mip_time_limit);
            // Step 3. Refine the solution by Dinkelbach
            return DinkelbachBinary(result_lb, upper_bound, qpbo_config.dinkelbach_iterations, qpbo_config.epsilon, qpbo_config.mip_time_limit);
        } else {
            return Dinkelbach(result_lb, qpbo_config.dinkelbach_iterations, qpbo_config.mip_time_limit);
        }      
    }
    
    void add_config_params(boost::json::object& config_obj) override {
        // First add CEP params
        CEP::add_config_params(config_obj);
        // Then add CEP_QPBO specific params
        auto& qpbo_config = static_cast<Config&>(config);
        config_obj["step_size"] = qpbo_config.step_size;
        config_obj["ub_mip_bound"] = qpbo_config.ub_mip_bound;
        config_obj["dinkelbach_iterations"] = qpbo_config.dinkelbach_iterations;
        config_obj["epsilon"] = qpbo_config.epsilon;
        config_obj["mip_time_limit"] = qpbo_config.mip_time_limit;
        config_obj["use_binary"] = qpbo_config.use_binary;
    }
};