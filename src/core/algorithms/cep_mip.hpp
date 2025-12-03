#include "cep.hpp"
#include <gurobi_c++.h>

class CEP_MIP : public CEP {
public:
    struct CEP_MIP_Config : public CEP_Config {
        bool use_binary = true;
        unsigned dinkelbach_iterations = 30;
        double epsilon = -0.00001;
        double mip_time_limit = 300.0;
        
        void load_from_json(const string& filename) override {
            CEP_Config::load_from_json(filename);
            
            ifstream file(filename);
            string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
            auto json = json::parse(content).as_object();
            
            if (json.contains("use_binary")) use_binary = json.at("use_binary").as_bool();
            if (json.contains("dinkelbach_iterations")) dinkelbach_iterations = json.at("dinkelbach_iterations").to_number<unsigned>();
            if (json.contains("epsilon")) epsilon = json.at("epsilon").to_number<double>();
            if (json.contains("mip_time_limit")) mip_time_limit = json.at("mip_time_limit").to_number<double>();
        }
        
        void add_to_json(json::object& cfg) const override {
            CEP_Config::add_to_json(cfg);
            cfg["use_binary"] = use_binary;
            cfg["dinkelbach_iterations"] = dinkelbach_iterations;
            cfg["epsilon"] = epsilon;
            cfg["mip_time_limit"] = mip_time_limit;
        }
    };

    CEP_MIP(const CEP_MIP_Config& cfg) : CEP(cfg) {}

    pair<vector<Vertex>, bool> RunMIP(const CEP_MIP_Config& cfg, double lambda) {
        try {
            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            env.start();
            GRBModel model = GRBModel(env);
            
            auto n = num_vertices(G);
            vector<GRBVar> x(n);
            for (size_t i = 0; i < n; i++) {
                x[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
            }
            
            GRBQuadExpr obj = 0.0;
            for (size_t i = 0; i < n; i++) {
                obj += (loop_weight[i] - lambda) * x[i];
            }
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                obj += G[*ei].weight * x[source(*ei, G)] * x[target(*ei, G)];
            }
            
            model.setObjective(obj, GRB_MAXIMIZE);
            model.set(GRB_DoubleParam_TimeLimit, cfg.mip_time_limit);
            model.optimize();
            vector<Vertex> solution;
            if (model.get(GRB_IntAttr_SolCount) > 0) {
                for (size_t i = 0; i < n; i++) {
                    if (x[i].get(GRB_DoubleAttr_X) > 0.5) {
                        solution.push_back(i);
                    }
                }
            }
            return {solution, model.get(GRB_IntAttr_Status) == GRB_OPTIMAL};
        } catch (GRBException& e) {
            throw runtime_error("Gurobi exception: " + string(e.getMessage()));
        }
    }

    SubgraphResult DinkelbachBinary(const CEP_MIP_Config& cfg, const SubgraphResult& cep_result, double upper_bound) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < cfg.dinkelbach_iterations; iter++) {
            if ((cfg.epsilon > 0 ? (lower_bound / upper_bound) : (lower_bound - upper_bound)) >= cfg.epsilon) {
                break;
            }
            auto lambda = (lower_bound + upper_bound) / 2.0;
            auto [solution, success] = RunMIP(cfg, lambda);
            // FIX: if MIP returns empty solution because of time limit, should not set upper_bound = lambda
            if (solution.empty() && success) {
                upper_bound = lambda;
            } else {
                auto density = ComputeDensity(solution);
                if (density > lower_bound) {
                    lower_bound = density;
                    best_solution = solution;
                } else {
                    // throw runtime_error("Dinkelbach: computed density is less than best density");
                    // Due to computation failure, or hit time limit, which means it cannot find a better solution under current lambda within time limit
                    // So we do not update lower_bound or upper_bound here
                    break;
                }
            }
        }
        return {best_solution, lower_bound};
    }

    SubgraphResult Dinkelbach(const CEP_MIP_Config& cfg, const SubgraphResult& cep_result) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < cfg.dinkelbach_iterations; iter++) {
            auto solution = RunMIP(cfg, lower_bound).first;
            auto density = ComputeDensity(solution);
            if (density > lower_bound) {
                lower_bound = density;
                best_solution = solution;
            } else {
                break;
            }
        }
        return {best_solution, lower_bound};
    }

    SubgraphResult Run(const CEP_MIP_Config& cfg) {
        auto cep_result = CEP::Run(cfg);
        if (cfg.use_binary) {
            auto upper_bound = *max_element(pos_weight.begin(), pos_weight.end());
            return DinkelbachBinary(cfg, cep_result, upper_bound);
        } else {
            return Dinkelbach(cfg, cep_result);
        }
    }
};