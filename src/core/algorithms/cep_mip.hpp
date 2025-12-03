#include "cep.hpp"
#include <gurobi_c++.h>

class CEP_MIP : public CEP {
public:
    struct Config : public CEP::Config {
        unsigned dinkelbach_iterations = 30;
        double epsilon = -0.00001;
        double mip_time_limit = 300.0;
        bool use_binary = true;
        
        void load_from_json(const string& filename) {
            CEP::Config::load_from_json(filename);
            
            std::ifstream file(filename);
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            auto json = boost::json::parse(content).as_object();
            
            if (json.contains("dinkelbach_iterations")) dinkelbach_iterations = json.at("dinkelbach_iterations").to_number<unsigned>();
            if (json.contains("epsilon")) epsilon = json.at("epsilon").to_number<double>();
            if (json.contains("mip_time_limit")) mip_time_limit = json.at("mip_time_limit").to_number<double>();
            if (json.contains("use_binary")) use_binary = json.at("use_binary").as_bool();
        }
    };

    CEP_MIP(const string& config_file) : CEP(config_file) {
        static_cast<Config&>(config).load_from_json(config_file);
    }

    pair<vector<Vertex>, bool> RunMIP(double lambda, double mip_time_limit) {
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
            model.set(GRB_DoubleParam_TimeLimit, mip_time_limit);
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

    SubgraphResult DinkelbachBinary(const SubgraphResult& cep_result, double upper_bound, unsigned iterations, double epsilon, double mip_time_limit) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < iterations; iter++) {
            if ((epsilon > 0 ? (lower_bound / upper_bound) : (lower_bound - upper_bound)) >= epsilon) {
                break;
            }
            auto lambda = (lower_bound + upper_bound) / 2.0;
            auto [solution, success] = RunMIP(lambda, mip_time_limit);
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

    SubgraphResult Dinkelbach(const SubgraphResult& cep_result, unsigned iterations, double mip_time_limit) {
        auto [best_solution, lower_bound] = cep_result;
        for (auto iter = 0; iter < iterations; iter++) {
            auto solution = RunMIP(lower_bound, mip_time_limit).first;
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

    SubgraphResult Run() {
        auto& mip_config = static_cast<Config&>(config);
        auto cep_result = CEP::Run();
        if (mip_config.use_binary) {
            auto upper_bound = *max_element(pos_weight.begin(), pos_weight.end());
            return DinkelbachBinary(cep_result, upper_bound, mip_config.dinkelbach_iterations, mip_config.epsilon, mip_config.mip_time_limit);
        } else {
            return Dinkelbach(cep_result, mip_config.dinkelbach_iterations, mip_config.mip_time_limit);
        }
    }
    
    void add_config_params(boost::json::object& config_obj) override {
        // First add CEP params
        CEP::add_config_params(config_obj);
        // Then add CEP_MIP specific params
        auto& mip_config = static_cast<Config&>(config);
        config_obj["dinkelbach_iterations"] = mip_config.dinkelbach_iterations;
        config_obj["epsilon"] = mip_config.epsilon;
        config_obj["mip_time_limit"] = mip_config.mip_time_limit;
        config_obj["use_binary"] = mip_config.use_binary;
    }
};