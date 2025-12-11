#include "cep.hpp"
#include "../../external/QPBO/QPBO.h"
#include <gurobi_c++.h>

class EXACT : public CEP {
public:
    struct EXACT_Config : public CEP_Config {
        // For finding upper bound
        double step_size = 1.02;
        // Direct MIP bound
        unsigned direct_mip_bound = 100;
        // Binary search related params
        unsigned dinkelbach_iterations = 100;
        double epsilon = -0.001;
        // MIP related params
        double mip_time_limit = 600.0;

        // Components toggle for ablation study
        bool enable_cep_init = false;
        bool enable_binary_search = false;
        bool enable_qpbo = false;
        bool enable_qpbo_probe = false;
        bool enable_graph_pruning = false;
        bool enable_pruning_set = false;
        bool enable_cep_middle = false;
        bool enable_cep_lambda = false;
        bool enable_qpboi = false;
        bool enable_mip_init = false;
        bool enable_mip_constrains_vertex_lb = false;
        bool enable_mip_constrains_vertex_ub = false;
        bool enable_cep_final = false;
        
        void load_from_json(const string& filename) override {
            CEP_Config::load_from_json(filename);
            
            ifstream file(filename);
            string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
            auto json = json::parse(content).as_object();

            if (json.contains("step_size")) step_size = json.at("step_size").to_number<double>();
            if (json.contains("direct_mip_bound")) direct_mip_bound = json.at("direct_mip_bound").to_number<unsigned>();
            if (json.contains("dinkelbach_iterations")) dinkelbach_iterations = json.at("dinkelbach_iterations").to_number<unsigned>();
            if (json.contains("epsilon")) epsilon = json.at("epsilon").to_number<double>();
            if (json.contains("mip_time_limit")) mip_time_limit = json.at("mip_time_limit").to_number<double>();

            if (json.contains("enable_cep_init")) enable_cep_init = json.at("enable_cep_init").as_bool();
            if (json.contains("enable_binary_search")) enable_binary_search = json.at("enable_binary_search").as_bool();
            if (json.contains("enable_qpbo")) enable_qpbo = json.at("enable_qpbo").as_bool();
            if (json.contains("enable_qpbo_probe")) enable_qpbo_probe = json.at("enable_qpbo_probe").as_bool();
            if (json.contains("enable_graph_pruning")) enable_graph_pruning = json.at("enable_graph_pruning").as_bool();
            if (json.contains("enable_pruning_set")) enable_pruning_set = json.at("enable_pruning_set").as_bool();
            if (json.contains("enable_cep_middle")) enable_cep_middle = json.at("enable_cep_middle").as_bool();
            if (json.contains("enable_cep_lambda")) enable_cep_lambda = json.at("enable_cep_lambda").as_bool();
            if (json.contains("enable_qpboi")) enable_qpboi = json.at("enable_qpboi").as_bool();
            if (json.contains("enable_mip_init")) enable_mip_init = json.at("enable_mip_init").as_bool();
            if (json.contains("enable_mip_constrains_vertex_lb")) enable_mip_constrains_vertex_lb = json.at("enable_mip_constrains_vertex_lb").as_bool();
            if (json.contains("enable_mip_constrains_vertex_ub")) enable_mip_constrains_vertex_ub = json.at("enable_mip_constrains_vertex_ub").as_bool();
            if (json.contains("enable_cep_final")) enable_cep_final = json.at("enable_cep_final").as_bool();
        }
        
        void add_to_json(json::object& cfg) const override {
            CEP_Config::add_to_json(cfg);

            cfg["step_size"] = step_size;
            cfg["direct_mip_bound"] = direct_mip_bound;
            cfg["dinkelbach_iterations"] = dinkelbach_iterations;
            cfg["epsilon"] = epsilon;
            cfg["mip_time_limit"] = mip_time_limit;

            cfg["enable_cep_init"] = enable_cep_init;
            cfg["enable_binary_search"] = enable_binary_search;
            cfg["enable_qpbo"] = enable_qpbo;
            cfg["enable_qpbo_probe"] = enable_qpbo_probe;
            cfg["enable_graph_pruning"] = enable_graph_pruning;
            cfg["enable_pruning_set"] = enable_pruning_set;
            cfg["enable_cep_middle"] = enable_cep_middle;
            cfg["enable_cep_lambda"] = enable_cep_lambda;
            cfg["enable_qpboi"] = enable_qpboi;
            cfg["enable_mip_init"] = enable_mip_init;
            cfg["enable_mip_constrains_vertex_lb"] = enable_mip_constrains_vertex_lb;
            cfg["enable_mip_constrains_vertex_ub"] = enable_mip_constrains_vertex_ub;
            cfg["enable_cep_final"] = enable_cep_final;
        }
    };

private:
    enum class Indicator {
        QPBO_UB,
        QPBO_LB,
        UNDER_MIP_DIRECT_UB,
        CEP_DENSITY_LB,
        QPBOI_LB,
        MIP_INDIRECT_WITH_INIT,
        MIP_INDIRECT_NO_INIT,
        MIP_INDIRECT_WITH_HEURISTIC,
        NO_MIP,
        Not_Assigned
    };

    struct SubgraphResultEnhanced {
        vector<Vertex> nodes;
        double density;
        bool exact = false;
        Indicator info = Indicator::Not_Assigned;

        SubgraphResultEnhanced(vector<Vertex> nodes, double density) : nodes(nodes), density(density) {}
        SubgraphResultEnhanced(SubgraphResult r) : nodes(r.nodes), density(r.density) {}
        SubgraphResultEnhanced(SubgraphResult r, bool exact, Indicator info) : nodes(r.nodes), density(r.density), exact(exact), info(info) {}
        SubgraphResultEnhanced(vector<Vertex> nodes, double density, bool exact, Indicator info) : nodes(nodes), density(density), exact(exact), info(info) {}
    };

    struct QPBOResult {
        vector<int> labels;
        vector<Vertex> fixed_in;
        vector<Vertex> fixed_out;
        vector<Vertex> undecided;

        QPBOResult() = default;
        QPBOResult(const vector<bool> valid) {
            labels = vector<int>(valid.size(), 0);
            for (auto i = 0; i < valid.size(); i++) {
                if (valid[i]) {
                    undecided.push_back(i);
                    labels[i] = -1;
                }
            }
        }
    };
    using REAL = double;
    size_t vertex_lower_bound, vertex_upper_bound;
    SubgraphResult naive_lb;

public:
    EXACT(const EXACT_Config& cfg) : CEP(cfg) {
        InitializePositiveWeights();
        if (cfg.enable_mip_constrains_vertex_lb) {
            vertex_lower_bound = 3;
            naive_lb = NaiveLowerBound();
        }
        if (cfg.enable_mip_constrains_vertex_ub) {
            vertex_upper_bound = num_vertices(G);
        }
    }

    QPBOResult RunQPBO(const EXACT_Config& cfg, double lambda, bool improve, vector<Vertex> init_label = {}) {
        LOG("RunQPBO: lambda = " << lambda << ", improve = " << improve << ", init_label size = " << init_label.size());
        size_t n = num_vertices(G);
        unique_ptr<QPBO<REAL>> qpbo(new QPBO<REAL>(valid_count, 2 * num_edges(G)));
        qpbo->AddNode(valid_count);
        vector<size_t> node_to_id(num_vertices(G), -1);
        vector<size_t> id_to_node;
        id_to_node.reserve(valid_count);
        
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            if (valid[*vi]) {
                auto id = id_to_node.size();
                qpbo->AddUnaryTerm(id, 0.0, lambda-loop_weight[*vi]);
                node_to_id[*vi] = id;
                id_to_node.push_back(*vi);
            }
        }
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            auto u = source(*ei, G);
            auto v = target(*ei, G);
            if (valid[u] && valid[v]) {
                qpbo->AddPairwiseTerm(node_to_id[u], node_to_id[v], 0, 0, 0, -G[*ei].weight);
            }
        }
        
        QPBOResult result;
        result.labels = vector<int>(n, 0);
        
        if (cfg.enable_qpbo_probe) {
            // Use Probe to potentially fix more nodes
            vector<int> mapping(valid_count);
            QPBO<REAL>::ProbeOptions options;
            options.weak_persistencies = 1;
            options.directed_constraints = 2;
            
            qpbo->Probe(mapping.data(), options);
            // // Probe() internally calls Solve() at the end, so we can call ComputeWeakPersistencies()
            // qpbo->ComputeWeakPersistencies();
            
            // After Probe, the problem is transformed. Use mapping to interpret results.
            for (size_t i = 0; i < valid_count; i++) {
                auto i2n = id_to_node[i];
                if (mapping[i] < 2) {
                    // Node i was fixed to label mapping[i]
                    result.labels[i2n] = mapping[i];
                    if (mapping[i] == 0) {
                        result.fixed_out.push_back(i2n);
                    } else {
                        result.fixed_in.push_back(i2n);
                    }
                } else {
                    // Node i maps to new node (mapping[i]/2) with possible inversion
                    int new_node = mapping[i] / 2;
                    int inversion = mapping[i] % 2;
                    int new_label = qpbo->GetLabel(new_node);
                    
                    if (new_label >= 0) {
                        result.labels[i2n] = (new_label + inversion) % 2;
                        if (result.labels[i2n] == 0) {
                            result.fixed_out.push_back(i2n);
                        } else {
                            result.fixed_in.push_back(i2n);
                        }
                    } else {
                        result.labels[i2n] = -1;
                        result.undecided.push_back(i2n);
                    }
                }
            }
        } else if (!improve) {
            // Standard QPBO without Probe
            qpbo->Solve();
            qpbo->ComputeWeakPersistencies();
            
            for (size_t i = 0; i < valid_count; i++) {
                int label = qpbo->GetLabel(i);
                auto i2n = id_to_node[i];
                if (label == 0) {
                    result.labels[i2n] = 0;
                    result.fixed_out.push_back(i2n);
                } else if (label == 1) {
                    result.labels[i2n] = 1;
                    result.fixed_in.push_back(i2n);
                } else {
                    result.labels[i2n] = -1;
                    result.undecided.push_back(i2n);
                }
            }
        } else {
            // QPBOI improvement, set initial labels
            for (size_t i = 0; i < valid_count; i++) {
                qpbo->SetLabel(i, 0);
            }
            for (auto n : init_label) {
                qpbo->SetLabel(node_to_id[n], 1);
            }

            int order_array[pruning_set.size()];
            auto i = 0;
            for (auto it = pruning_set.rbegin(); it != pruning_set.rend(); ++it) {
                order_array[i++] = node_to_id[it->node];
            }

            qpbo->Improve(pruning_set.size(), order_array);

            for (size_t i = 0; i < valid_count; i++) {
                int label = qpbo->GetLabel(i);
                auto i2n = id_to_node[i];
                if (label == 0) {
                    result.labels[i2n] = 0;
                    result.fixed_out.push_back(i2n);
                } else if (label == 1) {
                    result.labels[i2n] = 1;
                    result.fixed_in.push_back(i2n);
                } else {
                    result.labels[i2n] = -1;
                    result.undecided.push_back(i2n);
                }
            }
        }

        return result;
    }

    SubgraphResult NaiveLowerBound() {
        SubgraphResult lb = {{}, 0.0};
        auto max_loop_weight_it = max_element(loop_weight.begin(), loop_weight.end());
        if (*max_loop_weight_it > lb.density) {
            lb = {{static_cast<size_t>(max_loop_weight_it - loop_weight.begin())}, *max_loop_weight_it};
        }
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            if (G[*ei].weight > 0) {
                auto u = source(*ei, G);
                auto v = target(*ei, G);
                auto density = (loop_weight[u] + loop_weight[v] + G[*ei].weight) / 2.0;
                if (density > lb.density) {
                    lb = {{u, v}, density};
                }
            }
        }
        return lb;
    }

    pair<SubgraphResult, bool> FindLowerBound(const EXACT_Config& cfg) {
        // 1. Run CEP to get initial solution
        auto result_lb = SubgraphResult{{}, 0.0};
        if (cfg.enable_cep_init) {
            
            result_lb = CEP::Run(cfg);
            CEP::Reset(true);
            LOG("FindLowerBound: CEP init lb: " << result_lb.density);
        }
        // 2. Need to find better lower bound among single nodes and edges if set vertex lb
        if (cfg.enable_mip_constrains_vertex_lb || !cfg.enable_cep_init) {
            if (naive_lb.density > result_lb.density) {
                result_lb = naive_lb;
                LOG("FindLowerBound: Updated to naive lb: " << result_lb.density);
            }
        }
        // 3. Run QPBO to check whether it is already optimal, or update lower bound if possible
        if (cfg.enable_graph_pruning) Pruning({}, result_lb.density);
        if (cfg.enable_qpbo) {
            auto result = QPBO_CEP_MIP(cfg, result_lb, result_lb.density, false);
            switch (result.info) {
                case Indicator::QPBO_UB:
                case Indicator::QPBO_LB:
                case Indicator::UNDER_MIP_DIRECT_UB:
                    // (1). Optimal found
                    if (result.nodes.empty() && result.exact) {
                        LOG("FindLowerBound: Optimal found.");
                        return {result_lb, true};
                    }

                    // (2). Have found a better lower bound
                    if (result.density > result_lb.density) {
                        result_lb = {result.nodes, result.density};
                        if (cfg.enable_mip_constrains_vertex_ub && result.exact) {
                            vertex_upper_bound = result.nodes.size() - 1;
                        }
                        LOG("FindLowerBound: Updated to QPBO_MIP lb: " << result_lb.density);
                        return {result_lb, false};
                    }
                case Indicator::NO_MIP:
                    // (3). MIP not succeeded, or have more than direct_mip_bound undecided nodes in QPBO
                    LOG("FindLowerBound: MIP failed or not run due to large undecided nodes");
                    return {result_lb, false};
                default:
                    string msg = "FindLowerBound Bug: Should only use QPBO and MIP in the FindLowerBound function.";
                    info += msg + " | ";
                    throw runtime_error(msg);
            }
        }
        return {result_lb, false};
    }

    double FindUpperBound(const EXACT_Config& cfg, SubgraphResult& result_lb) {
        assert(cfg.step_size > 1.0); // step_size should be larger than 1.0
        // 1. The naive way for upper bound is the maximum among positive weight sum of edges incident to each vertex
        auto upper_bound = (cfg.enable_pruning_set && pruning_set_on) ? pruning_set.rbegin()->key : *max_element(pos_weight.begin(), pos_weight.end());
        LOG("FindUpperBound: Naive upper bound: " << upper_bound);

        // 2. try to find a tighter upper bound by increasing from lower bound step by step
        auto lambda = result_lb.density * cfg.step_size;
        while (lambda < upper_bound) {
            auto result = QPBO_CEP_MIP(cfg, result_lb, lambda, false);
            switch (result.info) {
                case Indicator::QPBO_UB:
                case Indicator::QPBO_LB:
                case Indicator::UNDER_MIP_DIRECT_UB:
                    // (1). The density of any subgraph must be <= lambda
                    if (result.nodes.empty() && result.exact) {
                        return lambda;
                    }

                    // (2). Have found a better lower bound
                    if (result.density > result_lb.density) {
                        result_lb = {result.nodes, result.density};
                        if (cfg.enable_graph_pruning) Pruning({}, result.density);
                        if (result.exact) {
                            if (cfg.enable_mip_constrains_vertex_ub) {
                                vertex_upper_bound = result.nodes.size() - 1;
                            }
                            if (cfg.enable_mip_constrains_vertex_lb) {
                                upper_bound = min(upper_bound, lambda + result.nodes.size() * (result.density - lambda) / vertex_lower_bound);
                            }
                            lambda = result.density; // if the result is exact, result.density will be >= lambda
                        } else {
                            lambda = max(lambda, result.density); // if the result is not exact, result.density may be <= lambda_ub
                        }
                    }

                    // (3). Cannot find better lower bound, try next lambda_ub
                    break;
                case Indicator::NO_MIP:
                    // (4). Have more than direct_mip_bound undecided nodes in QPBO and did not run MIP, try next lambda_ub
                    break;
                default:
                    string msg = "FindUpperBound Bug: Should only use QPBO and MIP when undecided nodes are small in the FindUpperBound function.";
                    info += msg + " | ";
                    throw runtime_error(msg);
            }
            lambda *= cfg.step_size; // try next lambda
        }

        // 3. No tighter upper bound found, return the naive one
        return upper_bound;
    }

    pair<vector<Vertex>, bool> RunMIP(const EXACT_Config& cfg, QPBOResult& qpbo_result, double lambda, vector<size_t> initial_solution = {}) {
        LOG("RunMIP: Have " << qpbo_result.undecided.size() << " undecided nodes.");
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
            for (auto i : qpbo_result.undecided) {
                obj += (lambda - loop_weight[i]) * undecided_vars[i];
            }   

            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                auto u = source(*ei, G);
                auto v = target(*ei, G);
                if (!valid[u] || !valid[v]) continue;  // Skip invalid nodes
                auto w = G[*ei].weight;
                auto label_u = qpbo_result.labels[u];
                auto label_v = qpbo_result.labels[v];
                if (label_u == -1 && label_v == -1) {
                    obj += -w * undecided_vars[u] * undecided_vars[v];
                } else if (label_u == -1 && label_v == 1) {
                    obj += -w * undecided_vars[u];
                } else if (label_v == -1 && label_u == 1) {
                    obj += -w * undecided_vars[v];
                }
            }

            // Note: add vertex count constraint
            GRBLinExpr vertex_sum = 0.0;
            for (auto i : qpbo_result.undecided) {
                vertex_sum += undecided_vars[i];
            }
            if (cfg.enable_mip_constrains_vertex_ub) {
                model.addConstr(vertex_sum <= (vertex_upper_bound - qpbo_result.fixed_in.size()));
            }
            // Warn: Don't set it, because it will return non-empty result even if the lambda >= largest density
            // The non-empty result cannot reflect correct upper bound for Dinkelbach, will make the Dinkelbach break.
            // if (set_vertex_lb) {
            //     model.addConstr(vertex_sum >= max(static_cast<size_t>(0), vertex_lower_bound - qpbo_result.fixed_in.size()));
            // }

            // ============== Set initial solution if provided ==============
            if (cfg.enable_mip_init) {
                for (auto i : qpbo_result.undecided) {
                    undecided_vars[i].set(GRB_DoubleAttr_Start, 0);
                }
                for (auto i : initial_solution) {
                    undecided_vars[i].set(GRB_DoubleAttr_Start, 1); 
                }
            }
            
            model.setObjective(obj, GRB_MINIMIZE);
            model.set(GRB_DoubleParam_TimeLimit, cfg.mip_time_limit);
            model.optimize();
            vector<Vertex> selected = qpbo_result.fixed_in;
            if (model.get(GRB_IntAttr_SolCount) > 0) {
                for (auto i : qpbo_result.undecided) {
                    if (undecided_vars[i].get(GRB_DoubleAttr_X) > 0.5) {
                        selected.push_back(i);
                    }
                }
            }
            if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
                string msg = "RunMIP Fail: MIP timeout.";
                info += msg + " | ";
                LOG("RunMIP: MIP timeout.");
            }
            return {selected, model.get(GRB_IntAttr_Status) == GRB_OPTIMAL};

        } catch (GRBException& e) {
            string msg = "RunMIP Bug: " + string(e.getMessage());
            info += msg + " | ";
            throw runtime_error(msg);
        }
    }

    SubgraphResult CEPDensity(const EXACT_Config& cfg, const QPBOResult& qpbo_result, unsigned max_local_optima, const vector<Vertex>& mip_result = {}, double density = 0.0) {
        auto best = SubgraphResult{{}, 0.0};
        auto valid_original = valid;
        auto pos_weight_original = pos_weight;
        unsigned max_neg = 0;
        if (mip_result.empty()) {
            for (auto i : qpbo_result.fixed_out) {
                valid[i] = false;
            }
            max_neg = qpbo_result.fixed_in.size() + qpbo_result.undecided.size();
        } else {
            valid = vector<bool>(num_vertices(G), false);
            for (auto v : mip_result) {
                valid[v] = true;
            }
            best = SubgraphResult{mip_result, density};
            max_neg = mip_result.size();
        }
        
        pos_weight = vector<double>(num_vertices(G), -numeric_limits<double>::infinity());
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            if (!valid[*vi]) continue;
            double total_pos_weight = loop_weight[*vi];
            for (auto [ei, ee] = out_edges(*vi, G); ei != ee; ++ei) {
                auto neighbor = target(*ei, G);
                if (valid[neighbor] && G[*ei].weight > 0) {
                    total_pos_weight += G[*ei].weight;
                }
            }
            pos_weight[*vi] = total_pos_weight;
        }
        for (unsigned it = 0; it < max_local_optima; ++it) {
            auto anchor = std::distance(pos_weight.begin(), max_element(pos_weight.begin(), pos_weight.end()));
            if (anchor == Traits::null_vertex() || !valid[anchor]) break;
            auto result = LocalGreedy(cfg, anchor);
            if (result.density > best.density) {
                best = result;
            }
            PruningVector(result.nodes, best.density, false, false);
        }
        valid = valid_original;
        pos_weight = pos_weight_original;
        return best;
    }

    SubgraphResult CEPLambda(const EXACT_Config& cfg, const QPBOResult& qpbo_result, double lambda) {
        auto best = SubgraphResult{{}, 0.0};
        auto valid_original = valid;
        auto pos_weight_original = pos_weight;
        for (auto i : qpbo_result.fixed_out) {
            valid[i] = false;
        }
        if (!qpbo_result.fixed_in.empty()) {
            best = CEPLambdaLocal(cfg, qpbo_result, 0, qpbo_result.fixed_in.size() + qpbo_result.undecided.size(), lambda);
        } else {
            pos_weight = vector<double>(num_vertices(G), -numeric_limits<double>::infinity());
            for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
                if (!valid[*vi]) continue;
                double total_pos_weight = loop_weight[*vi];
                for (auto [ei, ee] = out_edges(*vi, G); ei != ee; ++ei) {
                    auto neighbor = target(*ei, G);
                    if (valid[neighbor] && G[*ei].weight > 0) {
                        total_pos_weight += G[*ei].weight;
                    }
                }
                pos_weight[*vi] = total_pos_weight;
            }
            for (unsigned it = 0; it < cfg.max_local_optima; ++it) {
                auto anchor = std::distance(pos_weight.begin(), max_element(pos_weight.begin(), pos_weight.end()));
                if (anchor == Traits::null_vertex() || !valid[anchor]) break;
                auto result = CEPLambdaLocal(cfg, qpbo_result, anchor, qpbo_result.fixed_in.size() + qpbo_result.undecided.size(), lambda);
                if (result.density > best.density) {
                    best = result;
                }
                PruningVector(result.nodes, 0, false, true);
            }
        }
        valid = valid_original;
        pos_weight = pos_weight_original;
        return best;
    }
    
    SubgraphResult CEPLambdaLocal(const EXACT_Config& cfg, const QPBOResult& qpbo_result, Vertex anchor, unsigned max_neg, double lambda) {
        // ============== Clear and initialization ==============
        fill(status.begin(), status.end(), Status::Out);
        fill(neighbor_in_count.begin(), neighbor_in_count.end(), 0);
        double current_weight_sum = 0.0;
        double max_f = -numeric_limits<double>::infinity();
        MaxHeap selected, fringe, best;
        vector<MaxHeap::handle_type> handles(num_vertices(G));
        unsigned neg_count = 0;
        Vertex next = Traits::null_vertex();
        if (qpbo_result.fixed_in.empty()) {
            next = anchor;
            status[anchor] = Status::Fringe;
            handles[anchor] = fringe.push({loop_weight[anchor], anchor});
        } else {
            std::unordered_map<Vertex, double> fringe_init;
            for (auto v : qpbo_result.fixed_in) {
                double init_weight_sum = loop_weight[v];
                for (auto [ei, ee] = out_edges(v, G); ei != ee; ++ei) {
                    if (!valid[target(*ei, G)]) continue;
                    if (qpbo_result.labels[v] == 1) {
                        init_weight_sum += G[*ei].weight;
                        neighbor_in_count[v] += 1;
                        if (v < target(*ei, G)) {
                            current_weight_sum += G[*ei].weight;
                        }
                    } else if (qpbo_result.labels[v] == -1) {
                        auto neighbor = target(*ei, G);
                        if (fringe_init.find(neighbor) == fringe_init.end()) {
                            fringe_init[neighbor] = G[*ei].weight + loop_weight[neighbor];
                        } else {
                            fringe_init[neighbor] += G[*ei].weight;
                        }
                        neighbor_in_count[neighbor] += 1;
                    }
                }
                status[v] = Status::In;
                handles[v] = selected.push({-init_weight_sum, v});
            }
            for (auto [node, key] : fringe_init) {
                status[node] = Status::Fringe;
                handles[node] = fringe.push({key, node});
            }
            best = selected;
            max_f = current_weight_sum - selected.size() * lambda;
            next = fringe.empty() ? Traits::null_vertex() : fringe.top().node;
        }

        // ============== Main loop ==============
        while (next != Traits::null_vertex() && neg_count < max_neg) {
            if (status[next] == Status::Fringe) {
                // ============== If node is "fringe" → move it to "in" ==============
                status[next] = Status::In;
                auto item = fringe.top();
                fringe.pop();
                handles[next] = selected.push({-item.key, next});
                current_weight_sum += item.key;
                for (auto [ei, ee] = out_edges(next, G); ei != ee; ++ei) {
                    auto neighbor = target(*ei, G);
                    if (!valid[neighbor]) continue;

                    double edge_weight = G[*ei].weight;
                    neighbor_in_count[neighbor] += 1;
                    if (status[neighbor] == Status::Out) {
                        // Move out → fringe
                        status[neighbor] = Status::Fringe;
                        double priority_key = edge_weight + loop_weight[neighbor];
                        handles[neighbor] = fringe.push({priority_key, neighbor});
                    } else if (status[neighbor] == Status::Fringe) {
                        // Update fringe neighbor's key
                        auto h = handles[neighbor];
                        fringe.update(h, {(*h).key + edge_weight, neighbor});
                    } else if (status[neighbor] == Status::In) {
                        // Update selected neighbor's key
                        auto h = handles[neighbor];
                        selected.update(h, {(*h).key - edge_weight, neighbor});
                    } else {
                        throw runtime_error("Invalid status in LocalGreedy");
                    }
                }
            } else if (status[next] == Status::In) {
                // ============== If node is "in" → move it to "fringe" ==============
                status[next] = Status::Fringe;
                auto item = selected.top();
                selected.pop();
                handles[next] = fringe.push({-item.key, next});
                current_weight_sum += item.key;
                for (auto [ei, ee] = out_edges(next, G); ei != ee; ++ei) {
                    auto neighbor = target(*ei, G);
                    if (!valid[neighbor]) continue;

                    double edge_weight = G[*ei].weight;
                    neighbor_in_count[neighbor] -= 1;
                    if (status[neighbor] == Status::Fringe) {
                        // Possibly move fringe → out if in_neighbor_count == 0
                        if (neighbor_in_count[neighbor] == 0) {
                            status[neighbor] = Status::Out;
                            fringe.erase(handles[neighbor]);
                            handles[neighbor] = MaxHeap::handle_type();
                        } else {
                            auto new_key = (*handles[neighbor]).key - edge_weight;
                            fringe.update(handles[neighbor], {new_key, neighbor});
                        }
                    } else if (status[neighbor] == Status::In) {
                        // Update selected neighbor's key
                        auto h = handles[neighbor];
                        selected.update(h, {(*h).key + edge_weight, neighbor});
                    } else {
                        throw runtime_error("Invalid status in LocalGreedy");
                    }
                }
            } else {
                throw runtime_error("Invalid status in LocalGreedy");
            }

            // ============== Compute the objective function ==============
            auto selected_count = selected.size();
            double f_value = current_weight_sum - selected_count * lambda;
            max_f = max(max_f, f_value);

            // ============== Compute marginal gains for top of each heap ==============
            double best_mg = -numeric_limits<double>::infinity();
            Vertex best_node = Traits::null_vertex();
            bool best_is_addition = false;
            if (!selected.empty()) {
                auto top_item = selected.top();
                best_mg = top_item.key + lambda;
                best_node = top_item.node;
            }
            if (!fringe.empty()) {
                auto top_item = fringe.top();
                double mg = top_item.key - lambda;
                if (mg > best_mg) {
                    best_mg = mg;
                    best_node = top_item.node;
                    best_is_addition = true;
                }
            }

            // ============== Determine next node ==============
            if (best_node == Traits::null_vertex()) {
                next = Traits::null_vertex();
            } else {
                if ((f_value + best_mg) <= max_f || next == best_node) {
                    neg_count++;
                    next = best_is_addition ? best_node : (fringe.empty() ? Traits::null_vertex() : fringe.top().node);
                } else {
                    neg_count = 0;
                    next = best_node;
                }
                if (f_value >= max_f) {
                    if (best_mg <= 0 || next == Traits::null_vertex()) {
                        best = selected;
                    }
                }
            }
        }

        // ============== Peeling phase: remove nodes to find maximum density ==============
        vector<Vertex> remove_order;
        remove_order.reserve(selected.size()); 
        size_t best_step = 0;
        while (!selected.empty()) {
            auto top_item = selected.top();
            selected.pop();
            remove_order.push_back(top_item.node);
            status[top_item.node] = Status::Out;
            for (auto [ei, ee] = out_edges(top_item.node, G); ei != ee; ++ei) {
                auto neighbor = target(*ei, G);
                if (!valid[neighbor] || status[neighbor] != Status::In) continue;
                auto h = handles[neighbor];
                selected.update(h, {(*h).key + G[*ei].weight, neighbor});
            }
            current_weight_sum += top_item.key;
            if (!selected.empty()) {
                double current_density = current_weight_sum - selected.size() * lambda;
                if (current_density > max_f) {
                    max_f = current_density;
                    best_step = remove_order.size();
                }
            }
        }
        if (best_step != 0) {
            return {{remove_order.begin() + best_step, remove_order.end()}, max_f};
        } else {
            vector<Vertex> best_nodes;
            for (auto& item : best) {
                best_nodes.push_back(item.node);
            }
            return {best_nodes, max_f};
        }
    }

    // Return: the result(solution node set and density),  whether it is exact, and the method name used to get the result
    SubgraphResultEnhanced QPBO_CEP_MIP(const EXACT_Config& cfg, const SubgraphResult& lb, double lambda, bool handle_large_undecided, double* upper_bound = nullptr) {
        // Run QPBO
        QPBOResult qpbo_result;
        if (cfg.enable_qpbo) {
            qpbo_result = RunQPBO(cfg, lambda, false);
            if (qpbo_result.undecided.empty()) { // QPBO fixed all nodes
                if (qpbo_result.fixed_in.empty()) { // all nodes are labeled as 0
                    // 1. if QPBO labels all nodes as 0, lambda is a new upper bound
                    LOG("QPBO_CEP_MIP: QPBO fixed all nodes as 0, lambda is a new upper bound: " << lambda);
                    return {SubgraphResult{}, true, Indicator::QPBO_UB};
                } else { // some are labeled as 1
                    // 2. if QPBO labels some nodes as 1, compute new density as lower bound
                    auto density = ComputeDensity(qpbo_result.fixed_in); // will be >= lambda
                    LOG("QPBO_CEP_MIP: QPBO fixed all nodes, new lower bound density: " << density);
                    return {qpbo_result.fixed_in, density, true, Indicator::QPBO_LB};
                }
            }
        } else {
            qpbo_result = QPBOResult(valid); // all valid nodes are labeled undecided
            LOG("QPBO_CEP_MIP: QPBO not enabled, all valid nodes are labeled undecided.");
        }

        // QPBO has undecided nodes or not enabled
        auto have_large_undecided = qpbo_result.undecided.size() > cfg.direct_mip_bound;
        if (have_large_undecided && !handle_large_undecided) {
            // 3. too many undecided nodes, skip MIP
            LOG("QPBO_CEP_MIP: Too many undecided nodes, skip MIP.");
            return {SubgraphResult{}, false, Indicator::NO_MIP};
        }

        // Compute MIP initialization when have large number of undecided nodes and want to handle them
        vector<Vertex> mip_init = {};
        if (have_large_undecided && handle_large_undecided) { // there are many undecided nodes and want to handle them
            if (cfg.enable_cep_middle) {
                auto cep_density_result = CEPDensity(cfg, qpbo_result, cfg.max_local_optima); // try to use CEPDensity to get better lower bound
                if (cep_density_result.density > lb.density) { // CEPDensity gets new lower bound
                    // 4. if CEPDensity gets new lower bound, return to update lower bound
                    LOG("QPBO_CEP_MIP: CEPDensity gets new lower bound: " << cep_density_result.density << " rather than " << lb.density);
                    return {cep_density_result, false, Indicator::CEP_DENSITY_LB};
                }
            }
            if (cfg.enable_mip_init) { // CEPDensity cannot get better lower bound, run CEPLambda and QPBOI on undecided nodes if allowed
                auto cep_lambda_result = cfg.enable_cep_lambda ? CEPLambda(cfg, qpbo_result, lambda) : SubgraphResult{qpbo_result.fixed_in, 0.0}; // get some initial solution from CEPLambda based on lambda
                // cout << "CEP Lambda result density: " << cep_lambda_result.density << endl;
                mip_init = cfg.enable_qpboi ? RunQPBO(cfg, lambda, true, cep_lambda_result.nodes).fixed_in : cep_lambda_result.nodes; // run QPBOI to improve the initial solution
                auto density = ComputeDensity(mip_init);
                if (density > lb.density) { // QPBOI gets new lower bound
                    // 5. if QPBOI gets new lower bound, return to update lower bound
                    LOG("QPBO_CEP_MIP: QPBOI gets new lower bound: " << density << " rather than " << lb.density);
                    return {mip_init, density, false, Indicator::QPBOI_LB};
                }
            }
        }

        // Undecided nodes are small, or handle large undecided nodes
        auto mip_result = RunMIP(cfg, qpbo_result, lambda, mip_init);
        auto density = ComputeDensity(mip_result.first);
        
        // Run CEP to try to improve the MIP result
        if (have_large_undecided && cfg.enable_cep_final && !mip_result.first.empty()) {
            // 6. run CEPDensity after MIP to try to further improve the solution
            auto cep_after_result = CEPDensity(cfg, qpbo_result, 1, mip_result.first, density); // run CEPDensity
            if (cep_after_result.density > density) { // CEPDensity after MIP gets better lower bound
                if (mip_result.second) { // update the constrains for MIP if MIP is optimal
                    if (cfg.enable_mip_constrains_vertex_ub) {
                        vertex_upper_bound = mip_result.first.size() - 1;
                    }
                    if (cfg.enable_mip_constrains_vertex_lb) {
                        *upper_bound = min(*upper_bound, lambda + mip_result.first.size() * (density - lambda) / vertex_lower_bound);
                    }
                }
                LOG("QPBO_CEP_MIP: CEPDensity after MIP gets new lower bound: " << cep_after_result.density << " rather than " << density);
                return {cep_after_result, false, Indicator::MIP_INDIRECT_WITH_HEURISTIC};
            }
        }

        // 7. if have small undecided nodes, or want to handle large undecided, return result from MIP
        auto indicator = have_large_undecided ? (cfg.enable_mip_init ? Indicator::MIP_INDIRECT_WITH_INIT : Indicator::MIP_INDIRECT_NO_INIT) : Indicator::UNDER_MIP_DIRECT_UB;
        LOG("QPBO_CEP_MIP: RunMIP on undecided nodes " << cfg.enable_mip_init ? "with" : "without" << " initialization, density: " << density);
        return {mip_result.first, density, mip_result.second, indicator};
    }

    bool Terminate(const EXACT_Config& cfg, double lower_bound, double upper_bound) {
        return (cfg.epsilon > 0 ? (lower_bound / upper_bound) : (lower_bound - upper_bound)) >= cfg.epsilon;
    }

    SubgraphResult DinkelbachBinary(const EXACT_Config& cfg, SubgraphResult& result_lb, double upper_bound) {
        for (auto iter = 0; iter < cfg.dinkelbach_iterations; iter++) {
            LOG("DinkelbachBinary: it " << iter << ": lb = " << result_lb.density << ", ub = " << upper_bound);
            if (Terminate(cfg, result_lb.density, upper_bound)) break;
            auto lambda = (result_lb.density + upper_bound) / 2.0;
            auto result = QPBO_CEP_MIP(cfg, result_lb, lambda, true, &upper_bound);
            switch (result.info) {
                case Indicator::QPBO_UB: // find a tighter upper bound
                case Indicator::QPBO_LB: // find a tighter lower bound
                case Indicator::UNDER_MIP_DIRECT_UB: // if success, can know whether lambda is an upper bound or get a better lower bound; otherwise cannot guarantee exactness, only could be used to update lower bound
                case Indicator::CEP_DENSITY_LB: // find a better lower bound
                case Indicator::QPBOI_LB: // find a better lower bound
                case Indicator::MIP_INDIRECT_WITH_INIT: // the same as UNDER_MIP_DIRECT_UB
                case Indicator::MIP_INDIRECT_NO_INIT: // the same as UNDER_MIP_DIRECT_UB
                case Indicator::MIP_INDIRECT_WITH_HEURISTIC: // the same as CEP_DENSITY_LB
                    // (1). The density of any subgraph must be <= lambda
                    // Can be evoked in QPBO_UB, UNDER_MIP_DIRECT_UB, and MIP_INDIRECT cases
                    if (result.nodes.empty() && result.exact) {
                        upper_bound = lambda;
                        break;
                    }

                    // (2). Have found a better lower bound
                    if (result.density > result_lb.density) {
                        result_lb = {result.nodes, result.density};
                        if (cfg.enable_graph_pruning) Pruning({}, result_lb.density);
                        if (result.exact) {
                            if (cfg.enable_mip_constrains_vertex_ub) {
                                vertex_upper_bound = result.nodes.size() - 1;
                            }
                            if (cfg.enable_mip_constrains_vertex_lb) {
                                upper_bound = min(upper_bound, lambda + result.nodes.size() * (result.density - lambda) / vertex_lower_bound);
                            }
                        }
                        break;
                    }

                    // (3). Can be evoked by MIP's failure due to time limit, or other reasons
                    LOG("DinkelbachBinary: MIP failure or hit time limit, and cannot improve lower bound.");
                    return result_lb;
                default: // should not happen
                    string msg = "DinkelbachBinary Bug: unexpected indicator from QPBO_CEP_MIP.";
                    info += msg + " | ";
                    throw runtime_error(msg);
            }
        }
        return result_lb;
    }

    SubgraphResult Dinkelbach(const EXACT_Config& cfg, SubgraphResult& result_lb) {
        for (auto iter = 0; iter < cfg.dinkelbach_iterations; iter++) {
            auto result = QPBO_CEP_MIP(cfg, result_lb, result_lb.density, true);
                if (result.density > result_lb.density) {
                    result_lb = {result.nodes, result.density};
                    if (result.exact && cfg.enable_mip_constrains_vertex_ub) {
                        vertex_upper_bound = result.nodes.size() - 1;
                    }
                } else {
                    break;
            }
        }
        return result_lb;
    }

    SubgraphResult Run(const EXACT_Config& cfg) {
        // Step 1. Result found by CEP as initial solution
        LOG("Start Running on " << cfg.input);
        auto [result_lb, optima] = FindLowerBound(cfg);
        if (optima) return result_lb;
        LOG("FindLowerBound: Lower bound set to: " << result_lb.density);
        if (cfg.enable_pruning_set) PruningModeToggle(cfg, 0, true);
        if (cfg.enable_graph_pruning) Pruning({}, result_lb.density);
        
        if (cfg.enable_binary_search) {
            // Step 2. Find an upper bound for QPBO
            auto upper_bound = FindUpperBound(cfg,result_lb);
            LOG("FindUpperBound:Upper bound set to: " << upper_bound);

            // Step 3. Refine the solution by Dinkelbach
            return DinkelbachBinary(cfg, result_lb, upper_bound);
        } else {
            return Dinkelbach(cfg, result_lb);
        }      
    }

    void Reset(const EXACT_Config& cfg) {
        if (cfg.enable_mip_constrains_vertex_ub) {
            vertex_upper_bound = num_vertices(G);
        }
        CEP::Reset(true);
    }
};