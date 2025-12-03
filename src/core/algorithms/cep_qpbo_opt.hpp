#include "cep.hpp"
#include "../../external/QPBO/QPBO.h"
#include <gurobi_c++.h>

class CEP_QPBO_OPT : public CEP {
public:
    struct Config : public CEP::Config {
        double step_size = 1.02;
        unsigned ub_mip_bound = 100;
        unsigned dinkelbach_iterations = 30;
        double epsilon = -0.00001;
        double mip_time_limit = 300.0;
        bool use_binary = true;
        bool use_probe = false;
        bool skip_mip_init = true;
        bool heuristic_after_mip = true;
        
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
            if (json.contains("use_probe")) use_probe = json.at("use_probe").as_bool();
            if (json.contains("skip_mip_init")) skip_mip_init = json.at("skip_mip_init").as_bool();
            if (json.contains("heuristic_after_mip")) heuristic_after_mip = json.at("heuristic_after_mip").as_bool();
        }
    };

private:
    enum class Indicator {
        QPBO_UB,
        QPBO_LB,
        MIP_DIRECT,
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
    };
    using REAL = double;
    size_t vertex_lower_bound, vertex_upper_bound;
    const bool set_vertex_lb = true;

public:
    CEP_QPBO_OPT(const string& config_file) : CEP(config_file) {
        static_cast<Config&>(config).load_from_json(config_file);
        InitializePositiveWeights();
        vertex_lower_bound = 3; // initialized, but not used in RunMIP (explained there).
            vertex_upper_bound = num_vertices(G);
        }

    QPBOResult RunQPBO(double lambda, bool probe = false, bool improve = false, vector<Vertex> init_label = {}) {
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
        
        if (probe) {
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

    pair<SubgraphResult, bool> FindLowerBound(bool use_probe = false) {
        // 1. Run CEP to get initial solution
        auto result = CEP::Run();
        Reset();
        // 2. Need to find better lower bound among single nodes and edges if set_vertex_lb
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
        // 3. Run QPBO to check whether it is already optimal, or update lower bound if possible
        Pruning({}, result.density);
        auto pre_qpbo = QPBO_CEP_MIP(result, result.density, use_probe, 0, 0, false, 0, true);
        PruningModeToggle(0, 0, true);
        Pruning({}, result.density);
        switch (pre_qpbo.info) {
            case Indicator::QPBO_UB:
                // (1). Optimal found
                return {result, true};
            case Indicator::NO_MIP:
                // (2). Have undecided nodes in QPBO
                return {result, false};
            case Indicator::QPBO_LB:
                // (3). New lower bound found
                return {{pre_qpbo.nodes, pre_qpbo.density}, false};
            default:
                throw runtime_error("Should only use QPBO in the FindLowerBound function.");
        }
    }

    double FindUpperBound(SubgraphResult& result_lb, double step_size, bool use_probe, unsigned direct_mip_bound, unsigned mip_time_limit) {
        assert(step_size > 1.0); // step_size should be larger than 1.0
        // 1. The naive way for upper bound is the maximum among positive weight sum of edges incident to each vertex
        auto upper_bound = pruning_set.rbegin()->key;

        // 2. try to find a tighter upper bound by increasing from lower bound step by step
        auto lambda = result_lb.density * step_size;
        while (lambda < upper_bound) {
            auto result = QPBO_CEP_MIP(result_lb, lambda, use_probe, direct_mip_bound, mip_time_limit, false, 0, true);
            switch (result.info) {
                case Indicator::QPBO_UB:
                case Indicator::QPBO_LB:
                case Indicator::MIP_DIRECT:
                    // (1). The density of any subgraph must be <= lambda
                    if (result.nodes.empty() && result.exact) {
                        return lambda;
                    }

                    // (2). Have found a better lower bound
                    if (result.density > result_lb.density) {
                        result_lb = {result.nodes, result.density};
                        Pruning({}, result.density);
                        if (result.exact) {
                            vertex_upper_bound = result.nodes.size() - 1;
                            if (set_vertex_lb) {
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
                    throw runtime_error("Should only use QPBO and MIP when undecided nodes are small in the FindUpperBound function.");
            }

            lambda *= step_size; // try next lambda
        }

        // 3. No tighter upper bound found, return the naive one
        return upper_bound;
    }

    pair<vector<Vertex>, bool> RunMIP(QPBOResult& qpbo_result, double lambda, double mip_time_limit, vector<size_t> initial_solution = {}) {
        cerr << "Running MIP on " << qpbo_result.undecided.size() << " undecided nodes." << endl;
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
                if (qpbo_result.labels[i] == -1) {
                    obj += (lambda - loop_weight[i]) * undecided_vars[i];
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
            model.addConstr(vertex_sum <= (vertex_upper_bound - qpbo_result.fixed_in.size()));
            // Warn: Don't set it, because it will return non-empty result even if the lambda >= largest density
            // The non-empty result cannot reflect correct upper bound for Dinkelbach, will make the Dinkelbach break.
            // if (set_vertex_lb) {
            //     model.addConstr(vertex_sum >= max(static_cast<size_t>(0), vertex_lower_bound - qpbo_result.fixed_in.size()));
            // }


            // ============== Set initial solution if provided ==============
            for (auto i : qpbo_result.undecided) {
                undecided_vars[i].set(GRB_DoubleAttr_Start, (!initial_solution.empty() && initial_solution[i] == 1) ? 1 : 0);
            }
            
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

    SubgraphResult CEPDensity(const QPBOResult& qpbo_result, unsigned max_local_optima, const vector<Vertex>& mip_result = {}, double density = 0.0) {
        auto best = SubgraphResult{{}, 0.0};
        auto valid_original = valid;
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
            auto result = LocalGreedy(anchor, max_neg);
            if (result.density > best.density) {
                best = result;
            }
            PruningVector(result.nodes, best.density, false);
        }
        valid = valid_original;
        return best;
    }

    SubgraphResult CEPLambda(const QPBOResult& qpbo_result, unsigned max_local_optima, double lambda) {
        auto best = SubgraphResult{{}, 0.0};
        auto valid_original = valid;
        for (auto i : qpbo_result.fixed_out) {
            valid[i] = false;
        }
        if (!qpbo_result.fixed_in.empty()) {
            best = CEPLambdaLocal(qpbo_result, 0, qpbo_result.fixed_in.size() + qpbo_result.undecided.size(), lambda);
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
            for (unsigned it = 0; it < max_local_optima; ++it) {
                auto anchor = std::distance(pos_weight.begin(), max_element(pos_weight.begin(), pos_weight.end()));
                if (anchor == Traits::null_vertex() || !valid[anchor]) break;
                auto result = CEPLambdaLocal(qpbo_result, anchor, qpbo_result.fixed_in.size() + qpbo_result.undecided.size(), lambda);
                if (result.density > best.density) {
                    best = result;
                }
                PruningVector(result.nodes, 0, false, true);
            }
        }
        valid = valid_original;
        return best;
    }
    
    SubgraphResult CEPLambdaLocal(const QPBOResult& qpbo_result, Vertex anchor, unsigned max_neg, double lambda) {
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
    SubgraphResultEnhanced QPBO_CEP_MIP(const SubgraphResult& lb, double lambda, bool use_probe, int direct_mip_bound, unsigned mip_time_limit, bool handle_large_undecided, unsigned max_local_optima, bool skip_mip_init = true, bool heuristic_after_mip = true, double* upper_bound = nullptr) {
        auto qpbo_result = RunQPBO(lambda, use_probe, false);
        if (qpbo_result.undecided.empty()) { // QPBO fixed all nodes
            if (qpbo_result.fixed_in.empty()) { // all nodes are labeled as 0
                // 1. if QPBO labels all nodes as 0, lambda is a new upper bound
                return {SubgraphResult{}, true, Indicator::QPBO_UB};
            } else { // some are labeled as 1
                // 2. if QPBO labels some nodes as 1, compute new density as lower bound
                auto density = ComputeDensity(qpbo_result.fixed_in); // will be >= lambda
                return {qpbo_result.fixed_in, density, true, Indicator::QPBO_LB};
            }
        } else { // QPBO has undecided nodes
            if (qpbo_result.undecided.size() < direct_mip_bound) { // undecided nodes are small
                // 3. if undecided nodes are small enough, run MIP directly
                auto mip_result = RunMIP(qpbo_result, lambda, mip_time_limit);
                auto density = ComputeDensity(mip_result.first);
                return {mip_result.first, density, mip_result.second, Indicator::MIP_DIRECT};
            } else if (handle_large_undecided) { // there are many undecided nodes and want to handle them
                auto cep_density_result = CEPDensity(qpbo_result, max_local_optima); // try to use CEPDensity to get better lower bound
                if (cep_density_result.density > lb.density) { // CEPDensity gets new lower bound
                    // 4. if CEPDensity gets new lower bound, return to update lower bound
                    return {cep_density_result, false, Indicator::CEP_DENSITY_LB};
                } else if (!skip_mip_init) { // CEPDensity cannot get better lower bound, run CEPLambda and QPBOI on undecided nodes
                    auto cep_lambda_result = CEPLambda(qpbo_result, max_local_optima, lambda); // get some initial solution from CEPLambda based on lambda
                    auto qpboi_result = RunQPBO(lambda, false, true, cep_lambda_result.nodes); // run QPBOI to improve the initial solution
                    auto density = ComputeDensity(qpboi_result.fixed_in);
                    if (density > lb.density) { // QPBOI gets new lower bound
                        // 5. if QPBOI gets new lower bound, return to update lower bound
                        return {qpboi_result.fixed_in, density, false, Indicator::QPBOI_LB};
                    } else { // QPBOI cannot get better lower bound
                        // 6. if QPBOI cannot get better lower bound, run MIP on undecided nodes
                        auto mip_result = RunMIP(qpbo_result, lambda, mip_time_limit, qpboi_result.fixed_in); // run MIP with initial solution from QPBOI
                        auto density = ComputeDensity(mip_result.first);
                        return {mip_result.first, density, mip_result.second, Indicator::MIP_INDIRECT_WITH_INIT};
                    }
                } else { // skip CEPLambda and QPBOI, run MIP directly without initialization
                    // 7. run MIP directly on undecided nodes
                    auto mip_result = RunMIP(qpbo_result, lambda, mip_time_limit); // run MIP
                    auto density = ComputeDensity(mip_result.first);
                    if (!heuristic_after_mip || mip_result.first.empty()) { // no heuristic after MIP or MIP returns empty result
                        return {mip_result.first, density, mip_result.second, Indicator::MIP_INDIRECT_NO_INIT};
                    } else { // run heuristic after MIP to try to further improve the solution
                        if (mip_result.second) { // update the constrains for MIP if MIP is optimal
                            vertex_upper_bound = mip_result.first.size() - 1;
                            if (set_vertex_lb) {
                                *upper_bound = min(*upper_bound, lambda + mip_result.first.size() * (density - lambda) / vertex_lower_bound);
                            }
                        }

                        // 8. run CEPDensity after MIP to try to further improve the solution
                        auto cep_after_result = CEPDensity(qpbo_result, 1, mip_result.first, density); // run CEPDensity
                        if (cep_after_result.density >= density) { // CEPDensity after MIP gets better or equal lower bound
                            cerr << "cep_after_mip gets new lower bound: " << cep_after_result.density << " rather than " << density << endl;
                            return {cep_after_result, false, Indicator::MIP_INDIRECT_WITH_HEURISTIC};
                        } else { // should not happen
                            throw runtime_error("Bugs in QPBO_CEP_MIP with heuristic after MIP.");
                        }
                    }
                }
            } else {
                return {SubgraphResult{}, false, Indicator::NO_MIP};
            }
        }
    }

    bool Terminate(double lower_bound, double upper_bound, double epsilon) {
        return (epsilon > 0 ? (lower_bound / upper_bound) : (lower_bound - upper_bound)) >= epsilon;
    }

    SubgraphResult DinkelbachBinary(SubgraphResult& result_lb, double upper_bound, unsigned iterations, double epsilon, double mip_time_limit, bool use_probe, unsigned direct_mip_bound, unsigned max_local_optima, bool skip_mip_init, bool heuristic_after_mip) {
        for (auto iter = 0; iter < iterations; iter++) {
            cerr << "lower_bound = " << result_lb.density << ", upper_bound = " << upper_bound << endl;
            if (Terminate(result_lb.density, upper_bound, epsilon)) break;
            auto lambda = (result_lb.density + upper_bound) / 2.0;
            auto result = QPBO_CEP_MIP(result_lb, lambda, use_probe, direct_mip_bound, mip_time_limit, true, max_local_optima, skip_mip_init, heuristic_after_mip, &upper_bound);
            switch (result.info) {
                case Indicator::QPBO_UB: // find a tighter upper bound
                case Indicator::QPBO_LB: // find a tighter lower bound
                case Indicator::MIP_DIRECT: // if success, can know whether lambda is an upper bound or get a better lower bound; otherwise cannot guarantee exactness, only could be used to update lower bound
                case Indicator::CEP_DENSITY_LB: // find a better lower bound
                case Indicator::QPBOI_LB: // find a better lower bound
                case Indicator::MIP_INDIRECT_WITH_INIT: // the same as MIP_DIRECT
                case Indicator::MIP_INDIRECT_NO_INIT: // the same as MIP_DIRECT
                case Indicator::MIP_INDIRECT_WITH_HEURISTIC: // the same as CEP_DENSITY_LB
                    // (1). The density of any subgraph must be <= lambda
                    // Can be evoked in QPBO_UB, MIP_DIRECT, and MIP_INDIRECT cases
                    if (result.nodes.empty() && result.exact) {
                        upper_bound = lambda;
                        break;
                    }

                    // (2). Have found a better lower bound
                    if (result.density > result_lb.density) {
                        result_lb = {result.nodes, result.density};
                        Pruning({}, result_lb.density);
                        if (result.exact) {
                            vertex_upper_bound = result.nodes.size() - 1;
                            if (set_vertex_lb) {
                                upper_bound = min(upper_bound, lambda + result.nodes.size() * (result.density - lambda) / vertex_lower_bound);
                            }
                        }
                        break;
                    }

                    // (3). Can be evoked by MIP's failure due to time limit, or other reasons
                    cerr << "DinkelbachBinary: Might be caused by MIP failure or time limit." << endl;
                    return result_lb;
                default: // should not happen
                    throw runtime_error("Bugs in DinkelbachBinary.");
            }
        }
        return result_lb;
    }

    SubgraphResult Dinkelbach(SubgraphResult& result_lb, unsigned iterations, double mip_time_limit, bool use_probe, unsigned direct_mip_bound, unsigned max_local_optima, bool skip_mip_init, bool heuristic_after_mip) {
        for (auto iter = 0; iter < iterations; iter++) {
            auto result = QPBO_CEP_MIP(result_lb, result_lb.density, use_probe, direct_mip_bound, mip_time_limit, true, max_local_optima, skip_mip_init, heuristic_after_mip);
                if (result.density > result_lb.density) {
                    result_lb = {result.nodes, result.density};
                    if (result.exact) {
                        vertex_upper_bound = result.nodes.size() - 1;
                    }
                } else {
                    break;
            }
        }
        return result_lb;
    }

    SubgraphResult Run() {
        auto& opt_config = static_cast<Config&>(config);
        // Step 1. Result found by CEP as initial solution
        auto [result_lb, optima] = FindLowerBound(opt_config.use_probe);
        if (optima) return result_lb;
        
        if (opt_config.use_binary) {
            // Step 2. Find an upper bound for QPBO
            auto upper_bound = FindUpperBound(result_lb, opt_config.step_size, opt_config.use_probe, opt_config.ub_mip_bound, opt_config.mip_time_limit);
            // Step 3. Refine the solution by Dinkelbach
            return DinkelbachBinary(result_lb, upper_bound, opt_config.dinkelbach_iterations, opt_config.epsilon, opt_config.mip_time_limit, opt_config.use_probe, opt_config.ub_mip_bound, config.max_local_optima, opt_config.skip_mip_init, opt_config.heuristic_after_mip);
        } else {
            return Dinkelbach(result_lb, opt_config.dinkelbach_iterations, opt_config.mip_time_limit, opt_config.use_probe, opt_config.ub_mip_bound, config.max_local_optima, opt_config.skip_mip_init, opt_config.heuristic_after_mip);
        }      
    }
    
    void add_config_params(boost::json::object& config_obj) override {
        // First add CEP params
        CEP::add_config_params(config_obj);
        // Then add CEP_QPBO_OPT specific params
        auto& opt_config = static_cast<Config&>(config);
        config_obj["step_size"] = opt_config.step_size;
        config_obj["ub_mip_bound"] = opt_config.ub_mip_bound;
        config_obj["dinkelbach_iterations"] = opt_config.dinkelbach_iterations;
        config_obj["epsilon"] = opt_config.epsilon;
        config_obj["mip_time_limit"] = opt_config.mip_time_limit;
        config_obj["use_binary"] = opt_config.use_binary;
        config_obj["use_probe"] = opt_config.use_probe;
        config_obj["skip_mip_init"] = opt_config.skip_mip_init;
        config_obj["heuristic_after_mip"] = opt_config.heuristic_after_mip;
    }
};