#include "../graph.hpp"

class CEP : public PGraph {
public:
    struct Config {
        string input;
        string output;
        bool reverse_weight = false;
        unsigned toggle_done = 2;
        unsigned toggle_left = 20;
        double max_neg = 200;
        unsigned max_local_optima = 10;
        bool do_peeling = false;
        unsigned num_iter = 1;
        
        void load_from_json(const string& filename) {
            std::ifstream file(filename);
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            auto json = boost::json::parse(content).as_object();
            
            if (json.contains("input")) input = json.at("input").as_string().c_str();
            if (json.contains("output")) output = json.at("output").as_string().c_str();
            if (json.contains("reverse_weight")) reverse_weight = json.at("reverse_weight").as_bool();
            if (json.contains("toggle_done")) toggle_done = json.at("toggle_done").to_number<unsigned>();
            if (json.contains("toggle_left")) toggle_left = json.at("toggle_left").to_number<unsigned>();
            if (json.contains("max_neg")) max_neg = json.at("max_neg").to_number<double>();
            if (json.contains("max_local_optima")) max_local_optima = json.at("max_local_optima").to_number<unsigned>();
            if (json.contains("do_peeling")) do_peeling = json.at("do_peeling").as_bool();
            if (json.contains("num_iter")) num_iter = json.at("num_iter").to_number<unsigned>();
        }
    };
    
    Config config;

protected: 
    struct MaxHeapNode {
        double key;
        Vertex node;
        bool operator<(const MaxHeapNode& other) const {
            return (key < other.key) || (key == other.key && node < other.node);
        }
    };
    using MaxHeap = heap::fibonacci_heap<MaxHeapNode>;
    enum class Status {
        Out,
        Fringe,
        In
    };
    vector<Status> status;
    vector<size_t> neighbor_in_count;
    vector<double> pos_weight;
    bool pruning_set_on = false;
    set<MaxHeapNode> pruning_set;
    vector<set<MaxHeapNode>::iterator> pruning_handles;
    size_t valid_count = 0;

public:
    CEP(const string& config_file) {
        config.load_from_json(config_file);
        ReadGraph(config.input, config.reverse_weight);
        status = vector<Status>(num_vertices(G), Status::Out);
        neighbor_in_count = vector<size_t>(num_vertices(G), 0);
        pos_weight = vector<double>(num_vertices(G), 0.0);
        valid_count = num_vertices(G);
    }

    unsigned ConvertMaxNeg(double max_neg) {
        return static_cast<unsigned>(max_neg * (max_neg < 1.0 ? num_vertices(G) : 1.0));
    }

    double ComputeDensity(const vector<Vertex>& nodes, bool iterate_edges = false) {
        if (nodes.empty()) return 0.0;
        double total_weight_sum = 0.0;
        vector<bool> selected(num_vertices(G), false);
        if (iterate_edges) {
            for (auto node : nodes) {
                selected[node] = true;
                total_weight_sum += loop_weight[node];
            }
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                if (selected[source(*ei, G)] && selected[target(*ei, G)]) {
                    total_weight_sum += G[*ei].weight;
                }
            }
        } else {
            for (auto node : nodes) {
                selected[node] = true;
                total_weight_sum += loop_weight[node];
                for (auto [ei, ee] = out_edges(node, G); ei != ee; ++ei) {
                    if (selected[target(*ei, G)]) {
                        total_weight_sum += G[*ei].weight;
                    }
                }
            }
        }
        return total_weight_sum / nodes.size();
    }

    SubgraphResult Peeling() {
        MinHeap pq;
        vector<MinHeap::handle_type> handles(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            double degree = loop_weight[*vi];
            for (auto [ei, ee] = out_edges(*vi, G); ei != ee; ++ei) {
                degree += G[*ei].weight;
            }
            auto h = pq.push(MinHeapNode{degree, *vi});
            handles[*vi] = h;
        }
        auto current_weight_sum = total_weight;
        auto current_vertex_count = num_vertices(G);
        auto current_density = current_vertex_count > 0 ? current_weight_sum / current_vertex_count : 0.0;
        vector<Vertex> remove_order;
        remove_order.reserve(num_vertices(G)); 
        size_t best_step = 0;
        double best_density = current_density;

        while (!pq.empty()) {
            auto u = pq.top().node;
            pq.pop();
            valid[u] = false;
            remove_order.push_back(u);
            current_vertex_count--;
            current_weight_sum -= loop_weight[u];

            for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                auto v = target(*ei, G);
                double weight = G[*ei].weight;
                if (valid[v]) {
                    current_weight_sum -= weight;
                    auto new_key = (*handles[v]).key - weight;
                    pq.update(handles[v], MinHeapNode{new_key, v});
                }
            }

            current_density = current_vertex_count > 0 ? current_weight_sum / current_vertex_count : 0.0;
            if (current_density > best_density) {
                best_density = current_density;
                best_step = remove_order.size();
            }
        }
        fill(valid.begin(), valid.end(), true);
        return {{remove_order.begin() + best_step, remove_order.end()}, best_density};
    }

    vector<double> InitializePositiveWeights() {
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            double weight = G[*ei].weight;
            if (weight > 0) {
                pos_weight[source(*ei, G)] += weight;
                pos_weight[target(*ei, G)] += weight;
            }
        }
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            pos_weight[*vi] += loop_weight[*vi];
        }
        return pos_weight;
    }

    void PruningModeToggle(unsigned it, unsigned max_local_optima, bool force_on = false) {
        if ((!pruning_set_on && it >= config.toggle_done && max_local_optima - it >= config.toggle_left) || force_on) {
            pruning_set_on = true;
            pruning_handles.resize(num_vertices(G));
            for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
                if (valid[*vi]) {
                    pruning_handles[*vi] = pruning_set.insert({pos_weight[*vi], *vi}).first;
                }
            }
            valid_count = pruning_set.size();
        }
    }

    Vertex FindAnchor() {
        if (pruning_set_on) {
            return pruning_set.empty() ? Traits::null_vertex() : pruning_set.rbegin()->node;
        } else {
            return std::distance(pos_weight.begin(), max_element(pos_weight.begin(), pos_weight.end()));
        }
    }

    SubgraphResult LocalGreedy(Vertex anchor, unsigned max_neg) {
        // ============== Clear and initialization ==============
        fill(status.begin(), status.end(), Status::Out);
        fill(neighbor_in_count.begin(), neighbor_in_count.end(), 0);
        double current_weight_sum = 0.0;
        double max_f = -numeric_limits<double>::infinity();
        MaxHeap selected, fringe, best;
        vector<MaxHeap::handle_type> handles(num_vertices(G));
        status[anchor] = Status::Fringe;
        handles[anchor] = fringe.push({loop_weight[anchor], anchor});
        unsigned neg_count = 0;
        Vertex next = anchor;

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
            double f_value = selected_count > 0 ? current_weight_sum / selected_count : 0.0;
            max_f = max(max_f, f_value);

            // ============== Compute marginal gains for top of each heap ==============
            double best_mg = -numeric_limits<double>::infinity();
            Vertex best_node = Traits::null_vertex();
            bool best_is_addition = false;
            if (!selected.empty()) {
                auto top_item = selected.top();
                double new_sum = (selected_count > 1) ? (current_weight_sum + top_item.key) / (selected_count - 1) : 0.0;
                best_mg = new_sum - f_value;
                best_node = top_item.node;
            }
            if (!fringe.empty()) {
                auto top_item = fringe.top();
                double new_sum = (current_weight_sum + top_item.key) / (selected_count + 1);
                double mg = new_sum - f_value;
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
                double current_density = current_weight_sum / selected.size();
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

    void Pruning(const vector<Vertex>& nodes, double threshold_density) {
        pruning_set_on ? PruningSet(nodes, threshold_density) : PruningVector(nodes, threshold_density);
    }
    
    void PruningSet(const vector<Vertex>& nodes, double threshold_density) {
        for (auto node : nodes) {
            pruning_set.erase(pruning_handles[node]);
            valid[node] = false;
            for (auto [ei, ee] = out_edges(node, G); ei != ee; ++ei) {
                auto v = target(*ei, G);
                if (valid[v] && G[*ei].weight > 0) {
                    auto v_it = pruning_handles[v];
                    auto new_key = v_it->key - G[*ei].weight;
                    pruning_set.erase(v_it);
                    pruning_handles[v] = pruning_set.insert({new_key, v}).first;
                }
            }
        }
        while (!pruning_set.empty() && pruning_set.begin()->key < threshold_density) {
            auto it = pruning_set.begin();
            auto u = it->node;
            pruning_set.erase(it);
            valid[u] = false;
            for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                auto v = target(*ei, G);
                if (valid[v] && G[*ei].weight > 0) {
                    auto v_it = pruning_handles[v];
                    auto new_key = v_it->key - G[*ei].weight;
                    pruning_set.erase(v_it);
                    pruning_handles[v] = pruning_set.insert({new_key, v}).first;
                }
            }
        }
        valid_count = pruning_set.size();
    }

    void PruningVector(const vector<Vertex>& nodes, double threshold_density, bool maintain_valid_count = true, bool skip_threshold_pruning = false) {
        for (auto node : nodes) {
            valid[node] = false;
            if (maintain_valid_count) {
                --valid_count;
            }
            pos_weight[node] = -numeric_limits<double>::infinity();
            for (auto [ei, ee] = out_edges(node, G); ei != ee; ++ei) {
                auto v = target(*ei, G);
                if (valid[v]) {
                    double weight = G[*ei].weight;
                    if (weight > 0) {
                        pos_weight[v] -= weight;
                    }
                }
            }
        }
        if (skip_threshold_pruning) {
            return;
        }
        for (auto i = 0; i != pos_weight.size(); ++i) {
            if (valid[i] && pos_weight[i] < threshold_density) {
                valid[i] = false;
                if (maintain_valid_count) {
                    --valid_count;
                }
                pos_weight[i] = -numeric_limits<double>::infinity();
                for (auto [ei, ee] = out_edges(i, G); ei != ee; ++ei) {
                    auto v = target(*ei, G);
                    if (valid[v]) {
                        double weight = G[*ei].weight;
                        if (weight > 0) {
                            pos_weight[v] -= weight;
                        }
                    }
                }
            }
        }
    }

    SubgraphResult Run() {
        // Step 1. Contraction by Peeling
        SubgraphResult best = config.do_peeling ? Peeling() : SubgraphResult{{}, 0.0};

        // Step 2. Expansion by Multi-Local Search
        InitializePositiveWeights();
        auto converted_max_neg = ConvertMaxNeg(config.max_neg);
        for (unsigned it = 0; it < config.max_local_optima; ++it) {
            PruningModeToggle(it, config.max_local_optima);
            auto anchor = FindAnchor();
            if (anchor == Traits::null_vertex() || !valid[anchor]) break;
            auto result = LocalGreedy(anchor, converted_max_neg);
            if (result.density > best.density) {
                best = result;
            }
            Pruning(result.nodes, best.density);
        }
        return best;
    }

    void Reset(bool init_pos_weight = true) {
        fill(valid.begin(), valid.end(), true);
        valid_count = num_vertices(G);
        fill(status.begin(), status.end(), Status::Out);
        fill(neighbor_in_count.begin(), neighbor_in_count.end(), 0);
        fill(pos_weight.begin(), pos_weight.end(), 0.0);
        pruning_set_on = false;
        pruning_set.clear();
        pruning_handles.clear();
        if (init_pos_weight) {
            InitializePositiveWeights();
        }
    }
    
    void add_config_params(boost::json::object& config_obj) override {
        config_obj["toggle_done"] = config.toggle_done;
        config_obj["toggle_left"] = config.toggle_left;
        config_obj["max_neg"] = config.max_neg;
        config_obj["max_local_optima"] = config.max_local_optima;
        config_obj["do_peeling"] = config.do_peeling;
    }
};
