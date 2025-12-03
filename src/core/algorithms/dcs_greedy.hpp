#include "../graph.hpp"

class DCSGreedy : public PGraph {
public:
    struct Config {
        string input;
        string output;
        bool reverse_weight = false;
        unsigned num_iter = 1;
        
        void load_from_json(const string& filename) {
            std::ifstream file(filename);
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            auto json = boost::json::parse(content).as_object();
            
            if (json.contains("input")) input = json.at("input").as_string().c_str();
            if (json.contains("output")) output = json.at("output").as_string().c_str();
            if (json.contains("reverse_weight")) reverse_weight = json.at("reverse_weight").as_bool();
            if (json.contains("num_iter")) num_iter = json.at("num_iter").to_number<unsigned>();
        }
    };
    
    Config config;

    DCSGreedy(const string& config_file) {
        config.load_from_json(config_file);
        ReadGraph(config.input, config.reverse_weight);
    }

    SubgraphResult MaxEdge() {
        double max_weight = -numeric_limits<double>::infinity();
        vector<Vertex> max_edge;
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            auto current_weight = G[*ei].weight / 2;
            if (current_weight > max_weight) {
                max_weight = current_weight;
                max_edge = {source(*ei, G), target(*ei, G)};
            }
        }
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            if (loop_weight[*vi] > max_weight) {
                max_weight = loop_weight[*vi];
                max_edge = {*vi};
            }
        }
        return SubgraphResult{max_edge, max_weight};
    }

    SubgraphResult Peeling(bool positive_only = false) {
        fill(valid.begin(), valid.end(), true);
        MinHeap pq;
        vector<MinHeap::handle_type> handles(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            double degree = (positive_only && loop_weight[*vi] < 0) ? 0.0 : loop_weight[*vi];
            for (auto [ei, ee] = out_edges(*vi, G); ei != ee; ++ei) {
                double weight = G[*ei].weight;
                if (!positive_only || weight > 0) {
                    degree += weight;
                }
            }
            handles[*vi] = pq.push(MinHeapNode{degree, *vi});
        }
        auto current_weight_sum = total_weight;
        if (positive_only) {
            current_weight_sum = 0.0;
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                if (G[*ei].weight > 0) {
                    current_weight_sum += G[*ei].weight;
                }
            }
            for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
                if (loop_weight[*vi] > 0) {
                    current_weight_sum += loop_weight[*vi];
                }
            }
        }
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
            if (!positive_only || loop_weight[u] > 0) {
                current_weight_sum -= loop_weight[u];
            }
            current_vertex_count--;

            for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                auto v = target(*ei, G);
                double weight = G[*ei].weight;
                if ((!positive_only || weight > 0) && valid[v]) {
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

        if (positive_only) {
            // compute the real density
            vector<bool> selected(num_vertices(G), false);
            double total_weight_sum = 0.0;
            for (auto p = remove_order.begin() + best_step; p != remove_order.end(); ++p) {
                selected[*p] = true;
                total_weight_sum += loop_weight[*p];
            }
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                Vertex u = source(*ei, G);
                Vertex v = target(*ei, G);
                if (selected[u] && selected[v] && u <= v) {
                    total_weight_sum += G[*ei].weight;
                }
            }
            best_density = total_weight_sum / (remove_order.size() - best_step);
        }
        return {{remove_order.begin() + best_step, remove_order.end()}, best_density};
    }

    SubgraphResult MaxConnectedComponent(const vector<Vertex>& nodes) {
        std::unordered_set<Vertex> node_set(nodes.begin(), nodes.end());
        std::unordered_set<Vertex> visited;
        visited.reserve(nodes.size());
        SubgraphResult best {{}, -numeric_limits<double>::infinity()};

        for (auto start : nodes) {
            if (visited.count(start)) continue;
            
            // BFS to find component
            vector<Vertex> component = {start};
            visited.insert(start);
            
            for (auto i = 0; i < component.size(); i++) {
                Vertex u = component[i];
                for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                    Vertex v = target(*ei, G);
                    if (node_set.count(v) && !visited.count(v)) {
                        visited.insert(v);
                        component.push_back(v);
                    }
                }
            }
            
            double total_weight = 0.0;
            for (auto u : component) {
                for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                    Vertex v = target(*ei, G);
                    if (node_set.count(v) && u <= v) {
                        total_weight += G[*ei].weight;
                    }
                }
                total_weight += loop_weight[u];
            }
            double density = total_weight / component.size();
            if (density > best.density) {
                best = {component, density};
            }
        }        
        return best;
    }

    SubgraphResult Run() {
        auto S = MaxEdge();
        if (S.density <= 0) {
            return {{}, 0.0};
        }
        auto S1 = Peeling();
        auto S2 = Peeling(true);
        if (S1.density > S.density) S = S1;
        if (S2.density > S.density) S = S2;

        return MaxConnectedComponent(S.nodes);
    }
    
    void add_config_params(boost::json::object& config_obj) override {
        // DCSGreedy has no additional parameters beyond base config
    }
};