#include "../graph.hpp"

class NEG_DSD : public PGraph {
public:
    struct Config {
        vector<double> C_values = {1.0};
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
            if (json.contains("C_values")) {
                C_values.clear();
                for (auto& v : json.at("C_values").as_array()) {
                    C_values.push_back(v.as_double());
                }
            }
        }
    };
    
    Config config;

private:
    vector<double> pos_deg;
    vector<double> neg_deg;

public:
    NEG_DSD(const string& config_file) {
        config.load_from_json(config_file);
        ReadGraph(config.input, config.reverse_weight);
    }

    void InitializeDegrees() {
        pos_deg.assign(num_vertices(G), 0.0);
        neg_deg.assign(num_vertices(G), 0.0);
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            auto u = source(*ei, G);
            auto v = target(*ei, G);
            double weight = G[*ei].weight;
            if (weight > 0) {
                pos_deg[u] += weight;
                pos_deg[v] += weight;
            } else {
                neg_deg[u] += -weight;
                neg_deg[v] += -weight;
            }
        }
    }

    SubgraphResult Peeling(double C = 1.0) {
        fill(valid.begin(), valid.end(), true);
        MinHeap pq;
        vector<MinHeap::handle_type> handles(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            auto key = C * pos_deg[*vi] - neg_deg[*vi];
            key += loop_weight[*vi] > 0 ? C * loop_weight[*vi] : loop_weight[*vi];
            handles[*vi] = pq.push(MinHeapNode{key, *vi});
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
            current_weight_sum -= loop_weight[u];
            current_vertex_count--;

            for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                auto v = target(*ei, G);
                if (valid[v]) {
                    double weight = G[*ei].weight;
                    current_weight_sum -= weight;
                    auto new_key = (*handles[v]).key - (weight > 0 ? C * weight : weight);
                    pq.update(handles[v], MinHeapNode{new_key, v});
                }
            }

            current_density = current_vertex_count > 0 ? current_weight_sum / current_vertex_count : 0.0;
            if (current_density > best_density) {
                best_density = current_density;
                best_step = remove_order.size();
            }
        }
        return {{remove_order.begin() + best_step, remove_order.end()}, best_density};
    }

    SubgraphResult Run() {
        InitializeDegrees();
        SubgraphResult best {{}, -numeric_limits<double>::infinity()};
        for (auto C : config.C_values) {
            auto result = Peeling(C);
            if (result.density > best.density) {
                best = result;
            }
        }
        return best;
    }
    
    void add_config_params(boost::json::object& config_obj) override {
        boost::json::array c_values_array;
        for (const auto& c : config.C_values) {
            c_values_array.push_back(c);
        }
        config_obj["C_values"] = c_values_array;
    }
};