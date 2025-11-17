#pragma once

#include <gurobi_c++.h>
#include "QPBO/QPBO.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <set>

using namespace std;
using namespace boost;

class PGraph {
protected:
    struct EdgeProperty {
        double weight = 0.0;
    };

    using Graph = adjacency_list<vecS, vecS, undirectedS, no_property, EdgeProperty>;
    using Vertex = graph_traits<Graph>::vertex_descriptor;
    using Edge = graph_traits<Graph>::edge_descriptor;
    using Traits = graph_traits<Graph>;

    struct MinHeapNode {
        double key;
        Vertex node;
        bool operator<(const MinHeapNode& other) const {
            return (key > other.key) || (key == other.key && node > other.node);
        }
    };
    using MinHeap = heap::fibonacci_heap<MinHeapNode>;

    Graph G;
    vector<bool> valid;
    double total_weight = 0.0;
    vector<double> loop_weight;

    PGraph() = default;

public:
    PGraph(const string& input, bool reverse_weight) {
        ReadGraph(input, reverse_weight);
    }

    void ReadGraph(const string& input, bool reverse_weight) {
        ifstream infile(input);
        size_t n = 0, m = 0;
        string line;
        getline(infile, line);
        istringstream iss_first(line);
        iss_first >> n >> m;
        for (auto i = 0; i < n; ++i) {
            add_vertex(G);
        }
        valid = vector<bool>(n, true);
        loop_weight = vector<double>(n, 0.0);
        
        while (getline(infile, line)) {
            istringstream iss(line);
            Vertex u, v;
            double weight;
            iss >> u >> v >> weight;
            weight *= (reverse_weight ? -1.0 : 1.0);
            if (u == v) {
                loop_weight[u] += weight;
            } else {
                add_edge(u, v, EdgeProperty{weight}, G);
            }
            total_weight += weight;
        }
    }

    virtual ~PGraph() = default;

    struct SubgraphResult {
        vector<Vertex> nodes;
        double density;
    };

    void output(const string& filepath, double avg_time, SubgraphResult& result, int argc, char* argv[]) {
        ofstream out(filepath);
        if (!out) throw std::runtime_error("Cannot open " + filepath);
        std::sort(result.nodes.begin(), result.nodes.end());
        out << fixed << std::setprecision(6);
        out << "{\n"
            << "  \"time\": " << avg_time << ",\n"
            << "  \"density\": " << result.density << ",\n"
            << "  \"size\": " << result.nodes.size() << ",\n"
            << "  \"nodes\": [";
        for (auto i = 0; i < result.nodes.size(); ++i) {
            if (i) out << ", ";
            out << result.nodes[i];
        }
        out << "],\n" << "  \"command\": \"";
        for (auto i = 0; i < argc; ++i) {
            if (i) out << " ";
            out << argv[i];
        }
        out << "\"\n" << "}\n";
    }
};

class NEG_DSD : public PGraph {
private:
    vector<double> pos_deg;
    vector<double> neg_deg;

public:
    NEG_DSD(const string& input, bool reverse_weight)
        : PGraph(input, reverse_weight) {}

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

    SubgraphResult Run(const vector<double>& C_values) {
        InitializeDegrees();
        SubgraphResult best {{}, -numeric_limits<double>::infinity()};
        for (auto C : C_values) {
            auto result = Peeling(C);
            if (result.density > best.density) {
                best = result;
            }
        }
        return best;
    }
};

class DCSGreedy : public PGraph {
public:
    DCSGreedy(const string& input, bool reverse_weight)
        : PGraph(input, reverse_weight) {}

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
};

class CEP : public PGraph {
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

    const unsigned toggle_done = 2;
    const unsigned toggle_left = 20;

public:
    CEP(const string& input, bool reverse_weight) {
        ReadGraph(input, reverse_weight);
        status = vector<Status>(num_vertices(G), Status::Out);
        neighbor_in_count = vector<size_t>(num_vertices(G), 0);
        pos_weight = vector<double>(num_vertices(G), 0.0);
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

    void PruningModeToggle (unsigned it, unsigned max_local_optima) {
        if (!pruning_set_on) {
            if (it >= toggle_done && max_local_optima - it >= toggle_left) {
                pruning_set_on = true;
                pruning_handles.resize(num_vertices(G));
                for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
                    if (valid[*vi]) {
                        pruning_handles[*vi] = pruning_set.insert({pos_weight[*vi], *vi}).first;
                    }
                }
            }
        }
    }

    Vertex FindAnchor() {
        if (pruning_set_on) {
            return pruning_set.empty() ? Traits::null_vertex() : pruning_set.begin()->node;
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
    }

    void PruningVector(const vector<Vertex>& nodes, double threshold_density) {
        for (auto node : nodes) {
            valid[node] = false;
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
        for (auto i = 0; i != pos_weight.size(); ++i) {
            if (valid[i] && pos_weight[i] < threshold_density) {
                valid[i] = false;
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

    SubgraphResult Run(unsigned max_neg, unsigned max_local_optima, bool do_peeling) {
        // Step 1. Contraction by Peeling
        SubgraphResult best = do_peeling ? Peeling() : SubgraphResult{{}, 0.0};

        // Step 2. Expansion by Multi-Local Search
        InitializePositiveWeights();
        for (unsigned it = 0; it < max_local_optima; ++it) {
            PruningModeToggle(it, max_local_optima);
            auto anchor = FindAnchor();
            if (anchor == Traits::null_vertex() || !valid[anchor]) break;
            auto result = LocalGreedy(anchor, max_neg);
            if (result.density > best.density) {
                best = result;
            }
            pruning_set_on ? PruningSet(result.nodes, best.density) : PruningVector(result.nodes, best.density);
        }
        return best;
    }
};

class CEP_QPBO : public CEP {
private:
    struct QPBOResult {
        vector<int> labels;
        vector<Vertex> fixed_in;
        vector<Vertex> fixed_out;
        vector<Vertex> undecided;
    };
    using REAL = double;

public:
    CEP_QPBO(const string& input, bool reverse_weight)
        : CEP(input, reverse_weight) {
            InitializePositiveWeights();
        }

    QPBOResult RunQPBO(double lambda, bool improve = false) {
        size_t n = num_vertices(G);
        unique_ptr<QPBO<REAL>> qpbo(new QPBO<REAL>(n, 2 * num_edges(G)));
        qpbo->AddNode(n);
        
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            qpbo->AddUnaryTerm(*vi, 0.0, lambda-loop_weight[*vi]);
        }
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            qpbo->AddPairwiseTerm(source(*ei, G), target(*ei, G), 0, 0, 0, -G[*ei].weight);
        }
        
        qpbo->Solve();
        qpbo->ComputeWeakPersistencies();
        
        QPBOResult result;
        result.labels.resize(n);
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

        return result;
    }

    double FindUpperBound(double init_density, double step_size) {
        double lambda_ub = init_density * step_size;
        while (true) {
            auto result = RunQPBO(lambda_ub, false);
            if (result.undecided.empty() && result.fixed_in.empty()) {
                break;
            }
            lambda_ub *= step_size;
        }
        return min(lambda_ub, *max_element(pos_weight.begin(), pos_weight.end()));
    }

    vector<Vertex> RunMIP(QPBOResult& qpbo_result, double lambda, double mip_time_limit) {
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
            return selected;

        } catch (GRBException& e) {
            throw runtime_error("Gurobi exception: " + string(e.getMessage()));
            return qpbo_result.fixed_in;
        }
    }

    double ComputeDensity(const vector<Vertex>& nodes) {
        double total_weight_sum = 0.0;
        vector<bool> selected(num_vertices(G), false);
        for (auto node : nodes) {
            selected[node] = true;
            total_weight_sum += loop_weight[node];
        }
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            if (selected[source(*ei, G)] && selected[target(*ei, G)]) {
                total_weight_sum += G[*ei].weight;
            }
        }
        return total_weight_sum / nodes.size();
    }


    SubgraphResult Dinkelbach(double lower_bound, double upper_bound, unsigned iterations, double epsilon, double mip_time_limit) {
        vector<Vertex> best_solution;
        for (auto iter = 0; iter < iterations; iter++) {
            if ((upper_bound - lower_bound) < epsilon) {
                break;
            }
            double lambda = (lower_bound + upper_bound) / 2.0;
            QPBOResult qpbo_result = RunQPBO(lambda, false);
            auto solution = qpbo_result.undecided.empty() ? qpbo_result.fixed_in : RunMIP(qpbo_result, lambda, mip_time_limit);
            if (solution.empty()) {
                upper_bound = lambda;
            } else {
                double density = ComputeDensity(solution);
                if (density >= lower_bound) {
                    lower_bound = density;
                    best_solution = solution;
                } else {
                    throw runtime_error("Dinkelbach: computed density is less than best density");
                }
            }
        }
        return {best_solution, lower_bound};
    }

    SubgraphResult Run(unsigned max_neg_steps, unsigned max_local_optima, bool do_peeling, double step_size, unsigned dinkelbach_iterations, double epsilon, double mip_time_limit) {
        // Step 1. Result found by CEP as initial solution
        auto result = CEP::Run(max_neg_steps, max_local_optima, do_peeling);
        // if qpbo_result.undecided is empty and fixed_in is empty, we can directly return result found by GNDS as Optimal
        auto pre_qpbo = RunQPBO(result.density, false);
        if (pre_qpbo.undecided.empty() && pre_qpbo.fixed_in.empty()) {
            return result;
        }

        // Step 2. Find an upper bound for QPBO
        double upper_bound = FindUpperBound(result.density, step_size);

        // Step 3. Refine the solution by Dinkelbach
        return Dinkelbach(result.density, upper_bound, dinkelbach_iterations, epsilon, mip_time_limit);
    }
};