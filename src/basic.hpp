#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/heap/fibonacci_heap.hpp>
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
#include <unordered_set>

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
        for (size_t i = 0; i < n; ++i) {
            Vertex v = add_vertex(G);
        }
        valid = vector<bool>(n, true);
        loop_weight = vector<double>(n, 0.0);
        
        while (getline(infile, line)) {
            istringstream iss(line);
            size_t u, v;
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
        vector<size_t> nodes;
        double density;
    };

    void output(const string& filepath, double avg_time, SubgraphResult& result) {
        ofstream out(filepath);
        if (!out) throw std::runtime_error("Cannot open " + filepath);
        std::sort(result.nodes.begin(), result.nodes.end());
        out << fixed << std::setprecision(6);
        out << "{\n"
            << "  \"time\": " << avg_time << ",\n"
            << "  \"density\": " << result.density << ",\n"
            << "  \"size\": " << result.nodes.size() << ",\n"
            << "  \"nodes\": [";
        for (size_t i = 0; i < result.nodes.size(); ++i) {
            if (i) out << ", ";
            out << result.nodes[i];
        }
        out << "]\n}\n";
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
        vector<size_t> remove_order;
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
        vector<size_t> max_edge;
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
        vector<size_t> remove_order;
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
                size_t u = source(*ei, G);
                size_t v = target(*ei, G);
                if (selected[u] && selected[v] && u <= v) {
                    total_weight_sum += G[*ei].weight;
                }
            }
            best_density = total_weight_sum / (remove_order.size() - best_step);
        }
        return {{remove_order.begin() + best_step, remove_order.end()}, best_density};
    }

    SubgraphResult MaxConnectedComponent(const vector<size_t>& nodes) {
        std::unordered_set<size_t> node_set(nodes.begin(), nodes.end());
        std::unordered_set<size_t> visited;
        visited.reserve(nodes.size());
        SubgraphResult best {{}, -numeric_limits<double>::infinity()};

        for (auto start : nodes) {
            if (visited.count(start)) continue;
            
            // BFS to find component
            vector<size_t> component = {start};
            visited.insert(start);
            
            for (size_t i = 0; i < component.size(); i++) {
                Vertex u = component[i];
                for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                    size_t v = target(*ei, G);
                    if (node_set.count(v) && !visited.count(v)) {
                        visited.insert(v);
                        component.push_back(v);
                    }
                }
            }
            
            double total_weight = 0.0;
            for (auto u : component) {
                for (auto [ei, ee] = out_edges(u, G); ei != ee; ++ei) {
                    size_t v = target(*ei, G);
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
        cout << "Peeling result density: " << S1.density << endl;
        auto S2 = Peeling(true);
        cout << "Positive-only Peeling result density: " << S2.density << endl;
        if (S1.density > S.density) S = S1;
        if (S2.density > S.density) S = S2;

        return MaxConnectedComponent(S.nodes);
    }
};

class CEP : public PGraph {
private: 
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
        vector<size_t> remove_order;
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
        vector<size_t> remove_order;
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
            vector<size_t> best_nodes;
            for (auto& item : best) {
                best_nodes.push_back(item.node);
            }
            return {best_nodes, max_f};
        }
    }

    void PruningSet(const vector<size_t>& nodes, double threshold_density) {
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

    void PruningVector(const vector<size_t>& nodes, double threshold_density) {
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
        for (size_t i = 0; i != pos_weight.size(); ++i) {
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