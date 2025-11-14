/*
    * basic.hpp
    * author: Zhu Xiangju
    * 
    * Note: For all classes:
    * Note: I am using Fibonacci_heap from BGL. However, I find that for WS_setting_140:
    * Note: Runtime(smaller better): std::priority_queue + lazy update(label stale) 0.42s < BGL Fibonacci_heap with update_lazy() 0.52s < BGL Fibonacci_heap with update() (no lazy update) 0.59s < std::set without lazy update 0.98s
    * Note: This is because of Fibonacci_heap's complex structure and high constant overhead and the freqent update (erase + insert) operations.
    * Note: Priority_queue does not need to update keys, just insert new keys and label old keys as stale.
    * Note: Can change all update() to update_lazy() if wanted.
    * Note: Another thing, I know I don't need to store the id of each vertex in VertexProperty because of vecS, but I just want to eaze the further changes of graph to listS if needed.
    * 
    * Note: For CEP:
    * Note: Can use update_lazy() for Fibonacci_heap. Found it can make LocalGreedy a little faster
    * Note: The main bottleneck in CEP(apart from the peeling) is the initialization and update of the std::set/Fibonacci_heap/vector for storing the positive degree of nodes (can be 95%+ runtime ratio).
    * Note: I find that using a set is(can be, in many cases) more time-consuming than compute the positive weights on the fly in each local search iteration.
    * Note: This is because of the high overhead of set operations(especially when pruning all nodes wight positive weights smaller than a density).
    * Note: However, eventhough, the Peeling() at the beginning of Run() is still dominating (or have similar) the total runtime.
    * Note: If the local search iterations are more enought, the overhead of maintaining the set can be amortized, and using set can be faster.
    * Note: Otherwise, it is better to compute the positive weights on the fly.
    * 
    * Note: Comparison of using set or vector to store positive degrees in CEP (According to WS_setting_140):
    * Note: If using set, the initialization of the set and pruning a lot of nodes (possibly in the first one/few iterations) can be very time-consuming.
    * Note: However, as items are removed from the set, the size of the set decreases, and the update operations largely decreases.
    * Note: Thus the runtime for each local search iteration may decrease (maybe significantly) as the iterations proceed.
    * Note: The total runtime will keep roughly stable as the number of local search iterations increases.
    * Note: If using vector, the initialization is fast, and no pruning overhead. However, the total time for the Run() increasely nearly 
    * Note: (but not linear, because as more nodes invalidated, the on-the-fly computation decreases) linearly with the number of local search iterations.
    * Note: Thus I am using hybrid approach now: start with using vector, and switch to set after some iterations. To amortize the overhead of initialization and pruning,
    * Note: changing to set is only toggled when the number of local search iterations are still left a lot.
    * Note: And as the abs(pos_weight) decreases, I am using two Fibonacci_heap to store the positive and the reverse of positive degree separately.
    * Note: This is faster than using one set, as Fibonacci_heap can finish the decrease_key operation in O(1) time.
*/


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

using namespace std;
using namespace boost;

class PGraph {
protected:
    struct VertexProperty {
        size_t id = 0;
    };

    struct EdgeProperty {
        double weight = 0.0;
    };

    using Graph = adjacency_list<vecS, vecS, undirectedS, VertexProperty, EdgeProperty>;
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

    PGraph() = default;

public:
    PGraph(const string& input, bool reverse_weight) {
        ReadGraph(input, reverse_weight);
    }

    virtual void ReadGraph(const string& input, bool reverse_weight) {
        try {
            ifstream infile(input);
            size_t n = 0, m = 0;
            string line;
            getline(infile, line);
            istringstream iss_first(line);
            iss_first >> n >> m;
            for (size_t i = 0; i < n; ++i) {
                Vertex v = add_vertex(G);
                G[v].id = i;
            }
            valid = vector<bool>(n, true);
            
            while (getline(infile, line)) {
                istringstream iss(line);
                size_t u, v;
                double weight;
                iss >> u >> v >> weight;
                weight *= (reverse_weight ? -1.0 : 1.0);
                add_edge(u, v, EdgeProperty{weight}, G);
                total_weight += weight;
            }
        } catch(const std::exception& e) {
            cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
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
                if (u != v) {
                    pos_deg[v] += weight;
                }
            } else {
                neg_deg[u] += -weight;
                if (u != v) {
                    neg_deg[v] += -weight;
                }
            }
        }
    }

    SubgraphResult Peeling(double C = 1.0) {
        fill(valid.begin(), valid.end(), true);
        MinHeap pq;
        vector<MinHeap::handle_type> handles(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            auto h = pq.push(MinHeapNode{C * pos_deg[*vi] - neg_deg[*vi], *vi});
            handles[G[*vi].id] = h;
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
            valid[G[u].id] = false;
            remove_order.push_back(G[u].id);
            current_vertex_count--;

            for (auto [edge_it, edge_end] = out_edges(u, G); edge_it != edge_end; ++edge_it) {
                auto v = target(*edge_it, G);
                if (valid[G[v].id] || v == u) {
                    double weight = G[*edge_it].weight;
                    current_weight_sum -= weight;
                    if (u == v) continue;
                    auto new_key = (*handles[G[v].id]).key - (weight > 0 ? C * weight : weight);
                    pq.update(handles[G[v].id], MinHeapNode{new_key, v});
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
        Edge max_edge;
        for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
            if (G[*ei].weight > max_weight) {
                max_weight = G[*ei].weight;
                max_edge = *ei;
            }
        }
        return {{G[source(max_edge, G)].id, G[target(max_edge, G)].id}, max_weight / 2};
    }

    SubgraphResult Peeling(bool positive_only = false) {
        fill(valid.begin(), valid.end(), true);
        MinHeap pq;
        vector<MinHeap::handle_type> handles(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            double degree = 0.0;
            for (auto [edge_it, edge_end] = out_edges(*vi, G); edge_it != edge_end; ++edge_it) {
                double weight = G[*edge_it].weight;
                if (!positive_only || weight > 0) {
                    degree += weight;
                }
            }
            auto h = pq.push(MinHeapNode{degree, *vi});
            handles[G[*vi].id] = h;
        }
        auto current_weight_sum = total_weight;
        if (positive_only) {
            current_weight_sum = 0.0;
            for (auto [ei, ee] = edges(G); ei != ee; ++ei) {
                if (G[*ei].weight > 0) {
                    current_weight_sum += G[*ei].weight;
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
            valid[G[u].id] = false;
            remove_order.push_back(G[u].id);
            current_vertex_count--;

            for (auto [edge_it, edge_end] = out_edges(u, G); edge_it != edge_end; ++edge_it) {
                auto v = target(*edge_it, G);
                double weight = G[*edge_it].weight;
                if (positive_only && weight < 0) continue;
                if (valid[G[v].id] || v == u) {
                    current_weight_sum -= weight;
                    if (u == v) continue;
                    auto new_key = (*handles[G[v].id]).key - weight;
                    pq.update(handles[G[v].id], MinHeapNode{new_key, v});
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

    SubgraphResult MaxConnectedComponent(const vector<size_t>& nodes) {
        set<size_t> node_set(nodes.begin(), nodes.end());
        set<size_t> visited;
        SubgraphResult best {{}, -numeric_limits<double>::infinity()};

        for (auto start : nodes) {
            if (visited.count(start)) continue;
            
            // BFS to find component
            vector<size_t> component = {start};
            visited.insert(start);
            
            for (size_t i = 0; i < component.size(); i++) {
                Vertex u = vertex(component[i], G);
                for (auto [e, e_end] = out_edges(u, G); e != e_end; ++e) {
                    // if (G[*e].weight == 0.0) continue;
                    size_t v_id = G[target(*e, G)].id;
                    if (node_set.count(v_id) && !visited.count(v_id)) {
                        visited.insert(v_id);
                        component.push_back(v_id);
                    }
                }
            }
            
            double total_weight = 0.0;
            for (auto u_id : component) {
                Vertex u = vertex(u_id, G);
                for (auto [e, e_end] = out_edges(u, G); e != e_end; ++e) {
                    size_t v_id = G[target(*e, G)].id;
                    if (node_set.count(v_id) && u_id <= v_id) {
                        total_weight += G[*e].weight;
                    }
                }
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
private: 
    struct MaxHeapNode {
        double key;
        Vertex node;
        bool operator<(const MaxHeapNode& other) const {
            return key < other.key || (key == other.key && node < other.node);
        }
    };
    using MaxHeap = heap::fibonacci_heap<MaxHeapNode>;
    enum class Status {
        Out,
        Fringe,
        In
    };

    vector<Status> status;
    vector<double> loop_weight;
    vector<size_t> neighbor_in_count;
    // TODO: use set first, need to change to Fibonacci heap later to see performance
    set<MaxHeapNode> pruning_set;
    vector<set<MaxHeapNode>::iterator> pruning_handles;

public:
    CEP(const string& input, bool reverse_weight) {
        ReadGraph(input, reverse_weight);
    }

    void InitPruningSet() {
        pruning_handles.resize(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            double degree = loop_weight[G[*vi].id];
            for (auto [edge_it, edge_end] = out_edges(*vi, G); edge_it != edge_end; ++edge_it) {
                if (G[*edge_it].weight > 0) {
                    degree += G[*edge_it].weight;
                }
            }
            pruning_handles[G[*vi].id] = pruning_set.insert({degree, *vi}).first;
        }
    }

    void ReadGraph(const string& input, bool reverse_weight) override {
        try {
            ifstream infile(input);
            size_t n = 0, m = 0;
            string line;
            getline(infile, line);
            istringstream iss_first(line);
            iss_first >> n >> m;
            for (size_t i = 0; i < n; ++i) {
                Vertex v = add_vertex(G);
                G[v].id = i;
            }
            valid = vector<bool>(n, true);
            status = vector<Status>(n, Status::Out);
            loop_weight = vector<double>(n, 0.0);
            neighbor_in_count = vector<size_t>(n, 0);
            
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
        } catch(const std::exception& e) {
            cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    SubgraphResult Peeling() {
        MinHeap pq;
        vector<MinHeap::handle_type> handles(num_vertices(G));
        for (auto [vi, ve] = vertices(G); vi != ve; ++vi) {
            double degree = loop_weight[G[*vi].id];
            for (auto [edge_it, edge_end] = out_edges(*vi, G); edge_it != edge_end; ++edge_it) {
                degree += G[*edge_it].weight;
            }
            auto h = pq.push(MinHeapNode{degree, *vi});
            handles[G[*vi].id] = h;
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
            valid[G[u].id] = false;
            remove_order.push_back(G[u].id);
            current_vertex_count--;
            current_weight_sum -= loop_weight[G[u].id];

            for (auto [edge_it, edge_end] = out_edges(u, G); edge_it != edge_end; ++edge_it) {
                auto v = target(*edge_it, G);
                double weight = G[*edge_it].weight;
                if (valid[G[v].id]) {
                    current_weight_sum -= weight;
                    auto new_key = (*handles[G[v].id]).key - weight;
                    pq.update(handles[G[v].id], MinHeapNode{new_key, v});
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

    void Pruning(double threshold_density) {
        while (!pruning_set.empty() && pruning_set.begin()->key < threshold_density) {
            auto it = pruning_set.begin();
            auto u = it->node;
            pruning_set.erase(it);
            valid[G[u].id] = false;
            // pruning_handles[G[u].id] = MaxHeap::handle_type();

            for (auto [edge_it, edge_end] = out_edges(u, G); edge_it != edge_end; ++edge_it) {
                auto v = target(*edge_it, G);
                if (valid[G[v].id] && G[*edge_it].weight > 0) {
                    auto v_it = pruning_handles[G[v].id];
                    auto new_key = v_it->key - G[*edge_it].weight;
                    pruning_set.erase(v_it);
                    pruning_handles[G[v].id] = pruning_set.insert({new_key, v}).first;
                }
            }
        }
    }

    void Pruning(const SubgraphResult& result) {
        for (auto node_id : result.nodes) {
            auto u = vertex(node_id, G);
            pruning_set.erase(pruning_handles[G[u].id]);
            valid[G[u].id] = false;
            // pruning_handles[G[u].id] = MaxHeap::handle_type();

            for (auto [edge_it, edge_end] = out_edges(u, G); edge_it != edge_end; ++edge_it) {
                auto v = target(*edge_it, G);
                if (valid[G[v].id] && G[*edge_it].weight > 0) {
                    auto v_it = pruning_handles[G[v].id];
                    auto new_key = v_it->key - G[*edge_it].weight;
                    pruning_set.erase(v_it);
                    pruning_handles[G[v].id] = pruning_set.insert({new_key, v}).first;
                }
            }
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
        status[G[anchor].id] = Status::Fringe;
        handles[G[anchor].id] = fringe.push({loop_weight[G[anchor].id], anchor});
        unsigned neg_count = 0;
        Vertex next = anchor;

        // ============== Main loop ==============
        while (next != Traits::null_vertex() && neg_count < max_neg) {
            if (status[G[next].id] == Status::Fringe) {
                // ============== If node is "fringe" → move it to "in" ==============
                status[G[next].id] = Status::In;
                auto item = fringe.top();
                fringe.pop();
                handles[G[next].id] = selected.push({-item.key, next});
                current_weight_sum += item.key;
                for (auto [edge_it, edge_end] = out_edges(next, G); edge_it != edge_end; ++edge_it) {
                    auto neighbor = target(*edge_it, G);
                    auto neighbor_id = G[neighbor].id;
                    if (!valid[neighbor_id]) continue;

                    double edge_weight = G[*edge_it].weight;
                    neighbor_in_count[neighbor_id] += 1;
                    if (status[neighbor_id] == Status::Out) {
                        // Move out → fringe
                        status[neighbor_id] = Status::Fringe;
                        double priority_key = edge_weight + loop_weight[neighbor_id];
                        handles[neighbor_id] = fringe.push({priority_key, neighbor});
                    } else if (status[neighbor_id] == Status::Fringe) {
                        // Update fringe neighbor's key
                        auto h = handles[neighbor_id];
                        fringe.update(h, {(*h).key + edge_weight, neighbor});
                    } else if (status[neighbor_id] == Status::In) {
                        // Update selected neighbor's key
                        auto h = handles[neighbor_id];
                        selected.update(h, {(*h).key - edge_weight, neighbor});
                    } else {
                        throw runtime_error("Invalid status in LocalGreedy");
                    }
                }
            } else if (status[G[next].id] == Status::In) {
                // ============== If node is "in" → move it to "fringe" ==============
                status[G[next].id] = Status::Fringe;
                auto item = selected.top();
                selected.pop();
                handles[G[next].id] = fringe.push({-item.key, next});
                current_weight_sum += item.key;
                for (auto [edge_it, edge_end] = out_edges(next, G); edge_it != edge_end; ++edge_it) {
                    auto neighbor = target(*edge_it, G);
                    auto neighbor_id = G[neighbor].id;
                    if (!valid[neighbor_id]) continue;

                    double edge_weight = G[*edge_it].weight;
                    neighbor_in_count[neighbor_id] -= 1;
                    if (status[neighbor_id] == Status::Fringe) {
                        // Possibly move fringe → out if in_neighbor_count == 0
                        if (neighbor_in_count[neighbor_id] == 0) {
                            status[neighbor_id] = Status::Out;
                            fringe.erase(handles[neighbor_id]);
                            handles[neighbor_id] = MaxHeap::handle_type();
                        } else {
                            auto new_key = (*handles[neighbor_id]).key - edge_weight;
                            fringe.update(handles[neighbor_id], {new_key, neighbor});
                        }
                    } else if (status[neighbor_id] == Status::In) {
                        // Update selected neighbor's key
                        auto h = handles[neighbor_id];
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
            auto top_id = G[top_item.node].id;
            remove_order.push_back(top_id);
            status[top_id] = Status::Out;
            for (auto [edge_it, edge_end] = out_edges(top_item.node, G); edge_it != edge_end; ++edge_it) {
                auto neighbor = target(*edge_it, G);
                auto neighbor_id = G[neighbor].id;
                if (!valid[neighbor_id] || status[neighbor_id] != Status::In) continue;
                auto h = handles[neighbor_id];
                selected.update(h, {(*h).key + G[*edge_it].weight, neighbor});
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
                best_nodes.push_back(G[item.node].id);
            }
            return {best_nodes, max_f};
        }
    }

    SubgraphResult Run(unsigned max_neg, unsigned max_local_optima, bool do_peeling) {
        // Step 1. Contraction by Peeling
        InitPruningSet();
        SubgraphResult best = do_peeling ? Peeling() : SubgraphResult{{}, 0.0};

        // Step 2. Expansion by Multi-Local Search
        for (unsigned it = 0; it < max_local_optima; ++it) {
            Pruning(best.density);
            if (pruning_set.empty() || pruning_set.rbegin()->key <= best.density) break;
            auto result = LocalGreedy(pruning_set.rbegin()->node, max_neg);
            if (result.density > best.density) {
                best = result;
            }
            Pruning(result);
        }
        return best;
    }
};