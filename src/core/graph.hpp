#pragma once

// Note: Gurobi and QPBO headers are included in specific algorithm files that need them
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/json.hpp>
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

    template<typename ConfigType>
    void output(const string& filepath, double avg_time, SubgraphResult& result, const ConfigType& cfg, int argc, char* argv[]) {
        std::sort(result.nodes.begin(), result.nodes.end());
        
        // Build JSON object using Boost.JSON
        boost::json::object json_output;
        json_output["time"] = avg_time;
        json_output["density"] = result.density;
        json_output["size"] = result.nodes.size();
        
        // Convert nodes vector to JSON array
        boost::json::array nodes_array;
        for (const auto& node : result.nodes) {
            nodes_array.push_back(node);
        }
        json_output["nodes"] = nodes_array;
        
        // Build config object
        boost::json::object config_obj;
        config_obj["input"] = cfg.input;
        config_obj["output"] = cfg.output;
        config_obj["reverse_weight"] = cfg.reverse_weight;
        config_obj["num_iter"] = cfg.num_iter;
        
        // Add algorithm-specific parameters
        add_config_params(config_obj);
        json_output["config"] = config_obj;
        
        // Build command string
        string command;
        for (int i = 0; i < argc; ++i) {
            if (i) command += " ";
            command += argv[i];
        }
        json_output["command"] = command;
        
        // Write to file with pretty-printed formatting
        ofstream out(filepath);
        if (!out) throw std::runtime_error("Cannot open " + filepath);
        out << boost::json::serialize(json_output) << "\n";
    }
    
    // Virtual method to be overridden by derived classes
    virtual void add_config_params(boost::json::object& config_obj) {
        // Default: no additional params
    }
};
