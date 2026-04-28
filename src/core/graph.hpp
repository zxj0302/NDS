#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/json.hpp>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <set>

using namespace std;
using namespace boost;

#ifdef ENABLE_LOG
    #define LOG(msg) \
        do { cerr << msg << std::endl; } while (0)
#else
    #define LOG(msg) \
        do { } while (0)
#endif

class PGraph {
public:
    struct PGraph_Config {
        string input;
        string output;
        bool reverse_weight = false;
        unsigned num_iter = 1;
        double run_time_limit = 0.0;
        
        virtual void load_from_json(const string& filename) {
            ifstream file(filename);
            string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
            auto json = json::parse(content).as_object();
            
            if (json.contains("input")) input = json.at("input").as_string().c_str();
            if (json.contains("output")) output = json.at("output").as_string().c_str();
            if (json.contains("reverse_weight")) reverse_weight = json.at("reverse_weight").as_bool();
            if (json.contains("num_iter")) num_iter = json.at("num_iter").to_number<unsigned>();
            if (json.contains("run_time_limit")) run_time_limit = json.at("run_time_limit").to_number<double>();
        }

        virtual void add_to_json(json::object& cfg) const {
            cfg["input"] = input;
            cfg["output"] = output;
            cfg["reverse_weight"] = reverse_weight;
            cfg["num_iter"] = num_iter;
            cfg["run_time_limit"] = run_time_limit;
        }
    };

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
    string info = "| Start | ";

    PGraph() = default;

public:
    PGraph(const PGraph_Config& cfg) {
        ReadGraph(cfg);
    }

    // Build a new algorithm instance from an already loaded graph state.
    // This avoids re-reading the graph file when multiple algorithms share input.
    PGraph(const PGraph& other) = default;

    void ReadGraph(const PGraph_Config& cfg) {
        ifstream infile(cfg.input);
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
            weight *= (cfg.reverse_weight ? -1.0 : 1.0);
            if (u == v) {
                loop_weight[u] += weight;
            } else {
                add_edge(u, v, EdgeProperty{weight}, G);
            }
            total_weight += weight;
        }
    }

    virtual ~PGraph() = default;

    // Subclasses may add sub-process timing fields into the shared timings dict.
    virtual void add_timings_to_json(json::object&) const {}

    struct SubgraphResult {
        vector<Vertex> nodes;
        double density;
    };

    template<typename ConfigType>
    void output(const ConfigType& cfg, double avg_time, SubgraphResult& result) {
        json::object json_output;
        json_output["status"] = output_status;
        json_output["time"] = avg_time;
        json_output["density"] = result.density;
        json_output["size"] = result.nodes.size();
        sort(result.nodes.begin(), result.nodes.end());
        json::array nodes_array;
        for (const auto& node : result.nodes) {
            nodes_array.push_back(node);
        }
        json_output["nodes"] = nodes_array;
        
        // Build config object
        json::object config;
        config["info"] = info + " End |";
        cfg.add_to_json(config);
        json_output["config"] = config;

        // All timing data in one dict: total wall-clock + any sub-process breakdowns
        json::object timings_obj;
        timings_obj["total"] = avg_time;
        add_timings_to_json(timings_obj);
        json_output["timings"] = timings_obj;

        // Write to file with pretty-printed formatting
        ofstream out(cfg.output);
        if (!out) throw runtime_error("Cannot open " + cfg.output);
        
        // Recursive function to format JSON with proper indentation and fixed-point numbers
        function<string(const json::value&, int)> format_json;
        format_json = [&](const json::value& val, int indent_level) -> string {
            ostringstream result;
            string indent_str(indent_level * 2, ' ');
            
            if (val.is_object()) {
                result << "{\n";
                auto& obj = val.as_object();
                bool first = true;
                for (auto& [key, value] : obj) {
                    if (!first) result << ",\n";
                    first = false;
                    result << string((indent_level + 1) * 2, ' ') << "\"" << key << "\": ";
                    result << format_json(value, indent_level + 1);
                }
                result << "\n" << indent_str << "}";
            } else if (val.is_array()) {
                result << "[";
                auto& arr = val.as_array();
                bool first = true;
                for (auto& elem : arr) {
                    if (!first) result << ", ";
                    first = false;
                    result << format_json(elem, indent_level);
                }
                result << "]";
            } else if (val.is_double()) {
                result << val.as_double();
            } else if (val.is_int64()) {
                result << val.as_int64();
            } else if (val.is_uint64()) {
                result << val.as_uint64();
            } else if (val.is_bool()) {
                result << (val.as_bool() ? "true" : "false");
            } else if (val.is_string()) {
                result << "\"" << val.as_string().c_str() << "\"";
            } else {
                result << "null";
            }
            return result.str();
        };
        
        out << format_json(json_output, 0) << "\n";
    }
};
