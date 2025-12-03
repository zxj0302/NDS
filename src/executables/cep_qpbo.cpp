#include "../core/algorithms/cep_qpbo.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_json>" << endl;
        return EXIT_FAILURE;
    }

    string config_file = argv[1];
    CEP_QPBO graph(config_file);
    
    PGraph::SubgraphResult best_result;
    double total_time = 0.0;
    
    for (unsigned iter = 0; iter < graph.config.num_iter; ++iter) {
        auto start_time = chrono::high_resolution_clock::now();
        auto result = graph.Run();
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e9;
        total_time += time_seconds;
        
        if (iter == 0 || result.density > best_result.density) {
            best_result = result;
        }
        
        if (iter < graph.config.num_iter - 1) {
            graph.Reset();
        }
    }

    graph.output(graph.config.output, total_time / graph.config.num_iter, best_result, graph.config, argc, argv);

    return 0;
}