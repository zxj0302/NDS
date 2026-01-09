#include "../core/algorithms/miqcp.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_json>" << endl;
        return EXIT_FAILURE;
    }

    auto cfg = MIQCP::MIQCP_Config();
    cfg.load_from_json(argv[1]);
    MIQCP graph(cfg);
    
    PGraph::SubgraphResult first_result;
    double total_time = 0.0;
    
    for (unsigned iter = 0; iter < cfg.num_iter; ++iter) {
        auto start_time = chrono::high_resolution_clock::now();
        auto result = graph.Run(cfg);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e9;
        total_time += time_seconds;
        
        if (iter == 0) {
            first_result = result;
        }
        
        if (iter < cfg.num_iter - 1) {
            graph.Reset(cfg);
        }
    }

    graph.output(cfg, total_time / cfg.num_iter, first_result);

    return 0;
}
