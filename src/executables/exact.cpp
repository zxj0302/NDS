#include "../core/algorithms/exact.hpp"
#include <csignal>

volatile sig_atomic_t g_sigterm_received = 0;

static void sigterm_handler(int) {
    g_sigterm_received = 1;
}

int main(int argc, char* argv[]) {
    signal(SIGTERM, sigterm_handler);
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_json>" << endl;
        return EXIT_FAILURE;
    }

    auto cfg = EXACT::EXACT_Config();
    cfg.load_from_json(argv[1]);
    EXACT graph(cfg);
    
    PGraph::SubgraphResult first_result;
    double total_time = 0.0;
    
    chrono::time_point<chrono::high_resolution_clock> first_start_time;

    for (unsigned iter = 0; iter < cfg.num_iter; ++iter) {
        auto start_time = chrono::high_resolution_clock::now();
        if (iter == 0) first_start_time = start_time;

        auto result = graph.Run(cfg, start_time);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e9;
        total_time += time_seconds;

        if (iter == 0) {
            first_result = result;
        }

        if (g_sigterm_received) {
            double elapsed = chrono::duration_cast<chrono::nanoseconds>(end_time - first_start_time).count() / 1e9;
            graph.output(cfg, elapsed, first_result, "timeout", false, graph.final_upper_bound);
            return 0;
        }

        if (iter < cfg.num_iter - 1) {
            graph.Reset(cfg);
        }
    }

    graph.output(cfg, total_time / cfg.num_iter, first_result, "success", false, graph.final_upper_bound);

    return 0;
}