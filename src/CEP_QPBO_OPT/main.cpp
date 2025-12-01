#include "../basic.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 19) {
        cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <reverse_weight> [toggle_done] [toggle_left] [max_neg_steps] "
             << "[max_local_optima] [peeling] [step_size] [ub_mip_bound] [dinkelbach_iterations] [epsilon] [mip_time_limit] [use_binary] [use_probe] [skip_mip_init] [heuristic_after_mip] [num_its]" << endl;
        return EXIT_FAILURE;
    }

    string input = argv[1];
    string output = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    unsigned toggle_done = (argc >= 5) ? stoi(argv[4]) : 2;
    unsigned toggle_left = (argc >= 6) ? stoi(argv[5]) : 20;
    double max_neg_steps = (argc >= 7) ? stod(argv[6]) : 100;
    unsigned max_local_optima = (argc >= 8) ? stoi(argv[7]) : 10;
    bool do_peeling = (argc >= 9) ? (string(argv[8]) == "1") : true;
    double step_size = (argc >= 10) ? stod(argv[9]) : 1.05;
    unsigned ub_mip_bound = (argc >= 11) ? stoi(argv[10]) : 10;
    unsigned dinkelbach_iterations = (argc >= 12) ? stoi(argv[11]) : 10;
    double epsilon = (argc >= 13) ? stod(argv[12]) : 1e-4;
    double mip_time_limit = (argc >= 14) ? stod(argv[13]) : 300.0;
    bool use_binary = (argc >= 15) ? (string(argv[14]) == "1") : true;
    bool use_probe = (argc >= 16) ? (string(argv[15]) == "1") : true;
    bool skip_mip_init = (argc >= 17) ? (string(argv[16]) == "1") : true;
    bool heuristic_after_mip = (argc >= 18) ? (string(argv[17]) == "1") : true;
    unsigned num_its = (argc >= 19) ? stoi(argv[18]) : 1;

    CEP_QPBO_OPT graph(input, reverse_weight, toggle_done, toggle_left);
    CEP_QPBO_OPT::SubgraphResult first_result;
    double total_time = 0.0;

    for (unsigned it = 0; it < num_its; ++it) {
        auto graph_copy = graph;
        auto start_time = chrono::high_resolution_clock::now();
        auto result = graph_copy.Run(max_neg_steps, max_local_optima, do_peeling, step_size, ub_mip_bound, dinkelbach_iterations, epsilon, mip_time_limit, use_binary, use_probe, skip_mip_init, heuristic_after_mip);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        double time_seconds = duration.count() / 1e9;
        total_time += time_seconds;
        if (it == 0) {
            first_result = result;
        }
    }

    graph.output(output, total_time / num_its, first_result, argc, argv);

    return 0;
}