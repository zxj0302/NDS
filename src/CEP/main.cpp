#include "../basic.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 10) {
        cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <reverse_weight> [toggle_done] [toggle_left] [max_neg_steps] [max_local_optima] [peeling] [num_its]" << endl;
        return EXIT_FAILURE;
    }

    string input = argv[1];
    string output = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    unsigned toggle_done = (argc >= 5) ? stoi(argv[4]) : 2;
    unsigned toggle_left = (argc >= 6) ? stoi(argv[5]) : 20;
    double max_neg_steps = (argc >= 7) ? stod(argv[6]) : 100;
    unsigned max_local_optima = (argc >= 8) ? stoi(argv[7]) : 20;
    bool do_peeling = (argc >= 9) ? (string(argv[8]) == "1") : true;
    unsigned num_its = (argc >= 10) ? stoi(argv[9]) : 1;
    CEP graph(input, reverse_weight, toggle_done, toggle_left);
    CEP::SubgraphResult first_result;
    double total_time = 0.0;

    for (unsigned it = 0; it < num_its; ++it) {
        auto graph_copy = graph;
        auto start_time = chrono::high_resolution_clock::now();
        auto result = graph_copy.Run(max_neg_steps, max_local_optima, do_peeling);
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