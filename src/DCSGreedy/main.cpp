#include "../basic.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <reverse_weight> [num_iterations]" << endl;
        return EXIT_FAILURE;
    }

    string input = argv[1];
    string output = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    unsigned num_its = (argc >= 5) ? stoi(argv[4]) : 1;

    DCSGreedy graph(input, reverse_weight);
    DCSGreedy::SubgraphResult first_result;
    double total_time = 0.0;

    for (unsigned it = 0; it < num_its; ++it) {
        auto graph_copy = graph;
        auto start_time = chrono::high_resolution_clock::now();
        auto result = graph_copy.Run();
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