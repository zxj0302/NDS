#include "../basic.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <reverse_weight> <C_values> [num_iterations]" << endl;
        cerr << "C_values: comma-separated list (e.g., 0.1,1.0,10.0)" << endl;
        return EXIT_FAILURE;
    }

    string input = argv[1];
    string output = argv[2];
    bool reverse_weight = (string(argv[3]) == "1");
    vector<double> C_values;
    stringstream ss(argv[4]);
    string token;
    while (getline(ss, token, ',')) {
        C_values.push_back(stod(token));
    }
    unsigned num_its = (argc >= 6) ? stoi(argv[5]) : 1;

    NEG_DSD graph(input, reverse_weight);
    NEG_DSD::SubgraphResult first_result;
    double total_time = 0.0;

    for (unsigned it = 0; it < num_its; ++it) {
        auto graph_copy = graph;
        auto start_time = chrono::high_resolution_clock::now();
        auto result = graph_copy.Run(C_values);
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