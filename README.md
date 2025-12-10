# NDS - Negative Densest Subgraph Discovery

A high-performance C++ implementation of algorithms for discovering densest subgraphs in graphs with negative edge weights. This project provides multiple algorithmic approaches, from fast heuristics to exact optimization methods, combining techniques like local search, QPBO (Quadratic Pseudo-Boolean Optimization), and Mixed Integer Programming (MIP).

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Building](#building)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development Notes](#development-notes)

---

## Overview

The negative densest subgraph problem aims to find a subgraph that maximizes the ratio of total edge weights to the number of vertices, where edge weights can be negative. This problem has applications in social network analysis, bioinformatics, and community detection.

This project implements several algorithms:

- **NEG_DSD**: Negative Densest Subgraph Discovery baseline
- **DCSGreedy**: Greedy approximation algorithm
- **CEP**: Core Expansion with Peeling (heuristic method)
- **CEP_MIP**: CEP enhanced with Mixed Integer Programming
- **CEP_QPBO**: CEP with QPBO-based optimization
- **CEP_QPBO_OPT**: Optimized version combining CEP, QPBO, and MIP

---

## Algorithms

### 1. NEG_DSD (Baseline)

Basic algorithm for negative densest subgraph discovery.

### 2. DCSGreedy

Fast greedy approximation that builds the solution incrementally.

### 3. CEP (Core Expansion with Peeling)

A heuristic approach that:

- Starts with a core subgraph
- Expands by adding negative-weight neighbors
- Uses peeling to remove low-contribution vertices
- Employs local search for optimization

### 4. CEP_MIP

Enhances CEP with Mixed Integer Programming for exact solutions within the search space.

### 5. CEP_QPBO

Combines CEP with QPBO (Quadratic Pseudo-Boolean Optimization) for improved optimization.

### 6. CEP_QPBO_OPT (Recommended)

The most sophisticated algorithm combining:

- CEP for initialization
- QPBO for graph pruning and partial solutions
- MIP for exact optimization on reduced problem instances
- Iterative refinement with Dinkelbach's method

---

## Installation

### Prerequisites

**Required:**

- **C++17** compatible compiler (GCC 7+, Clang 5+, or Apple Clang 9+)
- **CMake** 3.26 or higher
- **Boost** libraries (tested with 1.88.0)

**Optional (for MIP-based algorithms):**

- **Gurobi Optimizer** 11.0+ (required for CEP_MIP, CEP_QPBO, CEP_QPBO_OPT)
  - Set `GUROBI_HOME` environment variable or install to default location
  - Requires valid license

**Python (for utilities and visualization):**

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, seaborn, networkx, PyYAML, tqdm, jupyter
- See `requirements.txt` for full dependencies

### Installing Dependencies

**macOS:**

```bash
# Install Boost
brew install boost

# Install CMake
brew install cmake

# Install Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Linux:**

```bash
# Ubuntu/Debian
sudo apt-get install libboost-all-dev cmake g++

# Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Building

### Quick Build

```bash
# Build all algorithms in Release mode
cd src
./build.sh
```

### Build Options

```bash
# Show all options
./build.sh --help

# Build specific algorithm only
./build.sh -a cep_qpbo_opt

# Build in Debug mode
./build.sh -t Debug

# Clean and rebuild
./build.sh -c

# Build without Gurobi (only non-MIP algorithms)
./build.sh --no-gurobi

# Use specific number of parallel jobs
./build.sh -j 8
```

### Manual CMake Build

```bash
mkdir -p build
cd build
cmake ../src
make -j$(nproc)
```

**Executables** are placed in `build/`:

- `build/neg_dsd`
- `build/dcs_greedy`
- `build/cep`
- `build/cep_mip`
- `build/cep_qpbo`
- `build/cep_qpbo_opt`

---

## Usage

### Command Line

All executables take a JSON configuration file as input:

```bash
./build/<algorithm> <config.json>
```

**Example:**

```bash
# Create a configuration file
cat > config.json << EOF
{
  "input": "./input/real-world/Referendum/Referendum.txt",
  "output": "./output/test/result.json",
  "reverse_weight": false,
  "num_iter": 1,
  "do_peeling": false,
  "max_local_optima": 10,
  "toggle_done": 2,
  "toggle_left": 20,
  "max_neg": 200,
  "use_binary": true,
  "use_probe": false,
  "step_size": 1.02,
  "direct_mip_bound": 100,
  "dinkelbach_iterations": 30,
  "epsilon": -1e-05,
  "mip_time_limit": 600,
  "skip_mip_init": true,
  "heuristic_after_mip": true
}
EOF

# Run the algorithm
./build/cep_qpbo_opt config.json
```

### Batch Experiments

For running multiple algorithms on multiple datasets, use `config.yaml` with the Jupyter notebooks:

```bash
# Edit config.yaml to configure datasets and algorithms
# Run batch experiments
jupyter notebook main.ipynb
```

The `config.yaml` file allows you to:

- Toggle multiple algorithms on/off
- Configure parameters for each algorithm
- Run experiments on real-world and synthetic datasets
- Automatically generate JSON config files for each run

### Input Format

Graph files should be in edge list format:

```
# Lines starting with # are comments
vertex1 vertex2 weight
vertex3 vertex4 weight
...
```

- Vertices are 0-indexed integers
- Weights can be positive or negative floating-point numbers
- Self-loops are supported

### Output Format

Algorithms output JSON files with the following structure:

```json
{
  "time": 0.0055,
  "density": 2.058,
  "size": 218,
  "nodes": [31, 33, 105, ...],
  "config": { /* input configuration */ }
}
```

### Batch Configuration (config.yaml)

The `config.yaml` file organizes batch experiments into three sections:

**real-world**: Configuration for real-world datasets

- `input`: List of graph files with toggle flags
- `weight_reverse`: Whether to reverse edge weights (true for signed networks)
- `output`: Output directory
- `competitors`: List of algorithms with their parameters

**synthetic**: Configuration for synthetic graphs

- `input_folder`: List of directories containing synthetic graphs
- Similar structure to real-world section

**test**: Quick testing configuration

- Simplified setup for testing individual graphs

The batch experiment runner (in `main.ipynb`) reads `config.yaml`, generates individual JSON config files for each algorithm/dataset combination, and runs them sequentially.

---

## Configuration

### JSON Configuration File

Each algorithm executable requires a JSON configuration file with the following structure:

**Common parameters (all algorithms):**

```json
{
  "input": "path/to/input/graph.txt",
  "output": "path/to/output/result.json",
  "reverse_weight": false,
  "num_iter": 1
}
```

- `input`: Path to input graph file
- `output`: Path to output JSON file
- `reverse_weight`: Whether to reverse edge weights (useful for signed networks)
- `num_iter`: Number of iterations to run (results are averaged)

**NEG_DSD specific:**

- No additional parameters (uses default C values)

**DCSGreedy specific:**

- No additional parameters

**CEP parameters:**

```json
{
  "do_peeling": false,
  "max_local_optima": 10,
  "toggle_done": 2,
  "toggle_left": 20,
  "max_neg": 200
}
```

- `do_peeling`: Enable/disable peeling phase
- `max_local_optima`: Maximum local search iterations
- `toggle_done`: Iterations before considering convergence
- `toggle_left`: Min iterations left before switching data structures
- `max_neg`: Max negative weight neighbors in expansion

**CEP_MIP additional parameters:**

```json
{
  "use_binary": false,
  "dinkelbach_iterations": 30,
  "epsilon": -1e-05,
  "mip_time_limit": 600
}
```

- `use_binary`: Use binary search for lambda values
- `dinkelbach_iterations`: Max Dinkelbach iterations
- `epsilon`: Convergence threshold
- `mip_time_limit`: Time limit per MIP solve (seconds)

**CEP_QPBO/CEP_QPBO_OPT additional parameters:**

```json
{
  "use_probe": false,
  "step_size": 1.02,
  "direct_mip_bound": 100,
  "skip_mip_init": true,
  "heuristic_after_mip": true
}
```

- `use_probe`: Enable QPBO probing (slower but more accurate)
- `step_size`: Step size for upper bound search
- `direct_mip_bound`: Upper bound for MIP node constraints
- `skip_mip_init`: Skip MIP initialization phase
- `heuristic_after_mip`: Run CEP refinement after MIP

---

## Project Structure

```
NDS/
├── src/
│   ├── CMakeLists.txt          # Main build configuration
│   ├── build.sh                # Build script
│   ├── core/
│   │   ├── graph.hpp           # Base graph class
│   │   └── algorithms/         # Algorithm implementations
│   │       ├── neg_dsd.hpp
│   │       ├── dcs_greedy.hpp
│   │       ├── cep.hpp
│   │       ├── cep_mip.hpp
│   │       ├── cep_qpbo.hpp
│   │       └── cep_qpbo_opt.hpp
│   ├── executables/            # Main entry points
│   │   ├── neg_dsd.cpp
│   │   ├── dcs_greedy.cpp
│   │   ├── cep.cpp
│   │   ├── cep_mip.cpp
│   │   ├── cep_qpbo.cpp
│   │   └── cep_qpbo_opt.cpp
│   ├── external/               # Third-party libraries
│   │   └── QPBO/              # QPBO library
│   ├── cmake/
│   │   └── FindGUROBI.cmake   # CMake module for Gurobi
│   └── utils/                  # Python utilities
│       ├── baselines.py
│       ├── painter.py
│       └── graph_generator/
├── build/                      # Build output directory
├── input/                      # Input graphs
│   ├── real-world/
│   └── synthetic/
├── output/                     # Algorithm outputs
├── config.yaml                 # Experiment configuration
├── requirements.txt            # Python dependencies
├── main.ipynb                  # Main experiment notebook
├── figures.ipynb               # Visualization notebook
└── README.md                   # This file
```

---

## Development Notes

### Implementation TODO:

1. try to use traditional methods (for non-genative weights only methods)
2. design output name for each different configuration for the same algorithm
3. When initializing the pos_weights for CEP, do more 聚合邻居的值！
4. 使用之前的来初始化ceplambda，使用ceplambda来初始化mip
5. CEP_MIP_QPBO里关于mip的部分可以优化一下

## Research Contributions

This project explores several novel contributions to the negative densest subgraph problem:

### Key Contributions:

1. convert the form from f(x)/g(x) to f(x)-\lambda g(x)
2. With initialization from a good value
3. peeling and reduce param size
4. good init for QPBO-I
5. expansion + peeling can truly insrease density sometimes
6. peeling might delete some important nodes wrongly,  and we find some cases that the peeling based methods may fail. However, run CEP that locally optimize the formula will focus on marginal gain, instead of peeling.
7. Two main contribution: (1) improve of existing heuristic method, and (2) workflow for exact method
8. explain why cannot use existing positive weighted solution like MaxFlow, and explain why cannot shift weights to positive ones.
9. Insights behind any tiny design.

---

## Experimental Evaluation

### Planned Experiments:

1. Compare the runtime & density across real-word and synthetic graphs **widely**. Should use many different simulator to see how it performs on different kinds of graphs. Can use avg. ranks, p-value, non-dominated ratio, avg. time to demonstrate.
2. Do the existing works for non-negative weights graphs really not working?
3. 消融实验：1. find lower_bound/upper_bound的必要性，即一个好的初始化能够提升返回的速度, 否则会有很多时间的额外消耗；但是很多情况下这样的时间消耗也是不可忽视的，导致up-down的方法比Dinkelbach更慢。2. 使用QPBO的必要性，即能够大幅削减MIP的时间；3.pruning 图的必要性，即提升QPBO的时间；4. 使用CEP in the middle的用处，即尽量避免MIP；5.初始化MIP并且限制node的必要性，或许能提升速度. 总结下来需要研究如下变种：（1）纯MIP，使用CEP初始化的二分（2）CEP初始化二分+QPBO+MIP （3）加上pruning的效果 （4）再加上使用CEP in the middle的效果 （5）再加上限制node的效果。
4. 总实验：（1）CEP的参数改变的影响 （2）Exact方法的每个组分的有用与否，即上述消融实验 （3）不同方法之间的对比，包含CEP的不同参数之间的，包含Exact方法的不同epsilon，以及baselines （4）不同类别的simulated图上的效果

---

## Implementation Insights & Debug Log

* [11.23 Mon] CEP 扩大neg count居然会减小找到的值！A: 详见Design。这是特性，过多加入（更大的max_neg）会使得peeling时不精确。
* [11.24 Tue] FindUpperBound can return -inf! A: No reset of the pos_weight and other structures after CEP::Run().
* [11.24 Tue] QPBO居然会返回非最优值！使用CEP的结果做pre qpbo时居然会返回全部out! A: It was a bug. I set the 'success' to 'false', which should be true if the undecided nodes of QPBO is empty.
* [11.26 Wed] Have set vertex constrains in MIP, also use the MIP result to refine upper bound.
* [11.26 Wed] Have changed the hard-coded numbers to params, and set percentage things.
* [11.30 Sun] Finished coding of "QPBO里削减图规模，设置array测试顺序，以及Improve时初始化". Prune the graph, and then use the QPBO process. Use Improve with initialization from CEPLambda.
* [12.01 Mon] Using solution from previous MIP run as next MIP's initial solution guess might be misledding, as the new lambda will be at least the same as the solution's density. If that solution is close to the current solution, then it is ok. Otherwise, it might be more time-consuming.
* [12.01 Mon] Implemented CEP after MIP, which might improve the result further. It does help.
* [12.01 Mon] Re-organize the code, like the order of params, and resue some code.
* [12.04 Thu] Re-organize the code, again. Changed the structure a lot, and use config structure instead of too many params.
* [12.05 Fri] I transferred the code from Macbook air M1 with 16G RAM (used Clang) to Windows system with R7 7735H and 32G RAM. M1 has 4 big cores (3.2 GHz) and 4 small cores (2.0 GHz), while R7 7735H has 8 cores (3.2 GHz). I expected that all algorithms should be faster than on Macbook. However, I find that most of them are are about 1.5-3 times slower. It always needs to take two times runtime. It shocked me a lot. I tried to use windows + docker + gcc, and also tried msvc the situation still there. The only difference is that when the graph is hard to solve, like the setting 5 in BA, the msvc is slightly faster than m1 while docker+gcc is much slower than m1. Also tried docker+clang, similar with docker+gcc. And this slowness shows different ratio on different algorithms. For CEP_MIP, which is the most slow one, it takes about 1-1.5 runtime, while for neg_dsd, dcs_greedy and cep it is about 2 times. For CEP_QPBO and CEP_QPBO_OPT, it takes about 2-4 times runtime. This is amazing!

---

## Algorithm Design Details

### For all classes:

```
I am using Fibonacci_heap from BGL. However, I find that for WS_setting_140: Runtime(smaller better): std::priority_queue + lazy update(label stale) 0.42s < BGL Fibonacci_heap with update_lazy() 0.52s < BGL Fibonacci_heap with update() (no lazy update) 0.59s < std::set without lazy update 0.98s. This is because of Fibonacci_heap's complex structure and high constant overhead and the freqent update (erase + insert) operations. Priority_queue does not need to update keys, just insert new keys and label old keys as stale. Can change all update() to update_lazy() if wanted. Another thing, can consider changing of graph to listS.
```

### For CEP:

```
Can use update_lazy() for Fibonacci_heap. Found it can make LocalGreedy a little faster The main bottleneck in CEP(apart from the peeling) is the initialization and update of the std::set/Fibonacci_heap/vector for storing the positive degree of nodes (can be 95%+ runtime ratio). I find that using a set is(can be) more time-consuming than compute the positive weights on the fly in each local search iteration. This is because of the high overhead of set operations(especially when pruning all nodes wight positive weights smaller than a density). However, eventhough, the Peeling() at the beginning of Run() is still dominating (or have similar) the total runtime. If the local search iterations are large enought(e.g. 30+), the overhead of maintaining the set can be amortized, and using set can be faster. Otherwise, it is better to compute the positive weights on the fly.
```

```
Comparison of using set or vector to store positive degrees in CEP (According to WS_setting_140):
If using set, the initialization of the set and pruning a lot of nodes (possibly in the first one/few iterations) can be very time-consuming. However, as items are removed from the set, the size of the set decreases, and the update operations largely decreases. Thus the runtime for each local search iteration may decrease (maybe significantly) as the iterations proceed. The total runtime will keep roughly stable as the number of local search iterations increases. If using vector, the initialization is fast, and no pruning overhead. However, the total time for the Run() increases nearly (but not linear, because as more nodes invalidated, the on-the-fly computation decreases) linearly with the number of local search iterations. Thus I am using hybrid approach now: start with using vector, and switch to set after some iterations. To amortize the overhead of initialization and pruning, changing to set is only toggled when the number of local search iterations are still left a lot. And as the abs(pos_weight) decreases, I was using two Fibonacci_heap to store the positive and the reverse of positive degree separately. This has similar speed with using one set, as Fibonacci_heap can finish the decrease_key operation in O(1) time. However, I changed back to using one set for convenience.
```

```
Have found that sometimes a smaller max_neg could make the final peeling phase in CEP find denser subgraph. This is because that as more nodes are added, the peeling might be affected and peel important nodes. When the max_neg is small, it might be less biased. Just might. This is not a bug maybe, it is characteristics. And it also proves that expansion with max_neg, and then peeling, is helpful!
```

```
Cannot shift edge weights to positive, as if do so, subgraphs with denser unweighted-density will get more gain from this shift. However, can we derive a solution take this into consideration? I mean like finding the density of densest unweighted subgraph and this is already the bound.
```

### For CEP_QPBO:

```
Using Probe() instead of QPBO standard can make the runtime longer, because sometimes it uses too much time to make the unlabeled set smaller, but not smaller that enough. However, the time cost of Probe() can be larger than MIP when the unlabeled set is already small(MIP runs fast enough to solve).Finding a tight upper bound for the init.
```

```
Sometimes the CEP_QPBO does not have significant improvement on the CEP_MIP, this is because the CEP_QPBO runs many iterations for lambda very close to optima, which and when is very hard to solve. However, CEP_MIP does not have a very accuray estimation of upper_bound, thus most iterations are used to with a large enough lambda, which makes the iterations fast. The last some iterations is the hard part, but it does not have so many 'last some'.
```

```
Theorem 1: Assume we use lambda in RunMIP find an 'exact' result s with n nodes and density rho, then all the subgraphs with density larger than rho would have fewer nodes than n_1. This is because we are minimizing lambda * #nodes - weight_sum = maximizing weight_sum - lambda * #nodes = (rho_subgraph - lambda) * #nodes. If there is a subgraph s' has rho' >= rho and n' > n, then the s' should be the exact result instead of s. Note that if the subgraph n is empty, it means all the subgraphs have density smaller than lambda. 
Apart from the above, we can also see that the upper bound of best density should be <= (rho - lambda) * #nodes, because the subgraph with largest density should have number of nodes at least 1. If its density is larger than (rho - lambda) * #nodes, the MIP should find it instead of s. Additionall, if when we set the lower bound, we iterate all 0, 1, and 2 nodes' coombination, i.e. empty set (give density 0), single node (give density max loop weight), and edge (give (loop[u]+loo[v]+w(u, v))/2), we can set the vertex_lower_bound as 3. Thus the best non-found density should be <= (rho - lambda) * #nodes / 3. 

Furthermore, as for a sequence of MIP running result, the lambda set \lambda1, \lambda2... should be increasing, and the result found \rho1, \rho2... should also be increasing. Thus the number of nodes n of densest subgraph should also be < (n1 * (rho1 - l1) / (rho2 - l1)) = Q , for any former MIP result n1 and later result n2 (have density rho2). Note that, Q is larger than n2, othersize in the former MIP iteration the result should be n2. Thus n2 (the node number from the latest MIP) should always be a tighter bound for n. 
```

```
Another thing to note is that, using up-down method to find the upper bound and use binary search later might help the search, because it prunes half search space each time, and has some guaranteed number of iterations to converge to some approximation. However, sometimes it takes longer time to converge too, which is due to the many search for lambda larger than the largest density. In this case, QPBO might have many nodes as un-labeled (actually the ground truth for them should all be 0), and run MIP for them takes time, and return an empty fixed_in. Lower down the upper bound and search again, until end. However, if use bottom-up method and use lower_bound as init lambda, it may converge faster and directly find the solution. The above difference is due to the upper_bound phase, which cannot find a tight upper bound due to QPBO' unlabeled set non-empty. This will cause the upper_bound estimation larger and larger, and shrink it again and again with MIP later.
```

```
Compared with purely peeling, CEP (with both expansion and peeling) combine better with QPBO, as QPBO's fixed_in set can be used as start point to expand, and its fixed_out set is already gone.
```

```
Pruning the graph truly helps the QPBO process to run faster.
```

---

## Performance Considerations

### Data Structure Trade-offs

The implementation uses different data structures based on workload:

- **std::priority_queue with lazy updates**: Often faster than Fibonacci heap due to lower overhead
- **Fibonacci heap**: Better for frequent decrease-key operations
- **std::set vs on-the-fly computation**: Hybrid approach switches based on iteration count

### Optimization Tips

1. **For large graphs**: Use CEP_QPBO_OPT with appropriate `max_neg` limit
2. **For quick results**: Use CEP or DCSGreedy
3. **For exact solutions**: Use CEP_MIP or CEP_QPBO_OPT with sufficient time limits
4. **Memory constraints**: Reduce `max_neg` and `ub_mip_bound` parameters

---

## License

[Add your license information here]

---

## Citation

[Add citation information if this is for a research paper]

---

## Contact

[Add contact information]
