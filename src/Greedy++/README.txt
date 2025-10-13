README FOR DENSEST SUBGRAPH CODE
ALGORITHM: GREEDY++ ITERATIVE PEELING

------------------------------------------------------------------------------------------------------
FILE INPUT FORMAT
------------------------------------------------------------------------------------------------------
For algorithms for unweighted graphs, the input graph file is required to be of the format:

================
n m
u1 v1
u2 v2
.
.
.
um vm
================

For algorithms for weighted graphs, the input graph file is required to be of the format:

================
n m
u1 v1 w1
u2 v2 w2
.
.
.
um vm wm
================

The tests folder contains several small graphs along with their solutions, to help with testing.

-----------------------------------------------------------------------------------------------------
COMPILATION IN LINUX
------------------------------------------------------------------------------------------------------

To compile all four versions of the code, i.e.,

1. exact max-flow based algorithm for unweighted graphs
2. exact max-flow based algorithm for weighted graphs
3. approximate iterative peeling for unweighted graphs
4. approximate iterative peeling for weighted graphs

simply run
```
sudo chmod +x compile_all.sh
./compile_all.sh
```

-----------------------------------------------------------------------------------------------------
COMPILATION IN WINDOWS
------------------------------------------------------------------------------------------------------

To compile all four versions of the code, i.e.,

1. exact max-flow based algorithm for unweighted graphs
2. exact max-flow based algorithm for weighted graphs
3. approximate iterative peeling for unweighted graphs
4. approximate iterative peeling for weighted graphs

simply run
```
compile_all.sh
```

-----------------------------------------------------------------------------------------------------
RUNNING ON LINUX
-----------------------------------------------------------------------------------------------------

To run the code, the syntax goes as follows:

1. exact max-flow based algorithm for unweighted graphs

```
./exact multiplier < input_file
```

2. exact max-flow based algorithm for weighted graphs

```
./exactweighted multiplier < input_file
```

3. approximate iterative peeling for unweighted graphs

```
./ip no_of_iterations < input_file
```

4. approximate iterative peeling for weighted graphs

```
./ipnw no_of_iterations < input_file
```

-----------------------------------------------------------------------------------------------------
RUNNING ON WINDOWS
-----------------------------------------------------------------------------------------------------

To run the code, the syntax goes as follows:

1. exact max-flow based algorithm for unweighted graphs

```
exact.exe multiplier < input_file
```

2. exact max-flow based algorithm for weighted graphs

```
exactweighted.exe multiplier < input_file
```

3. approximate iterative peeling for unweighted graphs

```
ip.exe no_of_iterations < input_file
```

4. approximate iterative peeling for weighted graphs

```
ipnw.exe no_of_iterations < input_file
```
