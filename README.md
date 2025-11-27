## TODO:

1. change to listS for edges
2. try to use traditional methods (for non-genative weights only methods)
3. design output name for each different configuration for the same algorithm
4. optimize the CEP_QPBO, can use Probe(), Improve() and other (like CEP in the middle) to improve
5. what if add many peelings in QPBO process?
6. QPBO里削减图规模，设置array测试顺序，以及Improve时初始化
7. dynamic lower_bound and upper_bound, use the CEP to find the bounds better after MIP or QPBO

---

## Contributions (should contain):

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



## Experiments TODO

1. Compare the runtime & density across real-word and synthetic graphs **widely**. Should use many different simulator to see how it performs on different kinds of graphs. Can use avg. ranks, p-value, non-dominated ratio, avg. time to demonstrate.
2. Analyze the parameter settings' influence on the CEP and other works. CEP with/without Peeling.
3. What if initialize the upper_bound in CEP_QPBO randomly, or don't use up-down method to find the upper_bound? Does QPBO really helps? What if Only use MIP?
4. Do the existing works for non-negative weights graphs really not working?
5. Runtime of different epsion for exact methods
6. which one is better, P+BP+I, or P+CEP+I?

---

## Logs and Debug

* [11.23 Mon] CEP 扩大neg count居然会减小找到的值！A: 详见Design。这是特性，过多加入（更大的max_neg）会使得peeling时不精确。
* [11.24 Tue] FindUpperBound can return -inf! A: No reset of the pos_weight and other structures after CEP::Run().
* [11.24 Tue] QPBO居然会返回非最优值！使用CEP的结果做pre qpbo时居然会返回全部out! A: It was a bug. I set the 'success' to 'false', which should be true if the undecided nodes of QPBO is empty.
* [11.26 Wed] Have set vertex constrains in MIP, also use the MIP result to refine upper bound.
* [11.26 Wed] Have changed the hard-coded numbers to params, and set percentage things.

---

## Design

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
