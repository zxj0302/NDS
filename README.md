TODO:

1. change to listS for edges

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
