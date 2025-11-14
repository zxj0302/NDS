TODO:

1. see if change the Fibonacci heap in peeling with Set or PQ, and use lazy_update will affect the runtime
2. when doing FindAnchor each iteration, make sure the id mapping instead of usign size_t
3. when using peeling, whether the FindAnchor, or Set works better
4. try to use listS instead of vecS, try to delete node id
5. fix the Vertex and size_t usage，make them consistent or delete id thing


Observations:

1. Use update_lazy() truly helps saves some time. About 0.07s in WS setting 140(update 0.59s, update_lazy() 0.52s)).
   Runtime(smaller better): std::priority_queue + lazy update(label stale) 0.42s < BGL Fibonacci_heap with update_lazy() 0.52s < BGL Fibonacci_heap with update() (no lazy update) 0.59s < std::set without lazy update 0.98s


2. Run() with Pruning_set takes ~0.6s (init before peeling), initialize after peeling saves some time.
