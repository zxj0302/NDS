#include <iostream>
#include <boost/heap/fibonacci_heap.hpp>
#include <set>

int main() {
    // Create a fibonacci_heap with default comparator
    boost::heap::fibonacci_heap<int> heap;
    
    // Insert some values
    heap.push(10);
    heap.push(5);
    heap.push(20);
    heap.push(3);
    heap.push(15);
    
    std::cout << "Inserted values: 10, 5, 20, 3, 15\n";
    std::cout << "Top element (default): " << heap.top() << "\n\n";
    
    // Pop all elements to see the order
    std::cout << "Order of elements (default heap):\n";
    while (!heap.empty()) {
        std::cout << heap.top() << " ";
        heap.pop();
    }
    std::cout << "\n\n";
    
    // Now try with std::greater to create min heap
    boost::heap::fibonacci_heap<int, boost::heap::compare<std::greater<int>>> min_heap;
    
    min_heap.push(10);
    min_heap.push(5);
    min_heap.push(20);
    min_heap.push(3);
    min_heap.push(15);
    
    std::cout << "With std::greater comparator:\n";
    std::cout << "Top element: " << min_heap.top() << "\n\n";
    
    std::cout << "Order of elements (with std::greater):\n";
    while (!min_heap.empty()) {
        std::cout << min_heap.top() << " ";
        min_heap.pop();
    }
    std::cout << "\n";


    struct MinHeapNode {
        double key;
        bool operator<(const MinHeapNode& other) const {
            return key > other.key;
        }
    };
    boost::heap::fibonacci_heap<MinHeapNode> custom_min_heap;
    custom_min_heap.push(MinHeapNode{10.5});
    custom_min_heap.push(MinHeapNode{5.2});
    custom_min_heap.push(MinHeapNode{20.1});
    custom_min_heap.push(MinHeapNode{3.3});
    custom_min_heap.push(MinHeapNode{15.4});
    std::cout << "\nWith custom MinHeapNode:\n";
    std::cout << "Top element: " << custom_min_heap.top().key << "\n\n";
    std::cout << "Order of elements (with custom MinHeapNode):\n";
    while (!custom_min_heap.empty()) {
        std::cout << custom_min_heap.top().key << " ";
        custom_min_heap.pop();
    }
    std::cout << "\n";


    std::multiset<MinHeapNode> ms;
    ms.insert(MinHeapNode{10.5});
    ms.insert(MinHeapNode{5.2});
    ms.insert(MinHeapNode{20.1});
    ms.insert(MinHeapNode{3.3});
    ms.insert(MinHeapNode{15.4});
    std::cout << "\nUsing std::multiset with custom MinHeapNode:\n";
    std::cout << "Order of elements:\n";
    for (const auto& node : ms) {
        std::cout << node.key << " ";
    }
    std::cout << "\n";

    
    return 0;
}