# Refactoring Complete - Final Version ✅

## Summary

Successfully refactored the NDS codebase with all requested improvements:

### Key Changes:

1. **Use Boost.JSON instead of custom parser** ✅
   - Removed custom `json_config.hpp`
   - Using `boost::json` library (included in `graph.hpp`)
   - All Config structs use `boost::json::parse()` and `to_number<T>()`

2. **Executables accept only config file** ✅
   - **All executables now**: `./program <config_json>`
   - **Before**: `./cep_qpbo_opt input output reverse toggle_done toggle_left max_neg ... (19 args!)`
   - **Now**: `./cep_qpbo_opt config.json` (1 arg!)

3. **Config struct inside each algorithm class** ✅
   - `NEG_DSD::Config`, `DCSGreedy::Config`, `CEP::Config`, etc.
   - Accessible via `algorithm.config`
   - Includes all parameters: `input`, `output`, `reverse_weight`, `num_iter`, and algorithm-specific params

4. **Functions use config directly** ✅
   - No more parameter passing between functions
   - `CEP::Run()` uses `config.max_neg`, `config.max_local_optima`, etc.
   - `FindLowerBound()` now takes no parameters, uses class config
   - `PruningModeToggle()` uses `config.toggle_done` and `config.toggle_left`

5. **num_iter properly implemented** ✅
   - All executables loop `config.num_iter` times
   - Tracks best result across iterations
   - Averages time across iterations
   - Calls `Reset()` between iterations (for CEP family)

## Architecture:

```
Algorithm Class (e.g., CEP)
├── Config struct (inside class)
│   ├── input, output, reverse_weight
│   ├── num_iter
│   ├── Algorithm-specific params
│   └── load_from_json(filename)
├── config member variable
├── Constructor(config_file)
│   └── Loads config, reads graph
└── Run() - uses config.* directly
```

## Example Usage:

### C++ Executable:
```bash
# Create config file
cat > config.json << EOF
{
  "input": "./input/real-world/Abortion/Abortion.txt",
  "output": "./output/result.json",
  "reverse_weight": false,
  "toggle_done": 2,
  "toggle_left": 20,
  "max_neg": 150,
  "max_local_optima": 5,
  "do_peeling": true,
  "num_iter": 3
}
EOF

# Run with single argument
./build/cep config.json
```

### Python Integration (baselines.py):
```python
# Python writes complete config
full_config = {
    'input': dataset_path,
    'output': output_path,
    'reverse_weight': False,
    'toggle_done': 2,
    'toggle_left': 20,
    'max_neg': 200,
    'num_iter': 1
}

config_path = write_config_json(full_config)
subprocess.run([program, config_path])
```

## Technical Details:

### JSON Parsing:
- Using `boost::json::parse(content).as_object()`
- Safe type conversion with `to_number<T>()`
- Works with both integers and floats in JSON

### Config Structure:
```cpp
class CEP : public PGraph {
public:
    struct Config {
        string input, output;
        bool reverse_weight = false;
        unsigned toggle_done = 2, toggle_left = 20;
        double max_neg = 200;
        unsigned max_local_optima = 10;
        bool do_peeling = false;
        unsigned num_iter = 1;
        
        void load_from_json(const string& filename);
    };
    
    Config config;  // Accessible throughout class
    
    CEP(const string& config_file);
    SubgraphResult Run();  // Uses config directly
};
```

### Inheritance:
- `CEP_MIP::Config extends CEP::Config`
- `CEP_QPBO::Config extends CEP::Config`
- `CEP_QPBO_OPT::Config extends CEP::Config`
- Child classes call `CEP::Config::load_from_json()` then load their own params

## Compilation:

```bash
cd build
cmake ../src
make -j4
```

All 6 executables compile successfully:
- ✅ neg_dsd
- ✅ dcs_greedy  
- ✅ cep
- ✅ cep_mip
- ✅ cep_qpbo
- ✅ cep_qpbo_opt

## Testing:

```bash
# Test CEP
./build/cep ./temp/test_cep_config.json

# Output:
# {
#   "time": 0.003489,
#   "density": 2.048354,
#   ...
# }
```

## Benefits:

1. **Dramatically simpler**: 19 arguments → 1 argument
2. **Type-safe**: Boost.JSON with proper type conversion
3. **No code duplication**: Config in one place, used everywhere
4. **Clean architecture**: Functions access config directly
5. **Easy to extend**: Add new parameter = update Config struct only
6. **Proper iteration handling**: num_iter works correctly
7. **Standard library**: Using Boost instead of custom code

---

**Status**: ✅ All issues fixed and tested
**Date**: 2025-12-02
