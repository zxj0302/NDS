# Code Refactoring Summary

## Overview
This refactoring improves the codebase by:
1. Introducing configuration structs for each algorithm with default values
2. Reducing the number of parameters passed to `Run()` functions
3. Creating a reusable JSON parsing utility (`json_config.hpp`)
4. Simplifying executables to accept JSON config files instead of many command-line arguments
5. Converting the main configuration from JSON to YAML for easier human editing

## Changes Made

### 1. Algorithm Configuration Structs

Each algorithm now has a `Config` struct with default values:

#### NEG_DSD::Config
```cpp
struct Config {
    vector<double> C_values = {1.0};
};
```

#### DCSGreedy::Config
```cpp
struct Config {
    // No additional parameters needed
};
```

#### CEP::Config
```cpp
struct Config {
    double max_neg = 200;
    unsigned max_local_optima = 10;
    bool do_peeling = false;
};
```

#### CEP_MIP::Config (extends CEP::Config)
```cpp
struct Config : public CEP::Config {
    unsigned dinkelbach_iterations = 30;
    double epsilon = -0.00001;
    double mip_time_limit = 300.0;
    bool use_binary = true;
};
```

#### CEP_QPBO::Config (extends CEP::Config)
```cpp
struct Config : public CEP::Config {
    double step_size = 1.05;
    unsigned ub_mip_bound = 100;
    unsigned dinkelbach_iterations = 30;
    double epsilon = -0.00001;
    double mip_time_limit = 300.0;
    bool use_binary = true;
};
```

#### CEP_QPBO_OPT::Config (extends CEP::Config)
```cpp
struct Config : public CEP::Config {
    double step_size = 1.02;
    unsigned ub_mip_bound = 100;
    unsigned dinkelbach_iterations = 30;
    double epsilon = -0.00001;
    double mip_time_limit = 300.0;
    bool use_binary = true;
    bool use_probe = false;
    bool skip_mip_init = true;
    bool heuristic_after_mip = true;
};
```

### 2. JSON Configuration Utility

A reusable JSON parser (`src/core/json_config.hpp`) provides:
- `JSONConfig::load_from_file(filename)` - Load JSON from file
- `JSONValue` - Type-safe value wrapper with conversions
- Methods: `has()`, `get_double()`, `get_int()`, `get_unsigned()`, `get_bool()`, `get_double_array()`

Each config struct supports:
- `from_json_file(const string& filename)` - Static factory method
- `load_from_json(const JSONConfig& json)` - Instance method for loading parameters

Example:
```cpp
CEP::Config config = CEP::Config::from_json_file("temp/config_abc123.json");
// Or:
JSONConfig json = JSONConfig::load_from_file("config.json");
CEP::Config config;
config.load_from_json(json);
```

### 3. Simplified Executables

All executables in `src/executables/` now accept **JSON config files** for algorithm parameters:

**NEG_DSD & DCSGreedy** (5 arguments):
```bash
./build/neg_dsd <input> <output> <reverse> <config_json>
./build/dcs_greedy <input> <output> <reverse> <config_json>
```

**CEP family** (7 arguments):
```bash
./build/cep <input> <output> <reverse> <toggle_done> <toggle_left> <config_json>
./build/cep_mip <input> <output> <reverse> <toggle_done> <toggle_left> <config_json>
./build/cep_qpbo <input> <output> <reverse> <toggle_done> <toggle_left> <config_json>
./build/cep_qpbo_opt <input> <output> <reverse> <toggle_done> <toggle_left> <config_json>
```

This is a **massive simplification**:
- CEP: 10 args → 7 args
- CEP_MIP: 14 args → 7 args
- CEP_QPBO: 16 args → 7 args
- CEP_QPBO_OPT: 19 args → 7 args

### 4. YAML Configuration

The main configuration file is now in YAML format (`config.yaml`) for easier editing:

```yaml
real-world:
  input:
    - path: ./input/real-world/Abortion/Abortion.txt
      toggle: true
  
  competitors:
    - name: CEP_QPBO_OPT
      toggle: true
      exe: ./build/cep_qpbo_opt
      params:
        max_neg: 200
        max_local_optima: 10
        do_peeling: false
        step_size: 1.02
        use_binary: true
```

### 5. Updated baselines.py

The `baselines.py` script now:
- Creates temporary JSON config files in `./temp/` directory (one per algorithm run)
- Uses MD5 hash of params to generate unique filenames: `config_abc123.json`
- Passes config file paths to executables instead of long command-line arguments
- Cleans up temporary config files after execution (cleanup in `finally` block)

Key changes:
```python
def write_config_json(params, config_dir="./temp"):
    """Write algorithm parameters to a JSON config file."""
    config_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    config_path = os.path.join(config_dir, f"config_{config_hash}.json")
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)
    return config_path
```

A new runner script `baselines_runner.py` provides a command-line interface:

```bash
# Run real-world experiments
python src/utils/baselines_runner.py config.yaml -e real-world

# Run synthetic experiments in folder mode
python src/utils/baselines_runner.py config.yaml -e synthetic -f

# Run without skipping existing outputs
python src/utils/baselines_runner.py config.yaml -e test --no-skip
```

## Benefits

1. **Dramatically Simpler Executables**: Reduced from 10-19 arguments to just 5-7 arguments
2. **Cleaner Code**: Algorithms no longer need 10+ parameters in their Run() functions
3. **Reusable JSON Parser**: `json_config.hpp` eliminates code duplication across all algorithms
4. **Easier Maintenance**: Adding new parameters only requires updating the config struct
5. **Default Values**: All parameters have sensible defaults defined in one place
6. **Better Configuration**: YAML is much easier for humans to read and edit than JSON
7. **Type Safety**: Config structs and JSONValue class provide compile-time type checking
8. **Automatic Cleanup**: Temporary JSON files are cleaned up automatically

## Migration Guide

### For Python Users

Replace:
```python
import json
config = json.load(open('config.json'))
```

With:
```python
import yaml
config = yaml.safe_load(open('config.yaml'))
```

### For C++ Users

Old way:
```cpp
CEP graph(input, reverse_weight, toggle_done, toggle_left);
auto result = graph.Run(max_neg, max_local_optima, do_peeling);
```

New way:
```cpp
CEP graph(input, reverse_weight, toggle_done, toggle_left);
CEP::Config config = CEP::Config::from_json_file("config.json");
auto result = graph.Run(config);
```

Or with default values:
```cpp
CEP graph(input, reverse_weight, toggle_done, toggle_left);
CEP::Config config;
config.max_neg = 150;  // Override only what you need
auto result = graph.Run(config);
```

## Files Modified

### New Files
- `src/core/json_config.hpp` - Reusable JSON parsing utility (JSONConfig and JSONValue classes)
- `config.yaml` - Human-friendly YAML configuration file
- `src/utils/baselines_runner.py` - Command-line interface for running experiments
- `temp/` - Directory for temporary JSON config files

### Algorithm Headers (src/core/algorithms/)
- `neg_dsd.hpp` - Added Config struct with `C_values`, uses json_config.hpp
- `dcs_greedy.hpp` - Added empty Config struct (no params), uses json_config.hpp
- `cep.hpp` - Added Config struct with 3 params, uses json_config.hpp
- `cep_mip.hpp` - Added Config extending CEP::Config with 4 additional params
- `cep_qpbo.hpp` - Added Config extending CEP::Config with 6 additional params
- `cep_qpbo_opt.hpp` - Added Config extending CEP::Config with 9 additional params

### Executables (src/executables/)
- `neg_dsd.cpp` - Simplified to 5 args (was complex arg parsing)
- `dcs_greedy.cpp` - Simplified to 5 args (was complex arg parsing)
- `cep.cpp` - Simplified from 10 args to 7 args
- `cep_mip.cpp` - Simplified from 14 args to 7 args
- `cep_qpbo.cpp` - Simplified from 16 args to 7 args
- `cep_qpbo_opt.cpp` - Simplified from 19 args to 7 args

### Python Scripts (src/utils/)
- `baselines.py` - Complete rewrite: writes JSON to temp/, clean subprocess calls, automatic cleanup

## Dependencies

New Python dependency:
```bash
pip install pyyaml
```

(This is likely already installed as it's a common dependency)
