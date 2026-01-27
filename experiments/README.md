# Experiments

This directory contains experimental results organized by model scale.

## Structure

- `01_pythia_160m/` - Small-scale experiments (4090-friendly)
- `02_llama_8b/` - Medium-scale experiments  
- `03_scaling_law/` - Community-contributed large-scale experiments

## Contributing Results

If you run experiments on larger models, please:
1. Create a subdirectory with your model name
2. Include:
   - Experimental configuration
   - Results (rank reduction metrics)
   - Visualizations
   - A brief README describing your setup

See our [Help Wanted issue](https://github.com/ForestR/NeuroEntropy/issues/1) for verification experiments on large models.
