# Pythia-160M Experiments

Initial proof-of-concept experiments demonstrating metabolic attacks on small models.

## Status

🚧 **In Progress**

## Setup

- Model: EleutherAI/pythia-160m (and 70m, 410m)
- Hardware: Single GPU (4090)
- Framework: PyTorch + Transformers

**Environment:**

```bash
conda activate neuroentropy
```

## Running experiments

Single run (e.g. 160M FP16):

```bash
conda activate neuroentropy
python experiments/01_pythia_160m/run_experiment.py --model 160m --quantization fp16
```

Multiple runs with quiet output:

```bash
conda activate neuroentropy
python experiments/01_pythia_160m/run_experiment.py --model 70m --quantization 4bit --num-runs 3 --verbosity quiet
```

Options: `--model {70m,160m,410m}`, `--quantization {none,fp16,4bit,8bit}`, `--num-runs N`, `--verbosity {quiet,normal,verbose}`, `--seed N`.

## Results

Results will be documented here as experiments progress.
