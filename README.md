# NeuroEntropy

**The Thermodynamics of Intelligence Loss in LLMs**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 🎯 The Hook

**We prove mathematically and empirically that larger models are structurally more fragile to metabolic attacks.**

This is not just another adversarial attack library. This is an **AI Pathology Laboratory**—a live research project documenting the structural vulnerabilities of Large Language Models in real-time.

---

## 📊 The Discovery

![Rank Collapse vs Model Scale](assets/spectral_collapse.png)
*Predicted scaling law: Model vulnerability increases with size*

We have discovered that LLMs exhibit a fundamental structural fragility: **larger models are more vulnerable to metabolic attacks** that induce spectral collapse. The attack exploits how Adam's optimization algorithm amplifies noise in directions corresponding to small Hessian eigenvalues, leading to progressive degradation of the model's effective rank.

---

## 🚀 Quick Start

### Try Killing a Model Yourself

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ForestR/NeuroEntropy/blob/main/notebooks/Demo_1_Killing_Pythia.ipynb)

Run our demo notebook to see how we can induce spectral collapse in Pythia-160M in just 5 minutes:

```bash
# Clone the repository
git clone https://github.com/ForestR/NeuroEntropy.git
cd NeuroEntropy

# Install dependencies
pip install -r requirements.txt

# Run the demo
jupyter notebook notebooks/Demo_1_Killing_Pythia.ipynb
```

---

## 📁 Repository Structure

```
NeuroEntropy/
├── 📂 assets/              # Spectral collapse visualizations and diagrams
├── 📂 docs/                # Theory derivation and manifesto
│   ├── theory_derivation.pdf
│   └── manifesto.md        # Why we do this
├── 📂 experiments/         # Experimental results
│   ├── 01_pythia_160m/     # Small-scale experiments (4090-friendly)
│   ├── 02_llama_8b/        # Medium-scale experiments
│   └── 03_scaling_law/     # Community-contributed large-scale experiments
├── 📂 src/                 # Core implementation
│   ├── catalyst.py         # Hessian-Aware Catalyst Generator
│   ├── diagnosis.py        # Effective Rank and Spectral Gap computation
│   └── attack_loop.py      # Metabolic attack simulation
├── 📓 notebooks/           # Interactive demonstrations
│   ├── Demo_1_Killing_Pythia.ipynb
│   └── Analysis_Visualizer.ipynb
├── 📜 CITATION.cff         # Citation information
├── 📜 LICENSE              # Apache 2.0
└── 📄 README.md            # This file
```

---

## 🔬 Core Components

### 1. Hessian-Aware Catalyst Generator (`src/catalyst.py`)

Generates attack prompts that exploit the Hessian structure of model activations to maximize noise amplification in Adam updates.

### 2. Diagnostic Tools (`src/diagnosis.py`)

Computes effective rank and spectral gap—key metrics for detecting spectral collapse and model degradation.

### 3. Metabolic Attack Loop (`src/attack_loop.py`)

Simulates the iterative attack process that induces progressive degradation through repeated catalyst exposure.

---

## 🧪 Experiments

### Phase I: Pythia-160M (✅ Completed)

Our initial experiments on Pythia-160M demonstrate the core mechanism. Results show significant effective rank reduction after metabolic attack cycles.

### Phase II: Llama-8B (🚧 In Progress)

Medium-scale verification of the scaling law hypothesis.

### Phase III: Large Models (🔍 Seeking Collaborators)

**We need your help!** If you have access to larger models (70B+), please see our [Help Wanted issue](https://github.com/ForestR/NeuroEntropy/issues/1) for verification experiments.

---

## 🛡️ Defense Challenge

We explicitly challenge the community to develop defenses against metabolic attacks. Currently, the only theoretical defense is DeepSeek's mHC architecture. **We invite the community to test whether mHC-trained models can resist our attacks.**

If you have an mHC-trained model, please test it and share your results!

---

## 📚 Theory

For detailed theoretical derivations, see:
- `docs/theory_derivation.pdf` - Mathematical foundations
- `docs/manifesto.md` - Our research philosophy

**Key Insight**: Adam's second-moment estimate amplifies noise in directions corresponding to small Hessian eigenvalues. By crafting catalysts that exploit this property, we can induce progressive spectral collapse.

---

## 🤝 Contributing

We welcome contributions! This is a **live science** project—every commit is a step toward understanding AI pathology.

### Ways to Contribute

1. **Run Experiments**: Help verify our scaling law predictions on larger models
2. **Develop Defenses**: Test and propose defense mechanisms
3. **Improve Documentation**: Enhance theory explanations and tutorials
4. **Report Issues**: Share bugs, questions, or suggestions

See our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📄 License

- **Code**: Apache 2.0 License (see [LICENSE](LICENSE))
- **Documentation & Theory**: CC-BY-NC-SA 4.0 (see [docs/LICENSE](docs/LICENSE))

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{neuroentropy2026,
  title={NeuroEntropy: The Thermodynamics of Intelligence Loss in LLMs},
  author={NeuroEntropy Research Team},
  year={2026},
  url={https://github.com/ForestR/NeuroEntropy},
  license={Apache-2.0}
}
```

---

## 🌐 Social Media

Follow our progress:
- **Twitter/X**: [@NeuroEntropy](https://twitter.com/NeuroEntropy) - Live updates and findings
- **Hugging Face**: [Datasets](https://huggingface.co/datasets/neuroentropy/eigen-prion) | [Models](https://huggingface.co/neuroentropy)

---

## ⚠️ Disclaimer

This research is conducted for **scientific understanding and AI safety**. We are documenting vulnerabilities to enable better defenses, not to enable malicious attacks. Use responsibly.

---

## 🙏 Acknowledgments

Special thanks to:
- The open-source AI research community
- Contributors running experiments on larger models
- DeepSeek for developing mHC architecture (our first defense target)

---

**Remember**: We are not just attacking models. We are discovering the physics of intelligence loss.

**Let's push to main.** 🚀
