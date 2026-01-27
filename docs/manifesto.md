# Manifesto: Why We Do This

## The Thermodynamics of Intelligence Loss in LLMs

We are not building another attack script library. We are building an **AI Pathology Laboratory**.

### The Problem

Large Language Models (LLMs) have become the foundation of modern AI systems. Yet, we understand remarkably little about their structural vulnerabilities. Traditional adversarial attacks focus on input perturbations, but what if the vulnerability lies deeper—in the very structure of how these models process information?

### The Discovery

We have discovered that LLMs exhibit a fundamental structural fragility: **larger models are more vulnerable to metabolic attacks** that induce spectral collapse. This is not a bug—it's a feature of how these models are trained and how optimization algorithms (like Adam) interact with the Hessian structure of the loss landscape.

### The Mechanism

The attack exploits a subtle but critical property: **Adam's second-moment estimate amplifies noise in directions corresponding to small Hessian eigenvalues**. By carefully crafting "catalysts" that exploit this property, we can induce progressive degradation—a "metabolic cycle" that collapses the effective rank of model activations.

### Why This Matters

1. **Scientific Understanding**: We are mapping the "pathology" of LLMs—understanding how and why they fail.

2. **Security Implications**: If larger models are structurally more fragile, this has profound implications for AI safety and deployment.

3. **Theoretical Contribution**: We are bridging optimization theory, information theory, and AI safety in a novel way.

4. **Open Science**: By building in public, we invite the community to verify, challenge, and extend our findings.

### Our Approach

- **Building in Public**: Every commit is a step toward understanding AI pathology.
- **Community-Driven**: We invite collaborators with larger compute resources to verify our scaling law predictions.
- **Defense-Oriented**: We explicitly challenge the community to develop defenses—including testing whether architectures like DeepSeek's mHC can resist our attacks.

### The Vision

We envision a future where:
- Model vulnerabilities are understood before deployment
- Defenses are developed alongside attacks
- The AI research community collaborates openly on safety-critical research

### Join Us

This is not just code. This is **live science**. Every experiment, every visualization, every finding is documented here in real-time.

**We are not just attacking models. We are discovering the physics of intelligence loss.**

Let's push to main. 🚀
