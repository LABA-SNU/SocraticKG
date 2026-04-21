<div align="center">

<img src="assets/banner.svg" width="100%" alt="SocraticKG — Knowledge Graph Construction via QA-Driven Fact Extraction"/>

[![ACL Findings 2026](https://img.shields.io/badge/ACL%20Findings-2026-b31b1b.svg)](https://2026.aclweb.org/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

[Paper](#) · [Quick Start](#quick-start) · [Results](#results) · [Citation](#citation)

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/main_figure_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/main_figure.png">
  <img src="assets/main_figure.png" width="85%" alt="SocraticKG framework overview"/>
</picture>

<sub>Unstructured text → 5W1H-guided QA pairs → atomic triples → canonicalized knowledge graph</sub>

</div>

---

## Overview

**SocraticKG** introduces question-answer pairs as a structured intermediate representation between raw text and extracted triples. Rather than prompting an LLM to extract triples in a single pass, we first generate self-contained QA pairs guided by the **5W1H framework** (*who, what, when, where, why, how*), then extract atomic triples from each unit.

This interrogative scaffold systematically surfaces implicit causal and procedural dependencies that direct extraction misses — producing graphs that are more complete *and* more structurally coherent, resolving the long-standing trade-off between factual coverage and connectivity.

---

## Method

The pipeline consists of three stages:

**1. 5W1H-Guided QA Generation**
The document is decomposed into self-contained, context-independent QA pairs. The 5W1H framework serves as an analytical lens that surfaces procedural and causal content alongside surface-level facts. All referential expressions (pronouns, definite descriptions) are resolved so that each QA pair stands alone as an extraction unit.

**2. Triple Extraction from QA**
Each QA pair is independently mapped to atomic `(entity₁, relation, entity₂)` triples. Operating on logically self-contained units narrows the semantic boundary of each extraction step and reduces errors typical of single-pass pipelines.

**3. Canonicalization**
Extracted triples are unified through an embedding-based cluster-then-refine procedure combining K-means clustering, hybrid dense-sparse retrieval (BM25 + cosine), and LLM-based synonym resolution — producing a compact, coherent graph with consolidated entities and relations.

---

## Results

### Factual Retention on MINE (%)

| Method | Qwen-2.5 | GPT-4o-mini | GPT-4o | Gemini-2.5 | Claude-4 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Direct Extraction | 66.5 | 68.5 | 78.1 | 84.6 | 86.8 |
| GraphRAG | 59.7 | 49.5 | 49.3 | 48.5 | 52.3 |
| KGGen | 56.7 | 44.3 | 66.4 | 62.5 | 69.1 |
| SoKG (w/o 5W1H) | 67.1 | 80.5 | 83.5 | 85.6 | 94.6 |
| **SoKG (Ours)** | **73.4** | **83.9** | **89.3** | **87.7** | **96.3** |

### Multi-hop Reasoning on HotpotQA Hard Bridge (%)

| Method | Qwen · 2-hop | Qwen · 3-hop | Claude · 2-hop | Claude · 3-hop |
|:---|:---:|:---:|:---:|:---:|
| Direct Extraction | 19.50 | 23.88 | 37.87 | 39.88 |
| GraphRAG | 23.00 | 25.00 | 46.62 | 52.12 |
| KGGen | 16.50 | 18.50 | 38.25 | 46.75 |
| Naive RAG | — | 20.13 | — | 47.88 |
| **SocraticKG** | **23.62** | **27.00** | **48.50** | **56.38** |

SocraticKG is the only KG-based method that **consistently outperforms Naive RAG** across all backbones and retrieval depths.

---

## Quick Start

### Installation

```bash
git clone https://github.com/LABA-SNU/SocraticKG.git
cd SocraticKG
pip install -r requirements.txt
```

### Environment

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

### Minimal Example

```python
from socratickg import SocraticKG

kg = SocraticKG(model="gpt-4o")
graph = kg.build(document="Your unstructured text here ...")
graph.save("my_kg.json")
```

### Reproducing Paper Results

```bash
# MINE benchmark (factual retention)
python scripts/run_mine.py --model claude-4 --method sokg

# HotpotQA downstream evaluation (multi-hop reasoning)
python scripts/run_hotpotqa.py --model claude-4 --hop 3
```

---

## Repository Structure

```
SocraticKG/
├── socratickg/
│   ├── qa_generation.py        # 5W1H-guided QA generation
│   ├── triple_extraction.py    # QA → triple extraction
│   ├── canonicalization.py     # Entity & relation unification
│   └── graph.py                # KG data structure
├── prompts/
│   ├── qa_5w1h.txt             # RO / PS / ID prompt archetypes
│   └── triple_extraction.txt
├── scripts/
│   ├── run_mine.py
│   └── run_hotpotqa.py
├── data/                       # MINE + HotpotQA samples
└── assets/
```

---

## Citation

```bibtex
@article{choi2026socratickg,
  title={SocraticKG: Knowledge Graph Construction via QA-Driven Fact Extraction},
  author={Choi, Sanghyeok and Jeon, Woosang and Yang, Kyuseok and Kim, Taehyeong},
  journal={arXiv preprint arXiv:2601.10003},
  year={2026}
}
```

---

## Acknowledgements

This work was supported by the Technology Innovation Program (MOTIE, Korea, RS-2025-25453780); the National Research Foundation of Korea (RS-2023-00302123) as part of the European Commission's Horizon Europe programme (Grant No. 101135576, INTEND); the IITP grant funded by the Korean government (MSIT) [RS-2021-II211343, AI Graduate School Program (Seoul National University)]; and the Creative-Pioneering Researchers Program through Seoul National University.

---

<div align="center">
<sub>LABA @ Seoul National University</sub>
</div>
