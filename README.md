<div align="center">
  <a href="https://andrew-xqy.github.io/XFlow/">
    <img src="https://raw.githubusercontent.com/Andrew-XQY/XFlow/main/images/logo.png"
         alt="XFlow Logo" width="404" height="156">
  </a>

  <p>
    <a href="https://andrew-xqy.github.io/XFlow/"><b>Documentation</b></a>
    ·
    <a href="https://github.com/Andrew-XQY/XFlow/issues">Report Bug</a>
    ·
    <a href="https://github.com/Andrew-XQY/XFlow/issues">Request Feature</a>
  </p>
</div>

![Downloads](https://img.shields.io/github/downloads/Andrew-XQY/XFlow/total)
![Contributors](https://img.shields.io/github/contributors/Andrew-XQY/XFlow?color=dark-green)
![Issues](https://img.shields.io/github/issues/Andrew-XQY/XFlow)
![License](https://img.shields.io/github/license/Andrew-XQY/XFlow)
![PyPI version](https://img.shields.io/pypi/v/xflow-py.svg)
---

## About the Project

**XFlow** is a lightweight modular machine-learning framework.

Originally created for physics research, it's now evolving toward generic scientific applications ML workflows: **Data → Processing → Modeling**

<p align="center">
  <img src="https://raw.githubusercontent.com/Andrew-XQY/XFlow/ab43da1ef082e09a683d1da21f82e9cef54d4033/images/Xflow.png"
       alt="XFlow Conceptual Design" width="800">
</p>

---

## Core Data Processing Pipeline (Computational Map example)
`flow` is a step-based computation map for data processing.

Inputs (possibly different data types) move through discrete steps. At each step, a sample either passes through unchanged (identity) or is transformed by a node. Nodes can be multi-input and multi-output, so the map can split and merge data streams. Optional meta nodes (debug, checks, routing) can log, validate, stop, or redirect (no loops, deterministic) the pipeline without changing the core step structure.

```mermaid
%%{init: {"themeVariables": {"fontSize": "15px"}, "flowchart": {"htmlLabels": true}}}%%
flowchart TD
  classDef src fill:#0b1220,stroke:#334155,stroke-width:1px,color:#e2e8f0;
  classDef op fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#e2e8f0;
  classDef io fill:#111827,stroke:#94a3b8,stroke-width:1px,color:#e5e7eb;
  classDef gate fill:#1f2937,stroke:#f59e0b,stroke-width:2px,color:#fde68a;
  classDef stop fill:#2a0f12,stroke:#fb7185,stroke-width:2px,color:#fecdd3;

  subgraph Inputs["Inputs"]
    DIR["dir: str<br/>/data/run_042"]:::src
    CFG["config: str<br/>YAML or JSON"]:::src
    A1["sensor A:<br/>array&lt;float&gt;"]:::src
    A2["sensor B:<br/>int"]:::src
  end

  READ["<b>ReadImages</b><br/>(dir -> images)"]:::op
  PARSE["<b>ParseConfig</b><br/>(str -> dict)"]:::op

  DIR --> READ
  CFG --> PARSE

  IMGS["images:<br/>tensor[H,W,C,N]"]:::io
  CONF["config:<br/>dict"]:::io

  READ --> IMGS
  PARSE --> CONF

  LOG["<b>LogConfig</b><br/>(print or save)"]:::op
  CONF --> LOG

  JOIN["<b>AlignAndEnrich</b><br/>(images -> 2 outputs)"]:::op
  IMGS --> JOIN

  subgraph JOIN_OUT[" "]
    direction LR
    ALN["aligned_images:<br/>tensor[...]"]:::io
    REP["report:<br/>md or json"]:::io
  end
  style JOIN_OUT fill:transparent,stroke:transparent

  JOIN --> ALN
  JOIN --> REP

  FUSE["<b>FuseSensors</b><br/>(2 signals -> 1 feature vector)"]:::op
  A1 --> FUSE
  A2 --> FUSE

  FEAT["features:<br/>vector&lt;float&gt;"]:::io
  FUSE --> FEAT

  GATE{"<b>QualityGate</b><br/>(meets requirements?)"}:::gate
  ALN --> GATE

  FIX["<b>Remediate</b><br/>(cleanup, re-run, notify)"]:::op
  STOP["STOP<br/>(fail fast)"]:::stop

  GATE -->|fail| FIX
  FIX --> STOP

  subgraph Outputs["Outputs"]
    OUT["artifacts:<br/>aligned_images + features + report"]:::io
  end

  ALN --> OUT
  FEAT --> OUT
  REP --> OUT

  GATE -->|pass| OUT
```

## Getting Started

### Installation

Install from PyPI:

```bash
pip install xflow-py
```

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Andrew-XQY/XFlow.git
cd XFlow
pip install -e .
```
---

## Built With

<p>
  <a href="https://www.python.org/"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" height="40px" width="40px" /></a>
  <a href="https://www.tensorflow.org/"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" height="40px" width="40px" /></a>
  <a href="https://keras.io/"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/keras/keras-original.svg" height="40px" width="40px" /></a>
  <a href="https://pytorch.org/"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="40px" width="40px" /></a>
</p>
</p>

- Python 3.12
- TensorFlow 2.x
- Keras 3.x
- PyTorch 2.5.x

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
