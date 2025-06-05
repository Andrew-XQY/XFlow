<p align="center">
  <a href="https://github.com/Andrew-XQY/XFlow">
    <img src="images/logo.png" alt="XFlow Logo" width="128" height="128">
  </a>
  <p align="center">
    A modular deep learning library for flexible research workflows and scalable training pipelines.
    <br/>
    <a href="https://github.com/Andrew-XQY/XFlow/issues">Report Bug</a>
    ·
    <a href="https://github.com/Andrew-XQY/XFlow/issues">Request Feature</a>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/Andrew-XQY/XFlow/total)
![Contributors](https://img.shields.io/github/contributors/Andrew-XQY/XFlow?color=dark-green)
![Issues](https://img.shields.io/github/issues/Andrew-XQY/XFlow)
![License](https://img.shields.io/github/license/Andrew-XQY/XFlow)

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Built With](#-built-with)
- [Getting Started](#-getting-started)
- [License](#-license)
- [Author](#-author)

---

## 🔍 About the Project

**XFlow** is a modular machine learning framework designed for training and evaluating models in structured research workflows. Originally built for image-based tasks in accelerator physics, it is flexible enough to generalize to broader deep learning applications.

> _Includes reusable modules for datasets, models, training loops, visualization, and logging._  
> _Current version: 0.1.0_

---

## ✨ Features

- Modular `core/` for dataset/model/trainer abstraction
- Plug-and-play task logic (`tasks/`)
- Built-in U-Net and ResNet models
- TensorFlow + custom trainer support
- Logging, metrics, and visualization helpers
- Clean `src/` layout with PyPI/pip support

---

## 🛠 Built With

<a href="https://www.python.org/"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" height="40px" width="40px" /></a>
<a href="https://www.tensorflow.org/"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" height="40px" width="40px" /></a>

- Python 3.12.8
- TensorFlow 2.16.2
- Keras 3.4.1

---

## 🚀 Getting Started

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Andrew-XQY/XFlow.git
cd XFlow
pip install -e .
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 