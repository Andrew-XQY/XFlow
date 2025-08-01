# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-commit hooks with black and isort for automatic code formatting
- Comprehensive Sphinx documentation system with GitHub Pages deployment
- API registry system for centralized module organization
- ImageReconstructionCallback for training visualization

### Changed
- Improved documentation structure with RTD theme and custom styling
- Enhanced CI/CD pipeline with automated documentation deployment

### Fixed
- Sphinx build warnings and duplicate object documentation

## [0.1.0] - 2025-08-01

### Added
- Initial release of XFlow framework
- Modular architecture with core abstractions
- Data pipeline management system
- Base model implementations
- Training utilities and callbacks
- TensorFlow/Keras integration
- Basic documentation structure

### Core Modules
- **Data Module**: Pipeline management, loaders, and preprocessing
- **Models Module**: Base model classes and implementations
- **Trainers Module**: Training loops, callbacks, and utilities
- **Utils Module**: Configuration management, logging, and I/O utilities

### Dependencies
- TensorFlow 2.16.2
- Keras 3.4.1
- NumPy, Matplotlib, OpenCV
- Pydantic for configuration management

---

## How to maintain this changelog

### When making changes:
1. Add new entries to the "Unreleased" section
2. Use the following categories:
   - **Added** for new features
   - **Changed** for changes in existing functionality
   - **Deprecated** for soon-to-be removed features
   - **Removed** for now removed features
   - **Fixed** for any bug fixes
   - **Security** for vulnerability fixes

### When releasing:
1. Move "Unreleased" changes to a new version section
2. Add the release date
3. Create a new empty "Unreleased" section
4. Update the version in `pyproject.toml`

### Example entry format:
```markdown
## [1.1.0] - 2025-08-15

### Added
- New feature X that does Y
- Support for Z format in data loader

### Fixed
- Bug where model training would crash on empty datasets
- Memory leak in data preprocessing pipeline
```
