# NeuralLive Package

## Overview

This package represents the modular refactoring of the `adaptiveengineer` codebase as outlined in the Debug_Transformation.md Phase 1 implementation.

## Purpose

The NeuralLive package aims to:
1. Break down the monolithic `adaptiveengineer.py` into modular, maintainable components
2. Improve code organization and discoverability
3. Enable better testing and type checking
4. Facilitate future development of advanced features (consciousness, security, etc.)

## Structure

```
neuralive/
├── __init__.py           # Package initialization with backward compatibility
├── core/                 # Core functionality (agents, environment, lifecycle)
├── utils/                # Utility functions and helpers
├── consciousness/        # (Future) Self-awareness and meta-cognition
├── intelligence/         # (Future) Advanced AI and learning
├── swarm/                # (Future) Swarm coordination and consensus
├── security/             # (Future) Security and self-healing
└── governance/           # (Future) Ethics and constitutional framework
```

## Migration Strategy

The migration from `adaptiveengineer.py` to the modular structure follows these principles:

1. **Backward Compatibility**: Existing imports continue to work
2. **Incremental Refactoring**: Extract modules one at a time
3. **Test-Driven**: Run tests after each extraction
4. **No Breaking Changes**: Maintain all existing APIs

## Current Status

**Phase 1: Foundation Refactoring - Week 3-4**

- [x] Package structure created
- [x] Backward compatibility imports established
- [ ] Core modules extraction (in progress)
- [ ] Full modularization (future work)

## Usage

For now, continue using the original imports:

```python
from adaptiveengineer import AliveLoopNode, Memory, SocialSignal
```

As modules are extracted, you'll be able to use:

```python
from neuralive import AliveLoopNode, Memory, SocialSignal
# Or more specific imports:
from neuralive.core import Agent, Environment
```

## Development

This package is under active development as part of the Phase 1 refactoring effort. See `PHASE1_ANALYSIS_REPORT.md` and `Debug_Transformation.md` for more details.

## Version

Current version: 2.0.0 (Phase 1)

## License

Same as the parent project (see LICENSE file).
