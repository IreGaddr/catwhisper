# Contributing to CatWhisper

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/your-org/catwhisper.git
cd catwhisper
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCW_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

## Code Style

- C++20 features encouraged
- `std::expected` for error handling (no exceptions)
- RAII for all resources
- Meaningful variable names, no single-letter variables (except loop indices)
- Match existing code formatting

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`ctest`)
6. Commit with clear messages
7. Push and create a PR

## Commit Messages

```
type: brief description

Optional longer explanation.

Fixes #issue
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Code Review

All PRs require review. Be responsive to feedback.

## License

By contributing, you agree your contributions are dual-licensed under MIT and Apache 2.0.
