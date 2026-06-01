# Contributing

Thanks for contributing to efficientloftr-onnx-rs.

## Local setup

1. Install Rust stable via rustup.
2. Clone the repository.
3. Run:

```bash
cargo build
```

Optional helper:

```bash
just check
```

## Quality gates

Before opening a pull request, run:

```bash
just ci
```

Equivalent manual commands:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

## Pull request expectations

- Keep changes focused and small when possible.
- Add or update tests for behavior changes.
- Update README documentation for CLI or API changes.
- Ensure CI passes.
