# grampc-s-rs

Rust-native building blocks inspired by [GRAMPC-S](https://github.com/grampc/grampc-s).

This project is intended to be a pure Rust implementation. It should not depend on
the upstream C++ codebase, CMake build files, MATLAB scripts, Python bindings, or
foreign-language solver wrappers. New functionality should be implemented with Rust
code and Rust crates.

This is not a direct binding to the external GRAMPC solver. It ports the solver-independent
stochastic MPC pieces first:

- moment-based distributions and sampling,
- unscented, Stirling first-order, and composed Gaussian quadrature transformations,
- Gaussian, Chebyshev, and symmetric chance-constraint tightening,
- squared-exponential Gaussian-process residual models,
- lightweight problem and RK4 simulator traits,
- a double-integrator example based on the upstream example shape.

Run the tests:

```sh
cargo test
```

Run the example:

```sh
cargo run --example double_integrator
```

## Porting Notes

The upstream GRAMPC-S project also contains GRAMPC solver integration, Python/MATLAB bindings,
additional distributions, polynomial-chaos expansion, and more GP kernels. This crate is structured
so those layers can be added incrementally in Rust without changing the public foundations.

Rust-only porting priorities:

1. Add the remaining distributions and random sampling APIs.
2. Add polynomial types and polynomial-chaos expansion.
3. Add the missing point transformations, including Monte Carlo and Stirling second order.
4. Add the remaining Gaussian-process kernels and kernel composition.
5. Add Rust-native deterministic optimal-control/SMPC solver integration.
6. Port the upstream examples as Rust examples.
