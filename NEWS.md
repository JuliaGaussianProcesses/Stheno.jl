# NEWS

This document came into existence as of version 0.6. It should document breaking changes
between versions, and discuss new features.

If you find a breaking change this is not reported here, please either raise an issue or
make a PR to ammend this document.

## 0.8.0

### Breaking Changes

This version contains some re-naming, specifically
- `WrappedGP` -> `AtomicGP`,
- `wrap` -> `atomic`, and
- `CompositeGP` -> `DerivedGP`,
which better reflect what these types / functions represent in the context of a `GPPP`.
It's possible that you've never interacted with them, in which case there's nothing to
worry about.

Lots of code has been moved around in order to better organise everything.

A method of `vcat` has been added for the concatentation of `GPPPInput`s to produce
`BlockData`, and `BlockData` is no longer exported. In short: use `vcat`, rather than
`BlockData` directly.

Deprecations mentioned in the 0.7 release have also been dropped.

## 0.7.16
- Deprecate `approx_posterior` in favour of `posterior`. This is being removed because it has been removed in AbstractGPs in favour of `posterior`. It will be entirely removed in the next breaking release.
- Remove some redundant testing infrastructure and tidy up the file structure slightly.

## 0.7.15
Enable WrappedGP to work with any AbstractGP, not just the GP type.

## 0.7.14

AbstractGPs now takes care of everything sparsity-related.
Consequently, Stheno no longer tests anything ELBO-related, and the functionality you get
will depend entirely upon which version of AbstractGPs you're using.

## 0.7.0

### Breaking changes

This version contains a large number of _very_ breaking changes -- some of the basics of the package have completely changed. The over-riding concern has been to make Stheno work well with the AbstractGPs API. Of particular note is the new `GPPP` / `GaussianProcessProbabilisticProgramme` type, which represents a collection of processes and treats them as a single `AbstractGP`.

- The `GaussianProcessProbabilisticProgramme` type has been introduced, and is the new recommended way to work with Stheno.jl. See its docstring, README, and the examples directory for details.
- The `@model` macro is gone, and has been replaced with the `@gppp` macro. This new macro has quite different functionality, and produces a `GPPP` object.
- Stheno's own kernels have been replaced by ones from `KernelFunctions.jl`. This has removed a lot of code from the repo, but completely changes the function calls required to build GPs. To upgrade, consult KernelFunctions.jl for equivalent kernels.
- Stheno's internal GP type has been replaced with a wrapper type `WrappedGP` for `AbstractGP`s.
- Stheno's internal conditioning and approximate-conditioning functionality has been entirely removed, in favour of using AbstractGPs `posterior` and `approx_posterior` directly on entire `GPPP`s. This much simpler approach generally makes for correspondingly simpler code.

### New Features

- Kernels defined using `KernelFunctions.jl` should all work now.
- The @gppp macro.
- New `GPPPInput` type for working with `GPPP`s.
- `BlockData` exported for working with multiple processes in a `GPPP` at once.

## 0.6.1

- Fixed performance bug in reverse-mode gradient computation for the `ELBO`, whereby an `O(N^3)` computation happened in cases where it shouldn't.

## 0.6.0

### Breaking changes

- Lower-case kernel constructors (`eq()`, `eq(l)`, etc) are deprecated in favour of directly constructing the types via their upper-case names (`EQ()` etc), and the convenience function `kernel(EQ(); l=0.1, s=1.1)`. The previous design was hard to maintain and somewhat opaque. The new design is self-consistent and entirely straightforward to maintain.

### New Features
- Documentation is significantly improved in the form of many additional docstrings and a couple of new pages of documentation in the docs.
