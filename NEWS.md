# NEWS

This document came into existence as of version 0.6. It should document breaking changes
between versions, and discuss new features.

If you find a breaking change this is not reported here, please either raise an issue or
make a PR to ammend this document.

## 0.6.1

- Fixed performance bug in reverse-mode gradient computation for the `ELBO`, whereby an `O(N^3)` computation happened in cases where it shouldn't.

## 0.6.0

### Breaking changes

- Lower-case kernel constructors (`eq()`, `eq(l)`, etc) are deprecated in favour of directly constructing the types via their upper-case names (`EQ()` etc), and the convenience function `kernel(EQ(); l=0.1, s=1.1)`. The previous design was hard to maintain and somewhat opaque. The new design is self-consistent and entirely straightforward to maintain.

### New Features
- Documentation is significantly improved in the form of many additional docstrings and a couple of new pages of documentation in the docs.
