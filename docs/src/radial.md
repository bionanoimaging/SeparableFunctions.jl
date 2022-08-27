# Radial functions
These functions are purely radial, in which case only a quadrant is computed and then copied.
They are all based on a separable view version of `rr2` but then continue to calculate on a corner
of a grid, which is (as a last step) replicated and mirrored.
```@docs
calc_radial_symm
calc_radial_symm!
propagator_col
propagator_col!
phase_kz_col
phase_kz_col!
```
