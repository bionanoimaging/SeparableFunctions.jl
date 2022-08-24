# SeparableFunctions.jl

Here you can find the docstrings of currently implemented functions.

```@docs
calc_radial_symm!
copy_corners!
separable_view
separable_create
```

## SeparableFunctions Interface

The abstract `SeparableFunctions` definition
```@docs
SeparableFunctions
```

# Concrete separable examples
Functions that are separable.

<!-- ```@docs
gaussian_col
gaussian_lz
gaussian_sep
normal_col
normal_lz
normal_sep
ramp_col
ramp_lz
ramp_sep
rr2_col
rr2_lz
rr2_sep
box_col
box_lz
box_sep
sinc_col
sinc_lz
sinc_sep
exp_ikx_col
exp_ikx_lz
exp_ikx_sep
``` -->

# Radial functions
These functions are purely radial, in which case only a quadrant is computed and then copied.
They are based on a separable view version of `rr2` but then continue to calculate on a grid.
```@docs
calc_radial_symm!
calc_radial_symm
propagator_col
propagator_col!
```
