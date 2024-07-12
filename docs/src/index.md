# SeparableFunctions.jl

## SeparableFunctions.jl Interface
Separable Functions are multidimensional functions that can be written as a broadcast (outer product) of functions of individual dimensions, e.g. `f(x,y) = f(x).*f(y)`. In a more general way, this also applies to additions `ramp(x,y, slope) = ramp(x, slope_x) .+ ramp(y,slope_y)`. This package provides an easy-to-use performant interfact to represent such functions. The performance gains can be quite massive, particulary if the single function evaluation is expensive. E.g. a complex-valued phase ramp `exp.(1im.*(k_x.*x .+ k_y.*y)` is a separable product, which allows the complex exponential to be calculated only for the one-dimensional vectors rather than an N-dimensional array. 

In this package you will find generic tools to represent your own separable function as well as a number of predefined functions such a `gaussian_sep`, `rr2_sep`, `ramp_sep`, `box_sep`, `sinc_sep`, `exp_ikx_sep`. There are corresponding conveniance `_col` versions for collected arrays, but it is recommended to directly use the `_sep` versions. There are also versions (`_lz`) based on `LanzyArrays.jl` but since they are currently not performing that well, they are currently not exported.
To generate your own versions of separable functions you can use `calculate_broadcasted`.

Additionally this package also provides a convenient interface (`calc_radial_symm`) to quickly calculate (`calc_radial_symm`, `calc_radial_symm!`) radial functions. A fast calculation is achieved by calculating the values only for a fraction of the N-dimensional array and using effective copy operations exploiting symmetry. Also here example implementations, which either collect `_col` or modify existing array `!_col`,are provided. Currently implemented examples are `propagator_col`, `phase_kz_col`.

In a more general sense:
```@docs
SeparableFunctions
```

## Generic Ways to Create Seperable and Radial Functions 
```@docs
calculate_broadcasted
calculate_separables
calculate_separables_nokw
separable_view
separable_create
copy_corners!
```
