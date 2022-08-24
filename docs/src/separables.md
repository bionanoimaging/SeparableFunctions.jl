# Concrete SeparableFunctions
They all possess a similar interface and exist in three variants:
+ `function_col`:   collects data. This leads to an allocation of the result array, but is usually still faster then calculating the N-dimensional function directly at each position.
+ `function_lz`:    a lazy version based on `LazyArrays`. This only allocates the one-dimensional memory for the values along each dimension. A lazy array can even avoid allocation if for example a sum over the array is performed. The array can also be used in broadcasting operations. However, currently it does not work using `CUDA.jl`-style `CuArray` objects.
+ `function_sep`:   returns a calculated iterable of one-dimensional vectors. They can then by multiplied (or summed) together using `.*(res...)` or alike, if `res` is the result of the `function_sep` call. This has the advantage of working also with `CuArray` datatypes, but in contrast to the `function_lz` Version, during a sum the array will be collected first and then summed.

```@docs
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
```

