# Concrete SeparableFunctions
They all possess a similar interface and exist in three variants:
+ (recommended:) `function_sep`:   returns a broadcasted object, which is an operator combined with a calculated iterable of one-dimensional vectors (created using `calculate_separables`). This looks a lot like an array in terms of usage, yet it consumes far less memory and saves time when used in combination with other arrays. It also works with `CuArray` datatypes (provide the Array type as a first argument). Note that reduce operations about specific dimensions do not work directly on these objects but (currently) you need to call `collect` on it before applying for example`sum(collect(my_sep_object), dims=2)`.
+ `function_col`:   collects data. This leads to an allocation of the result array, but is usually still faster then calculating the N-dimensional function directly at each position. Internally, it calls the `function_sep` mechanism described below, which is for most applications preferable.
+ `function_lz`:    a lazy version based on `LazyArrays`. This only allocates the one-dimensional memory for the values along each dimension. A lazy array can even avoid allocation if for example a sum over the array is performed. The array can also be used in broadcasting operations. However, currently it does not work using `CUDA.jl`-style `CuArray` objects. This version is not recommended and not exported by the module.

```@docs
gaussian_sep
gaussian_col
normal_sep
normal_col
ramp_sep
ramp_col
rr2_sep
rr2_col
box_sep
box_col
sinc_sep
sinc_col
exp_ikx_sep
exp_ikx_col
```

