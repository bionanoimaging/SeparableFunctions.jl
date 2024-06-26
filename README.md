# SeparableFunctions.jl

| **Documentation**                       | **Build Status**                          | **Code Coverage**               |
|:---------------------------------------:|:-----------------------------------------:|:-------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] | [![][codecov-img]][codecov-url] |


Calculates multidimensional functions faster by exploiting their separability.
Often a function involves an operation such as a (complex) exponential which by itself is computationally relatively heavy. Yet a number of multidimenstional functions are separable, which means that they can be written as a product (or sum) of single-dimensional functions. A good example of a separable function is a Gaussian function:

$G(x)=e^{\frac{\mathbf{r}-\mathbf{r_0}}{\sqrt{2}\sigma}}=e^{-(\frac{r_x -x_0}{\sqrt{2}\sigma_x})^2} e^{-(\frac{r_y-y_0}{\sqrt{2}\sigma_x})^2} e^{-(\frac{r_z-z_0}{\sqrt{2}\sigma_z})^2}$

In this package, multidimensional functions are computed by first calculating their single-dimensional values and then creating the final multidimensional result by an outer product. Since multiplications and the broadcasting mechanism of Julia are fast compared to the evaluation of the function at each multidimensional position, the final result is calculated faster. The typical speedup can be an order of magnitude.

The package offers a general way of calculating separable functions as well as a `LazyArrays` version of that function which can then be used inside other expressions. Yet, these lazy versions are currently NOT recommended, since they are consistently slower that the separabel implementations. This is why the specific versions are currently also not exported.

The non-lazy version should currently also work with `CUDA.jl`, however the `LazyArrays` version does not. To nevertheless use separable expressions in `CUDA.jl`, you can reside to externally applying the broadcast operator to the separable expression (see the `gaussian_sep` example below).

The package further offers a number of predifined separable function implementations such as `gaussian_col()` collects a multidimensional array of a multidimensional Gaussian via a fast seperable implementation, `gaussian_lz()` yields a lazy representation via `LazyArrays` and `sep = gaussian_sep()` yields a `broadcasted` type, which behaves like an array, if used in an expression, but hast not yet been expanded to a full-sized array (recommended mode of using the package).

Another noteworthy example is the complex plain wave as represented by the respective function `exp_ikx_col()`, `exp_ikx_lz()`, `exp_ikx_sep()`.

All separable functions share a common interface with
Standard non-named arguments
+ the first optional argument being the type of the result array. Examples are `Array{Float64}`, `CuArray{Float32}` and the default depends on the function but uses a 32-bit result type where applicable.
+ the next argument is the `size` of the result data supplied as a tuple.
+ optionally a further argument can specify the `position` of zero of the array with respect to `offset` as given below. This allows for convenient N-dimensional shifting of the functions. 
Named arguments:
+ `offset`: is a optional named argument specifying center of the array. By default the Fourier center, `size().\div2 .+ 1` is chosen.
+ `scale`: multiplies the axes with these factors. This can be interpreted as a `pixelsize`.
Some functions have additional named arguments. E.g. `gaussian` has the additional named argument `sigma` specifying the width of the gaussian, even though this can also be controlled via `scale`. 
Note that this nomenclature is in large parts identical to the package `IndexFunArrays.jl`.

In general arguments can be supplied as single scalar values or vectors. If a scalar value is supplied, it will automatically be replicated as a vector. E.g. `sigma=10.0` for a  2-dimensional array will be interpreted as `sigma=(10.0, 10.0)`.


[docs-dev-img]: https://img.shields.io/badge/docs-dev-pink.svg
[docs-dev-url]: https://bionanoimaging.github.io/SeparableFunctions.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-darkgreen.svg
[docs-stable-url]: https://bionanoimaging.github.io/SeparableFunctions.jl/stable/

[CI-img]: https://github.com/bionanoimaging/SeparableFunctions.jl/actions/workflows/ci.yml/badge.svg
[CI-url]: https://github.com/bionanoimaging/SeparableFunctions.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/bionanoimaging/SeparableFunctions.jl/branch/main/graph/badge.svg?token=6XWI1M1MPB
[codecov-url]: https://codecov.io/gh/bionanoimaging/SeparableFunctions.jl