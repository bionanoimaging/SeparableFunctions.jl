# SeparableFunctions.jl
Calculates multidimensional functions faster by exploiting their separability.
Often a function involves an operation such as a (complex) exponential which by itself is computationally relatively heavy. Yet a number of multidimenstional functions are separable, which means that they can be written as a product (or sum) of single-dimensional functions. A good example of a separable function is a Gaussian function:

$G(x)=e^{\frac{\mathbf{r}}{2\sigma}}=e^{\frac{r_x}{2\sigma_x}} e^{\frac{r_y}{2\sigma_x}} e^{\frac{r_z}{2\sigma_z}}$

In this package, multidimensional functions are computed by first calculating their single-dimensional values and then creating the final multidimensional result by an outer product. Since multiplications and the broadcasting mechanism of Julia are fast compared to the evaluation of the function at each multidimensional position, the final result is calculated faster. The typical speedup can be an order of magnitude.

The package offers a general way of calculating separable functions as well as a `LazyArray` version of that function which can then be used inside other expressions.
The non-lazy version should currently also work with `CUDA.jl`, however the `LazyArray` version does not.

The package further offers a number of predifined separable function implementations such as `gaussian_sep` or `gaussian_sep_lz` for the separable version and the lazy separable version of a Gaussian. Another noteworthy example is the complex plain wave `exp_ikx_sep`.
