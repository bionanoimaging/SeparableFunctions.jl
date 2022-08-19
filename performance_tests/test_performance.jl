using SeparableFunctions

sz = (2000,1900)
sigma = (150.4,220.3)
# Create some functions by hand
fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@time my_gaussian = separable_view(fct, sz, (0.1,0.2), sigma);
@time q = collect(my_gaussian); # 0.004

@time w = create_separable(fct, sz, (0.1,0.2),sigma); # 0.004

# Test some predefined functions
@time my_gaussian = gaussian_sep_lz(Array{Float64}, sz, sigma);
@time my_gaussian = gaussian_sep_lz(sz, sigma);
@time q = collect(my_gaussian); # 0.002 sec

fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@time q = collect((prod(fct.(Tuple(c), (0.0,0.0), sigma)) for c in CartesianIndices(sz)));

using IndexFunArrays
@time q = collect(gaussian(sz, sigma=sigma));
@time e = collect(exp_ikx(sz, scale=(1.1,2.2))); # 0.19
@time e = exp_ikx_sep(sz, (1.1,2.2)); # 0.006

# @vt q w

# using CUDA
# CUDA.@time c_my_gaussian = separable_view(CuArray{Float32}, fct, sz, (0.1,0.2), sigma);
# CUDA.@time cq = collect(c_my_gaussian); # 0.004

# @time w = create_separable(CuArray{Float32}, fct, sz, (0.1,0.2), sigma); # 0.004

# ca = CuArray([1 2 3 4 5]);
# cb = CuArray([6 7 8 9 10]);
# cc = CuArray([17 17 17 17 17]);
# cq = LazyArray(@~(ca .* cb));
# cc .= cq;
# w = collect(cq); # error!

# a = [1 2 3 4 5];
# b = [6 7 8 9 10];
# q = LazyArray(@~(a .* b));
# c .= q

# x = CuArray{Float32}(undef, 10)
# x .= (i for i in 1:10) .* 
# CuArray((i for i in))
