using SeparableFunctions

sz = (2000,1900)
sigma = (150.4,220.3)
# Create some functions by hand
fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@time my_gaussian = separable_view(fct, sz, (0.1,0.2), sigma);
@time q = collect(my_gaussian); # 0.004

@time w = create_separable(fct, sz, (0.1,0.2),sigma); # 0.004

# Test some predefined functions
@time my_gaussian = gaussian_col(Array{Float64}, sz); # sigma=sigma
@time my_gaussian = gaussian_lz(Array{Float64}, sz, sigma);
@time my_gaussian = gaussian_lz(sz, sigma);
@time q = collect(my_gaussian); # 0.002 sec

fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@time q = collect((prod(fct.(Tuple(c), (0.0,0.0), sigma)) for c in CartesianIndices(sz)));

using IndexFunArrays
@time g = collect(gaussian(sz, sigma=sigma)); # 0.05
@time gs = gaussian_col(sz, sigma); # 0.003
@vt g gs

Δx = (1.1, 2.2) # ./ (pi .* sz)
@time e = collect(exp_ikx(sz, shift_by=Δx)); # 0.19
@time e .= exp_ikx(sz, shift_by=Δx); # 0.10
# @vp e
@time es = exp_ikx_col(sz, Δx); # 0.006
@time es .= exp_ikx_lz(sz, Δx); # 0.003
# @vtp e es
@time r = collect(rr2(sz)); # 0.01
@time rs = rr2_col(sz, 1.0); # 0.003
@time rs .= rr2_lz(sz, (1.0,1.0)); # 0.0015
# @vt r rs
@btime b = collect(box($sz)); # 0.012
@btime bs = box_col($sz, $sz./2); # 0.0005
bs = box_col(sz, sz./2);
@btime bs .= box_lz($sz, $sz./2); # 0.025  (!!!)
fct = (x, pos, d) -> abs(x -pos) < 500 
fct = (x) -> abs(x) < 500 
bs = box_sep(sz, sz./2); 
@btime res = .*($bs...) # 0.0004

# @vt b bs

# MWE:
using BenchmarkTools
x = 1:2000 .< 1000 
y = (1:1900)' .< 1000 
a = x .* y;
@btime a .= $x .* $y; # 2.8 ms
la = LazyArray(@~ x .* y);
@btime a = collect($la); # 2.6 ms
@btime a .= $la; # 30.8 ms (Problem!)

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
