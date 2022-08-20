using SeparableFunctions

sz = (2000,1900)
sigma = (150.4,220.3)
# Create some functions by hand
# the first arguments are always r: (evaluation position) and sz: (size)
fct = (r, sz, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@time my_gaussian = separable_view(fct, sz, (0.1,0.2), sigma);
@time q = collect(my_gaussian); # 0.004
@time w = separable_create(fct, sz, (0.1,0.2),sigma); # 0.004

# Test some predefined functions
@time my_gaussian = gaussian_col(Array{Float64}, sz); # sigma=sigma
@time my_gaussian = gaussian_lz(Array{Float64}, sz; sigma=sigma);
@time my_gaussian = gaussian_lz(sz; sigma=sigma);
@time q = collect(my_gaussian); # 0.002 sec

@time my_normal = normal_col(Array{Float64}, sz); 
@time my_normal = normal_lz(Array{Float64}, sz; sigma=sigma);
@time my_normal = normal_lz(sz; sigma=sigma);
@time q2 = collect(my_normal); # 
@time sum(my_normal) # 0.0035. Does NOT allocate! But is not as accurate.
my_sep = normal_sep(sz; sigma=sigma);
@time sum(.*(my_sep...)) # Does allocate. But is more accurate

fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@time q = collect((prod(fct.(Tuple(c), (0.0,0.0), sigma)) for c in CartesianIndices(sz))); # takes 3 sec!

using IndexFunArrays
@time g = collect(gaussian(sz, sigma=sigma)); # 0.05
@time gs = gaussian_col(sz, sigma=sigma); # 0.003
# @vt g gs

Δx = (1.1, 2.2) # ./ (pi .* sz)
@time e = collect(exp_ikx(sz, shift_by=Δx)); # 0.19
@time e .= exp_ikx(sz, shift_by=Δx); # 0.10
# @vp e
@time es = exp_ikx_col(sz; shift_by=Δx); # 0.006
@time es .= exp_ikx_lz(sz; shift_by=Δx); # 0.003
# @vtp e es
@time r = collect(rr2(sz)); # 0.01
@time rs = rr2_col(sz, scale=1.0); # 0.003
@time rs .= rr2_lz(sz, scale=(1.0,1.0)); # 0.0015
# @vt r rs
using BenchmarkTools
@btime b = collect(box($sz)); # 0.012
@btime bs = box_col($sz, boxsize=$sz./2); # 0.0005
bs = box_col(sz);
@btime bs .= box_lz($sz); # 0.025  (!!!)
fct = (x, pos, d) -> abs(x -pos) < 500 
fct = (x) -> abs(x) < 500 
bs = box_sep(sz, boxsize=sz./2); 
@btime res = .*($bs...) # 0.0004

# @vt b bs

# MWE:
# using BenchmarkTools
# x = 1:2000 .< 1000 
# y = (1:1900)' .< 1000 
# a = x .* y;
# @btime a .= $x .* $y; # 2.8 ms
# la = LazyArray(@~ x .* y);
# @btime a = collect($la); # 2.6 ms
# @btime a .= $la; # 30.8 ms (Problem!)

if false
    using CUDA
    CUDA.@time c_my_gaussian = gaussian_col(CuArray{Float32}, sz, sigma=sigma) #3 ms (GPU) vs 5 ms (CPU) 
    CUDA.@time c_sep = gaussian_sep(CuArray{Float32}, sz, sigma=sigma) 
    CUDA.@time c_my_gaussian2 = .*(c_sep...)  # 2 ms (GPU) vs 2 ms (CPU) # apply the separable collection
    CUDA.@time c_my_gaussian2 .= .*(c_sep...)  # 2 ms (GPU) vs 1 ms (CPU) # apply the separable collection in place. No allocation
end

# How is the performance for a propagator, which is only partially separable?
@time p = propagator(Float32, sz) # 0.08 sec
myrr2 = rr2_sep(sz)
@time p2 = exp.(1f0im .* sqrt.(max.(0f0, 100.0f0 .- (.+(myrr2...)))) .* 1f0)  # 0.036 sec 
@time p2 = exp.(1f0im .* sqrt.(max.(0.0, 100.0 .- (.+(myrr2...)))) .* 1.0)  # 0.04 sec 
