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
sz = (2000,2000)
sz = (200,200)
@btime p = propagator(Float32, $sz); # 0.14 sec / 1.2 ms
@btime myrr2 = rr2_sep($sz); # 3 µs  / 2 µs
myrr2 = rr2_sep(sz);
@btime @views myr = .*($myrr2...); # 1.9 ms(14 Mb) / 8.7 µs 
myr = .*(myrr2...); # to have one
f1(x) =  exp(1f0im * sqrt(max(0f0, 1f6 - x)) * 1f0)
# f2(x) = x # just for checking
c = zeros(ComplexF32, sz);
@btime @views c .= f1.($myr);  # 0.022 sec (0 Mb) / 354 µs
@btime calc_radial_symm!($c, $f1); # 0.018 (0 Mb) / 166 µs
@btime calc_radial_symm(Array{ComplexF32}, $sz, $f1); # 0.022 (45 Mb) / 172 µs
c = calc_radial_symm(Array{ComplexF32}, sz, f1); 

@btime $p = propagator(Float32, $sz); # 0.134 / 1.27 ms
@btime $d = propagator_col($sz); # 0.022 s/ 0.15 ms

p = propagator(Float32, sz); 
d = propagator_col(sz); 
@btime $d = propagator_col!($d); # 0.017 s/ 0.15 ms
d = propagator_col(CuArray{ComplexF32}, sz); 
@btime $d = propagator_col!($d); # 0.017 s/ 0.15 ms


@vtp p d
@btime @views c .= f1.(.+($myrr2...));  # 0.049 sec (0 Mb) / 512 µs
myrr2lz = rr2_lz(sz);
@btime c .= f1.($myrr2lz);  # 0.085 sec (0 Mb) / 800 µs
@btime c .= $myr;  # 3 ms / 4.3 µs 


sz = (2000,2000)
a = rand(ComplexF32, sz...);
b = rand(ComplexF32, sz...);
@time copy_corners!(a); # 0.003 vs 0.02
myrr2b = rr2_sep(sz);
#myrr2b = rr2_lz(sz);
# @time calc_radial_symm!(a, myrr2b, f1); # 0.009
@time calc_radial_symm!(a, f1), myrr2b; # 0.020 (0 Mb)
@btime calc_radial_symm!($a, $f1, $myrr2b); # 0.020 (0 Mb)
@btime @views b .= transpose($a); # 0.009 ms  Too expensive?
@btime @views b .= $a .* $a; # 0.002 ms  Too expensive?
@vp a

# @time p2 = exp.(1f0im .* sqrt.(max.(0.0, 100.0 .- (.+(myrr2...)))) .* 1.0);  # 0.04 sec . Double precision
a = rand(sz...);
@time b = a.+0; # 0.007
@time b = a.*a; # 0.006
@time b = a.^2; # 0.008
@time b = sqrt.(a); # 0.009
@time b = exp.(a); # 0.019
@time b = cis.(a); # 0.036

@time c = collect(exp.(1f0im .* sqrt.(max.(0f0, 100.0f0 .- (i*i+j*j))) .* 1f0) for i=1:sz[1], j=1:sz[2]); # 0.11 sec
@time c = map((x) -> exp.(1f0im .* sqrt.(max.(0f0, 100.0f0 .- x)) .* 1f0), .+(myrr2...)); # 0.06

