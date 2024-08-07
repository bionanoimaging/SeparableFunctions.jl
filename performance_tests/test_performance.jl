using SeparableFunctions
using BenchmarkTools

sz = (2000,1900)
sigma = (150.4,220.3)
# Create some functions by hand
# the first arguments are always r: (evaluation position) and sz: (size)
fct = (xyz, sz, pos, sigma)-> exp(-(xyz-pos)^2/(2*sigma^2))
@time my_gaussian = separable_view(fct, sz, (0.1,0.2), sigma); # 0.0005
@btime q = collect($my_gaussian); # 2.25 ms
@btime w = separable_create($fct, $sz, (0.1,0.2), $sigma); # 2.5ms  # not typestable!

# Test some predefined functions
@btime my_gaussian = gaussian_col(Array{Float64}, $sz); # 5 ms
@btime my_gaussian = SeparableFunctions.gaussian_lz(Array{Float64}, $sz; sigma=$sigma); # 70µs, not typestable!
@btime my_gaussian = SeparableFunctions.gaussian_lz($sz; sigma=$sigma); # 71µs, not typestable!
@btime q = collect($my_gaussian); # 2.2 ms

@btime my_normal = normal_col(Array{Float64}, $sz); # 4.8 ms
@btime my_normal = SeparableFunctions.normal_lz(Array{Float64}, $sz; sigma=$sigma); # 69 µs, not typestable!
@btime my_normal = SeparableFunctions.normal_lz($sz; sigma=$sigma); # 28 µs, not typestable!
my_normal = SeparableFunctions.normal_lz(sz; sigma=sigma);
@btime q2 = collect($my_normal); # 2.3 ms
@btime sum($my_normal) # 6.7 ms. Does NOT allocate! But is not as accurate.
my_sep = normal_sep(sz; sigma=sigma);  # not typestable!
@btime sum($my_sep) # 5 ms

fct_idx = (r, sz)-> r
w = separable_create(fct_idx, (10,10)) # 2.5ms  # not typestable!


# compare to using CartesianIndices
fct = (r, pos, sigma)-> exp(-(r-pos)^2/(2*sigma^2))
@btime q = collect((prod(fct.(Tuple(c), (0.0,0.0), $sigma)) for c in CartesianIndices($sz))); # takes 0.44 sec!

# compare to using IndexFunArrays
using IndexFunArrays
@btime g = collect(gaussian($sz, sigma=$sigma)); # 50 ms
@btime gs = gaussian_col($sz, sigma=$sigma); # 2 ms
# @vt g gs

############## complex exponentials
sz = (2000,1900)
Δx = (1.1, 2.2) # ./ (pi .* sz)
es = ones(ComplexF32, sz)
as = ones(ComplexF32, sz)
# from IndexFunArrays:
@btime e = collect(exp_ikx($sz, shift_by=$Δx)); # 111 ms
e = collect(exp_ikx(sz, shift_by=Δx));
@btime $e .*= exp_ikx($sz, shift_by=$Δx); # 105 ms
# @vp e
es = e;
@btime $es .*= exp_ikx_col($sz; shift_by=$Δx); # 10 ms
@btime $es .*= SeparableFunctions.exp_ikx_lz($sz; shift_by=$Δx); # 8 ms
@btime tmp = exp_ikx_sep($sz, shift_by=$Δx); # 47 µs  # not typestable
@btime $es .*= $tmp; # 6ms # seems to be the best and also works in Cuda
@btime $es .*= exp_ikx_sep($sz, shift_by=$Δx); # 6.6µs

# @time b = Broadcast.instantiate(Broadcast.broadcasted(*, tmp...))
# @time as .*= b; # 0.006 # seems to be the best and also works in Cuda

@btime SeparableFunctions.mul_exp_ikx!($es; shift_by=$Δx); # 10ms , is typstable (1 yellow union)

# @vtp e es
##### rr2 
@btime r = collect(rr2($sz)); # 6 ms
@btime rs = rr2_col($sz, scale=1.0); # 2 ms
@btime rs .= SeparableFunctions.rr2_lz($sz, scale=(1.0,1.0)); # 380 µs , not typestable
@btime rs .= rr2_sep($sz, scale=(1.0,1.0)); # 360 µs , not typestable
# @vt r rs

@btime b = collect(box($sz)); # 9 ms
@btime bs = box_col($sz, boxsize=$sz./2); # 2 ms
bs = box_col(sz);
@btime bs .= SeparableFunctions.box_lz($sz); # 100µs  (!!!)
#fct = (x, pos, d) -> abs(x -pos) < 500 
#fct = (x) -> abs(x) < 500 
bb = box_sep(sz, boxsize=sz./2); 
@btime bs .= $bb; # 95 µs

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
    sz = (2000,1900)
    sigma = (150.4,220.3)
    # CUDA.@time c_my_gaussian = gaussian_col(CuArray{Float32}, sz, sigma=sigma); #3 ms (GPU) vs 5 ms (CPU) 
    @btime CUDA.@sync c_my_gaussian = gaussian_col(CuArray{Float32}, $sz, sigma=$sigma); # 0.2 ms (GPU) vs 5 ms (CPU)
    # CUDA.@time c_sep = gaussian_sep(CuArray{Float32}, sz, sigma=sigma); # 1ms
    @btime CUDA.@sync c_my_gaussian = gaussian_sep(CuArray{Float32}, $sz, sigma=$sigma); # 0.2 ms (GPU) vs 5 ms (CPU)
    # CUDA.@time c_my_gaussian2 = c_sep  # 2 ms (GPU) vs 2 ms (CPU) # apply the separable collection
    CUDA.@time c_my_gaussian .= c_sep;  # 2.5 ms (GPU) vs 1 ms (CPU) # apply the separable collection in place. No allocation
    @btime CUDA.@sync $c_my_gaussian .= $c_sep;  # 0.1 ms (GPU) vs 1 ms (CPU) # apply the separable collection in place. No allocation

    CUDA.@time d = propagator_col(CuArray{ComplexF32}, sz); # 0.011 s
    @btime CUDA.@sync $d .= propagator_col(CuArray{ComplexF32}, $sz); # 0.6 ms
    d = propagator_col(CuArray{ComplexF32}, sz); 
    CUDA.@time d = propagator_col!(d); # 0.01 (no allocation)
    @btime CUDA.@sync $d = propagator_col!($d); # 0.6 ms
end

# How is the performance for a propagator, which is only partially separable?
# Testing the radially separable speedups methods:
sz = (2000,2000)
# sz = (200,200)
@btime p = propagator(Float32, $sz); # 101 ms / 1.2 ms
@btime myrr2 = rr2_sep($sz); # 3 µs  / 2 µs
myrr2 = rr2_sep(sz);
# @btime @views myr = .*($myrr2...); # 1.9 ms(14 Mb) / 8.7 µs 
@btime myr = collect($myrr2) # 7.5 ms

rng_start = .-(sz.÷2)
rng_stop = rng_start .+ sz .-1
rng = Tuple(sta:sto for (sta,sto) in zip(rng_start, rng_stop))
ci = CartesianIndices(rng);
# qq2(index) = sum(abs2.(Tuple(index))); # no difference

qq2(index) = sum(Tuple(index).^2);
f1(x) =  exp(1f0im * sqrt(max(0f0, 1f6 - x)) * 1f0)
# f2(x) = x # just for checking
c = zeros(ComplexF32, sz);

@btime @views $c .= f1.($myr);  # 55 ms (0 Mb) / 354 µs
@btime calc_radial_symm!($c, $f1); # 22 ms (20 KiB) / 166 µs
@btime propagator_col!($c); # 17 ms s (20 KiB)/ 0.15 ms
@btime @views $c .= f1.(qq2.($ci));  # 0.104 sec (48 bytes) 
@btime @views $c .= f1.(.+($myrr2...));  # 0.051 ms (128 bytes)
ci_sep = [CartesianIndices((rng[1],0:0)), CartesianIndices((0:0, rng[2]))]
@btime @views $c .= f1.(qq2.($ci_sep[1]) .+ qq2.($ci_sep[2]));  # 0.078 sec (0 bytes) 

@btime calc_radial_symm(Array{ComplexF32}, $sz, $f1); # 0.019 (30 Mb) / 172 µs
c = calc_radial_symm(Array{ComplexF32}, sz, f1); 

p = propagator(Float32, sz); 
@btime $p = propagator(Float32, $sz); # 0.134 / 1.27 ms
d = propagator_col(sz); 
@btime $d = propagator_col($sz); # 0.016 s/ 0.15 ms
@btime $d = propagator_col!($d); # 0.0116 s (28 KiB)/ 0.15 ms

rr2_sep(CuArray{Float32}, (4,4,4))

a = rand(100,100,100);
@btime @inbounds @views $a[:,:,100:-1:60] .= $a[:,:,10:50]; # 0.836ms
@btime @inbounds @views $a[100*100*100:-1:60*100*100] .= $a[10*100*100:50*100*100]; # 0.126ms

a = rand(2000,2000);
@btime @inbounds @views $a[:,2000:-1:1100] .= $a[:,1:901]; # 3.25 ms
@btime @inbounds @views $a[2000*2000:-1:1100*2000] .= $a[1*2000:2000*901]; # 1 ms
@btime copy_corners!($a); # 7 ms vs. 1.859 ms

# lets test if a sincos(phi) is maybe faster than exp.(i phi)
sz = (123,1,130,4)
sz = (2230,1,2300,4)
a = Float32.((100.0-50.0) .* 2pi .* rand(sz...));
@btime @views $a[:, 1,:,1:2] .= $a[:, 1,:,3:4]; # 13.6 µs
@btime @views $a[:, 1:1,:,1:2] .= $a[:, 1:1,:,3:4]; # 13.8 µs
@btime @views $a[:, :,:,1:2] .= $a[:, :,:,3:4]; # 7.5 µs
b = reshape(a,(size(a,1)*size(a,2)*size(a,3),4));
@btime @views $b[:,1:2] .= $b[:,3:4]; # 3.2 ms / 17.8 µs

sz=(2000,2000)
k_max_rel = 0.5
xy_scale = 2 ./ (k_max_rel .* sz[1:2])
p = collect(phase_kz(sz, scale=xy_scale));
p2 = phase_kz_col(sz, scale=xy_scale);
maximum(abs.(p .- p2)) ./ maximum(abs.(p))
@btime p = collect(phase_kz($sz, scale=xy_scale)); # 12 ms , offset=(10.0 20.0)
@btime p2 = phase_kz_col($sz, scale=xy_scale); # 3.4 ms , wraps
# r2 = rr2_sep(sz, scale=xy_scale, offset=(10, 20)) # , wraps
@vt p p2 p.-p2

# compare with rr2 and 
sz=(1000,1000)
qq2(index) = sum(Tuple(index).^2);
rng_start = .-(sz.÷2)
rng_stop = rng_start .+ sz .-1
rng = Tuple(sta:sto for (sta,sto) in zip(rng_start, rng_stop))

y = zeros(sz);

@info "Compare Cartesian with rr2 and rr2_sep"
# 1.747, 0 bytes
@btime $y .= ($qq2).(CartesianIndices($rng)) .+ sqrt.(1.2.*($qq2).(CartesianIndices($rng)).*($qq2).(CartesianIndices($rng)));

@btime $y .= rr2($x) .+ sqrt.(1.2.*rr2($x).*rr2($x)); # 1.96 ms, 1.64 kiB
r2_sep = rr2_sep(sz)
@btime $y .= .+($r2_sep...) .+ sqrt.(1.2.* .+($r2_sep...) .* .+($r2_sep...)); # 2.45 ms, 576 bytes

y[:,1] .= r2_sep[1] .+ sqrt.(1.2.* r2_sep[1] .* r2_sep[1]); # 2.45 ms, 576 bytes
@btime for n=1:1000 
    $y[:,n] .= $r2_sep[1] .+ sqrt.(1.2.* $r2_sep[1] .* $r2_sep[1]) 
end; # 2.38 ms (336 KiB)

@btime $y .= ($qq2).(CartesianIndices($rng)); # 159 µs
@btime $y .= rr2(sz); # 940 µs
r2_sep = rr2_sep(sz)
@btime $y .= .+(r2_sep...); # 172 µs

@info "rr2 based"
y .= rr2(sz) .+ sqrt.(1.2.*rr2(sz).*rr2(sz));
@info "CartesianIndices based"
y .= (qq2).(CartesianIndices(rng)) .+ sqrt.(1.2.*(qq2).(CartesianIndices(rng)).*(qq2).(CartesianIndices(rng)));



@btime $y[:] .= .+($r2_sep...)[:] .+ sqrt.(1.2.* .+($r2_sep...)[:] .* .+($r2_sep...)[:]); # 5.08 ms, 576 bytes
r2 = .+(r2_sep...); 
@btime $y .= $r2 .+ sqrt.(1.2.* $r2 .* $r2); # 1.314 ms, 0 bytes
r2_lz = SeparableFunctions.rr2_lz(sz);
@btime $y .= r2_lz .+ sqrt.(1.2.* r2_lz .* r2_lz); # 22.537 ms 384 bytes


q = myexp.(a); # 60 ms / 49.7 ms (in place)
w = exp.(1im .* a); # 77 ms / 70 ms
z = cis.(a); # 58 ms / 51.92
maximum(abs.(q.-w))
maximum(abs.(q.-z))

@btime $r .= exp.(1im .* $a); # 77 ms / 70 ms / 63 ms
b = a .+ 0.1f0im;
@btime $r .= cis.($b); # 58 ms / 51.92 / 46 ms / 61 (complex input)

@vtp p d
@btime @views c .= f1.(.+($myrr2...));  # 0.049 sec (0 Mb) / 512 µs
myrr2lz = SeparableFunctions.rr2_lz(sz);
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

################
#### some gradient tests
using SeparableFunctions
using Zygote 
using IndexFunArrays

# sz = (1000, 1000)
sz = (64, 64)
dat = rand(Float32, sz...)
myxx = xx((sz[1],1)) .+0
myyy = yy((1, sz[2])) .+0

my_gaussian(off, sigma) = exp.(.-abs2.((myxx .- off[1])./sigma[1]) .- abs2.((myyy .- off[2])./sigma[2]))
loss = (off, sigma) -> sum(abs2.(my_gaussian(off, sigma) .- dat))

loss = (off, sigma) -> sum(abs2.(gaussian_nokw_sep(sz, off, 1.0f0, sigma) .- dat))
# mystart = (1.1f0,2.2f0)
mystart = [1.1f0,2.2f0]
mysigma = [2.0f0, 3.0f0]
loss(mystart, mysigma)

g = gradient(loss, mystart, mysigma); 

# gradient(gaussian_raw, 1.0, 2.0, 3.0)

@time g = gradient(loss, mystart, mysigma); 
@btime g = gradient($loss, $mystart, $mysigma); 
# 0.023 s full: 69 Mb (129 alloc, 1000x1000), benchmark: 18.853 ms (129 allocations: 68.79 MiB)
# old: 5.493 ms (309 allocations: 26.78 MiB)
# new: 5.412 ms (232 allocations: 26.78 MiB)
