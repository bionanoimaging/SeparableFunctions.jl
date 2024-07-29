using SeparableFunctions
using BenchmarkTools
using ComponentArrays
using Optim
using Random

sz = (64, 64)
sigma = (2.2, 3.3)

using IndexFunArrays
# Create some functions by hand
# the first arguments are always r: (evaluation position) and sz: (size)
fct = (xyz, sz, sigma)-> exp(-xyz^2/sqrt(2))

@time my_gaussian = separable_view(fct, sz, sigma); 
# @btime my_gaussian = separable_view($fct, $sz, $sigma); # Lazy Arrays: 4 µs
# @btime q = collect($my_gaussian); # 3.7 µs
w = separable_create(fct, sz, sigma)
# @btime w = separable_create($fct, $sz, $sigma); # 6.7 µs
# @btime q = collect($w); # 700 ns

# Test some predefined functions
@time my_gaussian = gaussian_col(sz; sigma = sigma); 
# @btime my_gaussian = gaussian_col($sz; sigma = $sigma); # 6 µs

##############
sz = (64,64)
mid = sz.÷2 .+1
myxx = Float32.(xx((sz[1],1)))
myyy = Float32.(yy((1, sz[2])))

my_full_gaussian(vec) = vec.bg .+ vec.intensity.*exp.(.-abs2.((myxx .- vec.off[1])./(sqrt(2f0)*vec.args[1])) .- abs2.((myyy .- vec.off[2])./(sqrt(2f0)*vec.args[2])))
# my_full_gaussian(vec) = vec.bg .+ vec.intensity.*exp.(.-abs2.(vec.sca[1].*(myxx .- vec.off[1])./(sqrt(2f0)*vec.sigma[1]))).*exp.(.- abs2.(vec.sca[2].*(myyy .- vec.off[2])./(sqrt(2f0)*vec.sigma[2])))

vec_true = ComponentVector(;bg=0.2f0, intensity=1.0f0, off = (2.2f0, 3.3f0), args = (2.4f0, 1.5f0))
# @btime my_full_gaussian($vec_true); # 42.800 μs (18 allocations: 16.70 KiB)

Random.seed!(42)
dat = 0.2f0 .* rand(Float32, sz...) .+ my_full_gaussian(vec_true)
dat_copy = copy(dat)

loss = (vec) -> sum(abs2.(my_full_gaussian(vec) .- dat))

bc_mem = gaussian_nokw_sep(sz, vec_true.off.+mid, 1f0, vec_true.args)
my_sep_gaussian(vec) = vec.bg .+ vec.intensity .* gaussian_nokw_sep(sz, vec.off .+mid, 1f0, vec.args; all_axes=bc_mem)
loss_sep = (vec) -> sum(abs2.(my_sep_gaussian(vec) .- dat))

# off_start = (2.0f0, 3.0f0)
# sigma_start = (3.0f0, 2.0f0)

off_start = [2.0f0, 3.0f0]
# off_start = [-32f0, -32f0]
sigma_start = [3.0f0, 2.0f0]
# sca_start = [1.2f0, 1.5f0]
startvals = ComponentVector(;bg = 0.5f0, intensity=1.0f0, off=off_start, args=sigma_start)

# @vt dat my_full_gaussian(startvals) my_sep_gaussian(startvals)

loss(startvals)
loss_sep(startvals)

using DifferentiationInterface
import ForwardDiff, Zygote, Tapir, ReverseDiff  # AD backends you want to use 
# import Enzyme # broken??

# @btime g = gradient($loss, $off_start, $sigma_start); # 64.600 μs (124 allocations: 157.73 KiB)
# @btime g = gradient($loss_sep, $off_start, $sigma_start); # 50.800 μs (374 allocations: 132.50 KiB)

# check Zygote directly:
using Zygote
g = Zygote.gradient(loss, startvals) #
@btime g = Zygote.gradient($loss, $startvals) # 85 µs, 214 kB
g = Zygote.gradient(loss_sep, startvals) # 
@btime g = Zygote.gradient($loss_sep, $startvals) # 59 µs, 409 allocs, 199 kB

v, g = value_and_gradient(loss, AutoForwardDiff(), startvals) # (54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))
v, g = value_and_gradient(loss, AutoZygote(),  startvals) #     (54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))
v, g = value_and_gradient(loss, AutoReverseDiff(),  startvals) #(54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))

v, g = value_and_gradient(loss_sep, AutoZygote(),  startvals) #     (54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))

# v, g = value_and_gradient(loss_sep, AutoReverseDiff(),  startvals) #(54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))

mymem = get_bc_mem(typeof(dat), size(dat), *)
@time fg!(dat, gaussian_raw, startvals.bg, startvals.intensity, startvals.off .+ sz.÷2 .+1, startvals.sigma; all_axes=mymem)

@btime fg!($dat, $gaussian_raw, $startvals.bg, $startvals.intensity, $startvals.off .+ $sz.÷2 .+1, $startvals.sigma; all_axes=$mymem)
# 35.3 µs, 165 allocs, 126 kB

# broken!
# value_and_gradient(loss, AutoTapir(),  startvals) #      
dat .= dat_copy
# value_and_gradient(loss, AutoEnzyme(),  startvals) # (off = Float32[0.0, 0.0], sigma = Float32[0.0, 0.0])

# value_and_gradient(loss_sep, AutoForwardDiff(), startvals) # (54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))
value_and_gradient(loss_sep, AutoZygote(),  startvals) #     (54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))
# value_and_gradient(loss_sep, AutoReverseDiff(),  startvals) #(54.162834f0, (off = Float32[-0.3461006, -1.4118625], sigma = Float32[-0.15418892, 0.4756395]))

# value_and_gradient(loss_sep, AutoTapir(),  startvals) #      

dat .= dat_copy
value_and_gradient(loss_sep, AutoEnzyme(),  startvals) # (54.16285f0, (off = Float32[-0.3461007, -1.4118639], sigma = Float32[-0.15418872, 0.47564003]))
dat .= dat_copy

# DifferentiationInterface.withgradient

# opt = Optim.Options(x_reltol=0.001)
opt = Optim.Options(iterations = 9) # why ?

@time res = Optim.optimize(loss, startvals, Optim.LBFGS(), opt; autodiff = :forward); # 
res.f_calls

@time res = Optim.optimize(loss, startvals, Optim.LBFGS(), opt); # 
res.f_calls

@btime res = Optim.optimize($loss, $startvals, Optim.LBFGS(), opt; autodiff = :forward);
# 1.06 ms, 11.13 ms
# Hand-Separated: 14 ms
# 10 ms

@btime res = Optim.optimize($loss, $startvals, Optim.LBFGS(), opt; autodiff = :finite);
# 4.065 ms (1989 allocations: 2.52 MiB), 40.440 ms (19519 allocations: 25.53 MiB)
# Hand-separated: 60.908 ms (20067 allocations: 24.28 MiB)
# 27 ms

function fg!(F, G, vec)
    val_pb = Zygote.pullback(loss, vec);
    # println("in fg!: F:$(!isnothing(F)) G:$(!isnothing(G))")
    if !isnothing(G)
        G .= val_pb[2](one(eltype(vec)))[1]
        # mutating calculations specific to g!
    end
    if !isnothing(F)
        # calculations specific to f
        return val_pb[1]
    end
end

myfg! = get_fg!(dat, gaussian_raw, loss=loss_anscombe_pos)

startvals_s = copy(startvals)
startvals_s.off .+= sz.÷2 .+1
G = copy(startvals)
myfg!(1, G, startvals_s)
G
od = OnceDifferentiable(Optim.NLSolversBase.only_fg!(fg!), startvals);
@time res = Optim.optimize(od, startvals,  Optim.LBFGS(), opt)
res.f_calls # 31
res.minimum # 13.927
# res.f_calls = 0

loss(startvals)
fg!(1.0, G, startvals)
G
od = OnceDifferentiable(Optim.NLSolversBase.only_fg!(fg!), startvals);
@btime res = Optim.optimize($od, $startvals, Optim.LBFGS(), $opt);
# Full: 4.804 ms (14435 allocations: 12.78 MiB)
# Sep: 4.099 ms (26269 allocations: 11.20 MiB)
# Hand-Separated: 2.397 ms (15472 allocations: 9.19 MiB)
# 2.5 ms

myfg!(1.0, G, startvals_s)
G
# opt = Optim.Options(iterations = 9); # why ?
odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), startvals_s);
@time reso = Optim.optimize(odo, startvals_s, Optim.LBFGS(), opt);
reso.f_calls # 31
reso.minimum # 13.927

odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), startvals_s);
@btime reso = Optim.optimize($odo, $startvals_s, Optim.LBFGS(), $opt);
# Zygote-free: 1.5 ms

using InverseModeling

gstartvals = ComponentVector(;offset = startvals.bg, i0=startvals.intensity, µ=startvals.off, σ=startvals.args)
@time res1, res2, res3 = gauss_fit(dat, gstartvals; iterations = 9);
res3.f_calls
res3.minimum
vec_true
res1

@btime res1, res2, res3 = gauss_fit($dat, $gstartvals; x_reltol=0.001);
# 4.37 ms (27575 allocations: 8.12 MiB)
@vt dat res2 (res2 .- dat)

# @btime Optim.optimize($loss, $off_start, $sigma_start, LBFGS(); autodiff = :forward); # 1.000 ms (10001 allocations: 1.53 MiB)
