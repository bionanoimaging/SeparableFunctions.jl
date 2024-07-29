using SeparableFunctions
using ComponentArrays
using Optim
using BenchmarkTools
using Noise
using CUDA

# simulate a gaussian blob with Poisson noise and fit it with a Gaussian function
sz = (7,7) # (1600, 1600)
many_fits = true
use_cuda = true
N = 10_000
hyperplanes = many_fits ? rand(Float32, (1, N)) : 0

off = [3.2f0, 3.5f0].+hyperplanes
sigma = [1.4f0, 1.1f0]
hyperplanes = many_fits ? rand(Float32, (1, N)) : 0
intensity = [50f0] .* (1 .+ hyperplanes)
vec_true = ComponentVector(;bg=10.0f0, intensity=intensity, off = off, args = sigma)

pdat = gaussian_vec(sz, vec_true)
dat = Float32.(poisson(Float64.(pdat)))

pdat = (use_cuda) ? CuArray(pdat) : pdat
dat = (use_cuda) ? CuArray(dat) : dat

# now prepare the fitting:
myfg! = get_fg!(pdat, gaussian_raw, length(sz); loss=loss_anscombe_pos, bg=7f0);
shyperplanes = many_fits ? zeros(Float32, (1, size(dat)[end])) : 0
soff = [4.0f0, 4.0f0] .+ shyperplanes
bg = 0.5f0
intensity = [45f0] .+ shyperplanes
sigma = [3.0f0, 2.0f0]
if (use_cuda)
    bg = CuArray([bg])
    intensity = CuArray(intensity)
    soff = CuArray(soff)
    sigma = CuArray(sigma)
end
startvals = ComponentVector(;bg=bg, intensity=intensity, off = soff, args = sigma)
opt = Optim.Options(iterations = 499); #
odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), startvals);

# and perform the fit
@time reso = Optim.optimize(odo, startvals, Optim.LBFGS(), opt);
# 2 sec, 5k fits/s (44.25 k allocations: 1.546 GiB, 7.35% gc time)
# with intensity variations: 26.833106 seconds (532.47 k allocations: 20.251 GiB, 5.99% gc time)
# in Cuda: 
reso.f_calls # 61
reso.minimum 
@vt pdat dat gaussian_vec(sz, startvals) gaussian_vec(sz, reso.minimizer)

odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), startvals);
if isa(dat, CuArray)
    @time CUDA.@sync reso = Optim.optimize(odo, startvals, Optim.LBFGS(), opt);
    @btime CUDA.@sync reso = Optim.optimize($odo, $startvals, Optim.LBFGS(), $opt);
    # with intensity variations:  7.634 s (11810218 allocations: 289.64 MiB)
else
    @btime reso = Optim.optimize($odo, $startvals, Optim.LBFGS(), $opt);
end
# Zygote-free CPU: 800 µs,  for 1600x1600: 2.7 sec
# Zygote-free GPU: 52 ms, for 1600x1600: 0.213 sec

using InverseModeling

gstartvals = ComponentVector(;offset = startvals.bg, i0=startvals.intensity, µ=startvals.off.-sz.÷2 .+1, σ=startvals.args)
@time res1, res2, res3 = gauss_fit(dat, gstartvals; iterations = 99);
res3.f_calls
res3.minimum
vec_true
res1

@btime res1, res2, res3 = gauss_fit($dat, $gstartvals; x_reltol=0.001);
# 4.37 ms (27575 allocations: 8.12 MiB)
@vt dat res2 gaussian_vec(sz, reso.minimizer) (res2 .- dat)  (gaussian_vec(sz, reso.minimizer) .- dat)

@time res1, res2, res3 = gauss_fit(dat);

@btime res1, res2, res3 = gauss_fit($dat);
# 5 ms (39192 allocations: 4.35 MiB)

# @btime Optim.optimize($loss, $off_start, $sigma_start, LBFGS(); autodiff = :forward); # 1.000 ms (10001 allocations: 1.53 MiB)
