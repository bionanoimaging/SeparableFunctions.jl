using SeparableFunctions
using ComponentArrays
using Optim
using BenchmarkTools
using Noise
using CUDA

# simulate a gaussian blob with Poisson noise and fit it with a Gaussian function
sz = (1600, 1600)
vec_true = ComponentVector(;bg=10.0f0, intensity=50f0, off = [8.2f0, 6.5f0], args = [2.4f0, 1.5f0])

dat = Float32.(poisson(Float64.(gaussian_vec(sz, vec_true))))

# dat = CuArray(dat)
# now prepare the fitting:
myfg! = get_fg!(dat, gaussian_raw, loss=loss_anscombe_pos, bg=7f0);
startvals = ComponentVector(;bg=0.5f0, intensity=45f0, off = [9f0, 7f0], args = [3.0f0, 2.0f0])

opt = Optim.Options(iterations = 19); #
odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), startvals);

# and perform the fit
@time reso = Optim.optimize(odo, startvals, Optim.LBFGS(), opt);
reso.f_calls # 61
reso.minimum 
@vt dat gaussian_vec(sz, startvals) gaussian_vec(sz, reso.minimizer)

odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), startvals);
if isa(dat, CuArray)
    @btime CUDA.@sync reso = Optim.optimize($odo, $startvals, Optim.LBFGS(), $opt);
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
