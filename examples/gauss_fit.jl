using SeparableFunctions
using ComponentArrays
using Optim
using BenchmarkTools
using Noise
using CUDA

# simulate a gaussian blob with Poisson noise and fit it with a Gaussian function
sz = (9, 9) # (1600, 1600)
many_off = true
many_int = true
many_bg = true
many_sig = true

use_cuda = false
DType = Float32
# N = 10_000
N = 1000
hp_off = many_off ? 2 .*rand(DType, (1, N)) : 0
hp_sig = many_sig ? zeros(DType, (1, N)) : 0
hp_int = many_int ? 1 .+ rand(DType, (1, N)) : 1

off = [5.2, 4.5] .+ hp_off
sigma = [1.4, 1.1] .+ hp_sig
intensity = [50] .* hp_int
bg = many_bg ? 10.2 .+ zeros(DType, (1, N)) : 10.2
vec_true = DType.(ComponentVector(;bg=bg, intensity=intensity, off = off, args = sigma))

# create the perfect spots:
pdat = gaussian_vec(sz, vec_true)
dat = DType.(poisson(Float64.(pdat)))

startvals = DType.(ComponentVector(gauss_start(dat, 0.2, length(sz))));
sum(abs2.(collect(startvals.off) .- vec_true.off))

pdat = (use_cuda) ? CuArray(pdat) : pdat
dat = (use_cuda) ? CuArray(dat) : dat
if (use_cuda)
    startvals = ComponentVector(;bg=CuArray(startvals.bg), intensity=CuArray(startvals.intensity), off = CuArray(startvals.off), args = CuArray(startvals.args))
end

# @vt pdat gaussian_vec(sz, startvals)
# qdat = copy(pdat)
# qdat .= qdat[:,:,1]
# now prepare the fitting:
# myfg! = get_fg!(dat, gaussian_raw, length(sz); loss=loss_anscombe_pos, bg=0.1f0);
# myfg! = get_fg!(dat, gaussian_raw, length(sz); loss=loss_gaussian);
myfg! = get_fg!(dat, gaussian_raw, length(sz); loss=loss_poisson_pos);
# hp_off2 = many_off ? zeros(DType, (1, size(dat)[end])) : 0
# soff = [4.0, 4.0] .+ hp_off2
# bg = [0.5] .+ shyperplanes
# intensity = [45.0] .+ hp_int
# sigma = [3.0, 2.0] .+ hp_sig
# if (use_cuda)
#     bg = CuArray([bg])
#     intensity = CuArray(intensity)
#     soff = CuArray(soff)
#     sigma = CuArray(sigma)
# end
# startvals = DType.(ComponentVector(;bg=bg, intensity=intensity, off = soff, args = sigma))
opt = Optim.Options(iterations = 150); #

if (false)
    G = copy(startvals)
    myfg!(1, G, startvals)

    myfg2! = get_fg!(pdat[:,:,1], gaussian_raw, length(sz); loss=loss_anscombe_pos, bg=0.1f0);
    sv = ComponentVector{Float32}(bg=startvals.bg[1], intensity=startvals.intensity[1], off = startvals.off[:,1], args = startvals.args[:,1])
    G2 = copy(sv)
    myfg2!(1, G2, sv)
    G2
end

# and perform the fit
svb = copy(startvals)
# svb.args = svb.args .* 1.2f0
odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!), svb);
@time reso = Optim.optimize(odo, svb, Optim.LBFGS(), opt);
# 14 sec, CUDA: 2.5 sec
# 2 sec, 5k fits/s (44.25 k allocations: 1.546 GiB, 7.35% gc time)
# with intensity variations: 26.833106 seconds (532.47 k allocations: 20.251 GiB, 5.99% gc time)
# in Cuda: 
reso.f_calls # 61   # 1766 für 10_000 fits, 155, 2.2 sec for 10_000 fits with all entries being vectors
reso.minimum # 
@vt pdat dat (gaussian_vec(sz, startvals).-dat) (gaussian_vec(sz, reso.minimizer).-dat)

success = sum(abs.(collect(startvals.off) .- vec_true.off), dims=1) .< 0.5
success = success .&& sum(abs.(collect(reso.minimizer.off) .- vec_true.off), dims=1) .< 0.5
sum(.!success)

# Try the same by calling optimizations in parallel (seems to work better!):
if false
    szz = size(dat)[end]
    # convert the startvals into a vector of individual startvals
    dat_a = Array(dat)
    svv = [DType.(ComponentVector(gauss_start(dat_a[:,:,n:n], 0.2, length(sz))))  for n in 1:szz];
    if (use_cuda)
        svv = [ComponentVector(;bg=CuArray(svv[n].bg), intensity=CuArray(svv[n].intensity),
            off = CuArray(svv[n].off), args = CuArray(svv[n].args)) for n in 1:szz]
    end
    # svb.args = svb.args .* 1.2f0
    myfg!v = [get_fg!(dat[:,:,n:n], gaussian_raw, length(sz); loss=loss_poisson_pos) for n in 1:szz];

    odov = [OnceDifferentiable(Optim.NLSolversBase.only_fg!(myfg!v[n]), svv[n]) for n in 1:szz];
    @time resov = Optim.optimize.(odov, svv, Ref(Optim.LBFGS()), Ref(opt));
    # seeems faster!
    svvoff = cat([svv[n].off for n in 1:szz]..., dims=2)
    resoff = cat([resov[n].minimizer.off for n in 1:szz]..., dims=2)
    success = sum(abs.(collect(svvoff) .- vec_true.off), dims=1) .< 0.5
    success = success .&& sum(abs.(collect(resoff) .- vec_true.off), dims=1) .< 0.5
    # all worked!
    sum(.!success)
end
# findfirst(.!success)
@vt collect(dat)[:,:,.!success[:]] collect(pdat)[:,:,.!success[:]] collect(gaussian_vec(sz, startvals))[:,:,.!success[:]] collect(gaussian_vec(sz, reso.minimizer))[:,:,.!success[:]]

ff = findfirst(.!success)[2]
sv = ComponentVector(off=startvals.off[:,ff], bg = startvals.bg[:,ff], intensity=startvals.intensity[:,ff], args = startvals.args[:,ff])
sdat = dat[:,:,ff]
afg! = get_fg!(sdat, gaussian_raw, length(sz); loss=loss_gaussian); # loss_poisson_pos
odo = OnceDifferentiable(Optim.NLSolversBase.only_fg!(afg!), sv);
@time reso = Optim.optimize(odo, sv, Optim.LBFGS(), opt);
@vt sdat gaussian_vec(sz, sv) gaussian_vec(sz, reso.minimizer)

trueoff = vec_true.off[:,collect(success[:])]
startoff = collect(startvals.off[:,success[:]])
minoff = collect(reso.minimizer.off[:,success[:]])
sum(abs2.(startoff .- trueoff), dims=2) # 0 failed, 68.29712, 42.612137
sum(abs2.(minoff .- trueoff), dims=2) # 21 failed, 57.92457, 33.990776 # Anscombe: 5 failed, 58.806953, 34.21795
@vt abs.(startoff .- trueoff) abs.(minoff .- trueoff)

sum(abs2.(collect(startvals.intensity) .- vec_true.intensity))
sum(abs2.(collect(reso.minimizer.intensity) .- vec_true.intensity))
sum(abs2.(collect(startvals.bg) .- vec_true.bg))
sum(abs2.(collect(reso.minimizer.bg) .- vec_true.bg))
sum(abs2.(collect(startvals.args) .- vec_true.args), dims=2)
sum(abs2.(collect(reso.minimizer.args) .- vec_true.args), dims=2)

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

# here is the LsqFit example from https://github.com/JuliaNLSolvers/LsqFit.jl
using LsqFit

x = collect(range(0, stop=200, length=201))
y = collect(range(0, stop=200, length=201))

xy = hcat(x, y)

function twoD_Gaussian(xy, p)
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = p
    a = (cos(theta)^2)/(2*sigma_x^2) + (sin(theta)^2)/(2*sigma_y^2)
    b = -(sin(2*theta))/(4*sigma_x^2) + (sin(2*theta))/(4*sigma_y^2)
    c = (sin(theta)^2)/(2*sigma_x^2) + (cos(theta)^2)/(2*sigma_y^2)

    # creating linear meshgrid from xy
    x = xy[:, 1]
    y = xy[:, 2]
    g = offset .+ amplitude .* exp.( - (a.*((x .- xo).^2) + 2 .* b .* (x .- xo) .* (y .- yo) + c * ((y .- yo).^2)))
    return g[:]
end

p0 = Float64.([3, 100, 100, 20, 40, 0, 10])
data = twoD_Gaussian(xy, p0)

# Noisy data
data_noisy = data + 0.2 * randn(size(data))

fit = LsqFit.curve_fit(twoD_Gaussian, xy, data_noisy, p0)
