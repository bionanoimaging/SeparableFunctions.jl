using Test
using IndexFunArrays
using SeparableFunctions
using FiniteDifferences
using Zygote
using Random
using ComponentArrays

function test_fct(T, fcts, sz, args...; kwargs...)
    AT = Array{T}
    ifa, fct = fcts
    a = let 
        if typeof(ifa) <: AbstractArray
            ifa
        else
            ifa(T, sz, args...; kwargs...)
        end
    end

    res = fct(AT, sz, args...; kwargs...)
    if typeof(res) <: Tuple
        res = res[2].(res[1]...)
    end
    # @test (typeof(res) <: AbstractArray) == false
    res = collect(res)
    @test (typeof(res) <: AbstractArray) == true

    @test a≈res
    @test eltype(res)==T

    all_axes = mem_fct(fct, AT, sz)# get_bc_mem(AT, sz, *) # zeros(T, prod(sz))
    res2 = fct(AT, sz, args...; all_axes = all_axes, kwargs...)
    if typeof(res2) <: Tuple
        res2 = res2[2].(res2[1]...)
    end
    # @test (typeof(res2) <: AbstractArray) == false
    res2 = collect(res2)
    @test (typeof(res2) <: AbstractArray) == true
    @test res≈res2
#    @test sum(abs.(all_axes)) > 0
end


function test_fct_t(fcts, sz, args...; kwargs...)
    test_fct(Float32, fcts, sz, args...;kwargs...)
    test_fct(Float64, fcts, sz, args...;kwargs...)
end

@testset "calculate_separables" begin
    sz = (13,15)
    fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
    offset = (2.2, -2.2); scale = (1.1, 1.2); 
    @time gauss_sep = calculate_separables(fct, sz, (0.5,1.0), offset = offset .+ (0.1,0.2), scale=scale)
    @test size(.*(gauss_sep...)) == sz
    # test with preallocated array
    all_axes = get_sep_mem(Array{Float32}, sz) # zeros(Float32, prod(sz))
    @time gauss_sep = calculate_separables(fct, sz, (0.5, 1.0), all_axes = all_axes)
    @test all_axes[1][7] ≈ 1.0
    @test all_axes[2][8] ≈ 1.0
end

@testset "gaussian" begin
    sz = (11,22)
    sigma = (11.2, 5.5)
    mygaussian = gaussian(sz, sigma=sigma)
    test_fct_t((mygaussian, gaussian_col), sz; sigma=sigma);    
    test_fct_t((mygaussian, SeparableFunctions.gaussian_lz), sz; sigma=sigma);    
    test_fct_t((mygaussian, gaussian_sep), sz; sigma=sigma);    
    offset = sz.÷2 .+1 ; scale = (1.0, 1.0); 
    test_fct_t((mygaussian, gaussian_nokw_sep), sz, offset, scale, sigma);    

    # test_fct_t((gaussian, gaussian_col, SeparableFunctions.gaussian_lz, gaussian_sep, *), sz; sigma=(11.2, 5.5));    
    # # test with preallocated array
    # all_axes = zeros(Float32, prod(sz))
    # test_fct_t((gaussian, gaussian_col, SeparableFunctions.gaussian_lz, gaussian_sep, *), sz; all_axes = all_axes, sigma=(11.2, 5.5));
end

@testset "rr2" begin
    sz = (11,22, 3)
    offset = (2,3,1) # try some offset not in the center
    scale = (2.2, 3.3, 1.0) # and a non-unity scale
    myrr2 = rr2(sz; offset=offset, scale=scale)
    test_fct_t((myrr2, rr2_col), sz; scale=scale, offset=offset);
    test_fct_t((myrr2, SeparableFunctions.rr2_lz), sz; scale=scale, offset=offset);
    test_fct_t((myrr2, rr2_sep), sz; scale=scale, offset=offset);
    test_fct_t((myrr2, rr2_nokw_sep), sz, offset, scale);

    offset = sz .÷ 2 .+1 # try some offset not in the center
    scale = (1.0, 1.0, 1.0) # and a non-unity scale
    myrr2 = rr2(sz; offset=offset, scale=scale)
    test_fct_t((myrr2, rr2_nokw_sep), sz); # should be the same as the default
end

@testset "box" begin
    sz = (11,22, 3)
    offset = (2,3,1)
    scale = (2.2, 3.3, 1.0)
    mybox = box(sz; offset=offset, scale=scale)
    test_fct_t((mybox, box_col), sz; scale=scale, offset=offset);
    test_fct_t((box, SeparableFunctions.box_lz), sz; scale=scale, offset=offset);
    test_fct_t((box, box_sep), sz; scale=scale, offset=offset);
    test_fct_t((mybox, box_nokw_sep), sz, offset, scale); 
end

@testset "ramp" begin
    sz = (11,22)
    slope = (1.0, 2.2)
    myxy = slope[1].*xx(sz) .+ slope[2].*yy(sz)
    test_fct_t((myxy, ramp_col,), sz; slope=slope);
    test_fct_t((myxy, SeparableFunctions.ramp_lz), sz; slope=slope);
    test_fct_t((myxy, ramp_sep), sz; slope=slope);
    test_fct_t((myxy, ramp_nokw_sep), sz, nothing, nothing, slope);
end

@testset "exp_ikx" begin
    sz = (11, 22, 4)
    shift_by = (1.1, 0.2, 2.2)
    myexp_ikx = exp_ikx(sz; shift_by = shift_by)
    # scale leads to problems! Since  exp_ikx(sz) ≈ exp_ikx(sz, scale=(1.0,1.0,1.0))   -> false
    test_fct(ComplexF32, (myexp_ikx, exp_ikx_col), sz; shift_by=shift_by);
    test_fct(ComplexF32, (myexp_ikx, SeparableFunctions.exp_ikx_lz), sz; shift_by=shift_by);
    test_fct(ComplexF32, (myexp_ikx, exp_ikx_sep), sz; shift_by=shift_by);
    test_fct(ComplexF32, (myexp_ikx, exp_ikx_nokw_sep), sz, nothing, nothing, shift_by);

    myshift = (0.1,0.2,0.3)
    a = ones(ComplexF64,sz)
    SeparableFunctions.mul_exp_ikx!(a; shift_by=myshift)
    @test exp_ikx(sz; shift_by = myshift) ≈ a
end

@testset "sinc" begin
    sz = (12, 23)
    scale = (1.1, 2.2)
    mysinc = sinc.(xx(sz; scale=scale)) .* sinc.(yy(sz; scale=scale));
    test_fct(Float32, (mysinc, sinc_col), sz; scale=scale);
    test_fct(Float32, (mysinc, SeparableFunctions.sinc_lz), sz; scale=scale);
    test_fct(Float32, (mysinc, sinc_sep), sz; scale=scale);
    test_fct(Float32, (mysinc, sinc_nokw_sep), sz, nothing, scale);
end

function test_copy_corners(sz)
    q = copy_corners!(reshape(collect(1:prod(sz)),sz), speedup_last_dim=false);
    w = copy_corners!(reshape(collect(1:prod(sz)),sz), speedup_last_dim=true);
    @test all(w .== q)
end

@testset "copy_corners" begin
    test_copy_corners((4,4))
    test_copy_corners((4,5))
    test_copy_corners((5,4))
    test_copy_corners((3,4,5))
    test_copy_corners((4,6,5))
    test_copy_corners((3,5,4))
    test_copy_corners((1,1,5))
    test_copy_corners((3,4,1))
end

@testset "radial speedup" begin
    sz = (233,244)
    f(r)=sinc(r/2f0)
    res3 = f.(rr(sz))
    res2 = radial_speedup(f, sz, oversample=8f0)
    @test maximum(abs.(res3 .- res2)) < 1e-6  # Linear: 1e-3, quadratic: 1.4e-5, Cubic: 5e-7
    res = calc_radial_symm(sz, f);
    @test maximum(abs.(res .- res3)) < 1e-7

    sigma = 50.0
    res4 = gaussian_col(sz, sigma=sigma) 
    res5 = radial_speedup_ifa(gaussian, sz; sigma=sigma) 
    @test maximum(abs.(res4 .- res5)) < 1e-6
end

function check_all(T, args...; kwargs...)
    for a in args[2:end]
        for r in 1:length(a)
            @test eltype(a[r]) == T
            @test all(isapprox.(a[r], args[1][r]; kwargs...))
        end
    end
end

function test_gradient(T, fct, sz, args...; kwargs...)
    RT = real(T)
    Random.seed!(1234)
    dat = rand(T, sz...)
    off0 = rand(RT, length(sz))
    sca0 = rand(RT, length(sz))
    argsc = ntuple((d)->RT.(args[d]), length(args))
    loss = (off, sca, args...) -> sum(abs2.(fct(sz, off, sca, args..., kwargs...) .- dat))
    # @show loss(off0, sca0, argsc...)
    gn = grad(central_fdm(5, 1), loss, off0, sca0, argsc...) # 5th order method, 1st derivative
    g = gradient(loss, off0, sca0, argsc...)

    fg! = get_fg!(dat, raw_fct(fct), length(sz); loss = loss_gaussian)
    if length(argsc)>0
        vec = ComponentVector(;off=off0, sca=sca0, args=[argsc[1]...])
    else
        vec = ComponentVector(;off=off0, sca=sca0)
    end
    G = copy(vec)
    f = fg!(1, G, vec)
    if length(argsc)>0
        G = (G.off, G.sca, G.args)
    else
        G = (G.off, G.sca)
    end
    
    check_all(RT, gn, g, G; rtol=1e-1)
end

@testset "gradient tests" begin
    rng = collect(1:0.1:2)
    sz = length(rng)

    loss = (x, sigma) -> sum(gaussian_raw.(x, sz, sigma))
    sigma0 = 2.0
    loss(rng, sigma0)
    g = gradient(loss, rng, sigma0)
    gn = grad(central_fdm(5, 1), loss, rng, sigma0) # 5th order method, 1st derivative
    @test g[1] ≈ gn[1]
    @test g[2] ≈ gn[2]
    # here some detailed tests for the complex-valued functions:

    sz = (2,2)
    sz = (22, 11)
    rfun(off, sca, arg) = real(sum(exp_ikx_nokw_sep(sz, off, sca, arg)))
    ifun(off, sca, arg) = imag(sum(exp_ikx_nokw_sep(sz, off, sca, arg)))
    lfun(off, sca, arg) = sum(abs2.(exp_ikx_nokw_sep(sz, off, sca, arg) .- 1.0 .- 1.0im))
    gn = grad(central_fdm(5, 1), ifun, (0.0,0.0), (1.0, 1.0), (0.0,0.0)) # 5th order method, 1st derivative
    g = gradient(ifun, (0.0,0.0), (1.0, 1.0), (0.0,0.0))
    check_all(Float64, gn, g; atol=0.001)

    gn = grad(central_fdm(5, 1), ifun, (0.0,0.0), (2.0, 3.0), (2.0, 1.0)) # 5th order method, 1st derivative
    g = gradient(ifun, (0.0,0.0), (2.0, 3.0), (2.0, 1.0))
    check_all(Float64, gn, g; atol=0.01)

    # problem only when the shift vector is diagonal and offset is non-zero
    gn = grad(central_fdm(5, 1), ifun, (0.3, 1.2), (0.6, 1.0), (1.0, 1.0)) # 5th order method, 1st derivative
    g = gradient(ifun, (0.3, 1.2), (0.6, 1.0), (1.0, 1.0))
    check_all(Float64, gn, g; atol=0.05)

    gn = grad(central_fdm(5, 1), rfun, (1.3, 1.2), (0.6, 1.5), (1.0, 2.0)) # 5th order method, 1st derivative
    g = gradient(rfun, (1.3, 1.2), (0.6, 1.5), (1.0, 2.0))
    check_all(Float64, gn, g; atol=0.02)

    off0 = (1.3f0, 1.2f0); sca0 = (0.6f0, 1.5f0); args0 = (1.0f0, 2.0f0)
    gn = grad(central_fdm(5, 1), lfun, off0 , sca0, args0) # 5th order method, 1st derivative
    g = gradient(lfun, off0 , sca0, args0)
    check_all(Float32, gn, g; atol=0.5)

    test_gradient(Float32, gaussian_nokw_sep, (11,22), (2.2, -0.8))
    test_gradient(Float64, gaussian_nokw_sep, (6, 22, 7), 2.0)
    test_gradient(Float32, gaussian_nokw_sep, (6,), 4.2f0)

    test_gradient(Float64, normal_nokw_sep, (6, 22, 7), (2.0, -3.1, 1.2))

    test_gradient(Float32, sinc_nokw_sep, (22, 11))
    test_gradient(Float32, ramp_nokw_sep, (22, 11), (1.0, 2.0))
    test_gradient(Float32, rr2_nokw_sep, (22, 11))
    test_gradient(ComplexF32, exp_ikx_nokw_sep, (22, 11), (1.0, 2.0))

    sz = (3,)
    loss2 = (off, sca, shift_by) -> sum(imag.(exp_ikx_nokw_sep(sz, off, sca, shift_by)))
    shift_by0 = 0.7
    sca0 = 1.4
    off0 = 0.3
    loss2(off0, sca0, shift_by0)
    g = gradient(loss2, off0, sca0, shift_by0)
    gn = grad(central_fdm(5, 1), loss2, off0, sca0, shift_by0) # 5th order method, 1st derivative
    @test all(isapprox.(g[1], gn[1], atol=5e-3))
    @test all(isapprox.(g[2], gn[2], atol=1e-2))

    sz = (11, 22, 7)
    loss2 = (off, sca, sigma) -> sum(gaussian_nokw_sep(sz, off, sca, sigma))
    sigma0 = 2.0
    sca0 = (0.9, 1.2, 0.4)
    off0 = (0.9, 1.2, 0.4)
    loss2(off0, sca0, sigma0)
    g = gradient(loss2, off0, sca0, sigma0)
    gn = grad(central_fdm(5, 1), loss2, off0, sca0, sigma0) # 5th order method, 1st derivative    

    @test all(isapprox.(g[1], gn[1], atol=5e-3))
    @test all(isapprox.(g[2], gn[2], atol=1e-2))

    # now the vec version with intensity, scale, offset and bg (just a single replica):
    for use_hyper_dims in (true, false)
        N_hyper = 2
        hyperint = (use_hyper_dims) ? 1.0 .+ zeros(1,N_hyper) : 1.0
        hyperoff = (use_hyper_dims) ? 1 .+ 0.2 .* zeros(1,N_hyper) : 1
        hyperarg = (use_hyper_dims) ? 1 .+ 0.2 .* zeros(1,N_hyper) : 1
        sz = (11, 22)
        # sz = (3, 3)
        vec_true = ComponentVector(;bg=0.0, intensity=1.0 .*hyperint, off = [2.2, 3.3].*hyperoff, sca = [1.3, 1.2], args = [2.4, 1.5].*hyperarg)
        dat = gaussian_vec(sz, vec_true)
        loss2 = (vec) -> sum(abs2.(gaussian_vec(sz, vec) .- dat))
        @test loss2(vec_true) == 0
        g = gradient(loss2, vec_true)[1]
        gn = grad(central_fdm(5, 1), loss2, vec_true)[1] # 5th order method, 1st derivatives
        myfg! = get_fg!(dat, gaussian_raw, length(sz); loss = loss_gaussian) 
        G = similar(gn) .* 0
        f = myfg!(1, G, vec_true)
        # maximum(abs.(G))
        check_all(Float64, zeros(length(g)), g, gn, G; atol=5e-12)

        # vec_start = ComponentVector(;bg=0.3, intensity=1.1, off = [2.3, 3.4], sca = [1.4, 1.3], args = [2.5, 1.6])
        vec_start = vec_true .+ 0.2
        gs = gradient(loss2, vec_start)[1]
        gns = grad(central_fdm(5, 1), loss2, vec_start)[1] # 5th order method, 1st derivatives
        Gs = similar(gns) .* 0
        fs = myfg!(1, Gs, vec_start)
        # maximum(abs.(Gs[:] .- gns[1][:]))
        for (mygn, myg, myfg) in zip(gns, gs, Gs)
            @test all(isapprox.(mygn, myg, atol=4e-7))
            @test all(isapprox.(mygn, myfg, atol=4e-6))
        end
    end
end

return
