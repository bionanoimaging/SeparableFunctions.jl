using Test
using IndexFunArrays
using SeparableFunctions

function test_fct(T, fcts, sz, args...; kwargs...)
    ifa, fct = fcts
    a = let 
        if typeof(ifa) <: AbstractArray
            ifa
        else
            ifa(T, sz, args...; kwargs...)
        end
    end

    res = fct(Array{T}, sz, args...; kwargs...)
    if typeof(res) <: Tuple
        res = res[2].(res[1]...)
    end
    # @test (typeof(res) <: AbstractArray) == false
    res = collect(res)
    @test (typeof(res) <: AbstractArray) == true

    @test a≈res
    @test eltype(res)==T

    all_axes = zeros(T, prod(sz))
    res2 = fct(Array{T}, sz, args...; all_axes = all_axes, kwargs...)
    if typeof(res2) <: Tuple
        res2 = res2[2].(res2[1]...)
    end
    # @test (typeof(res2) <: AbstractArray) == false
    res2 = collect(res2)
    @test (typeof(res2) <: AbstractArray) == true
    @test res≈res2
    @test sum(abs.(all_axes)) > 0
end


function test_fct_t(fcts, sz, args...; kwargs...)
    test_fct(Float32, fcts, sz, args...;kwargs...)
    test_fct(Float64, fcts, sz, args...;kwargs...)
end

@testset "calculate_separables" begin
    sz = (13,15)
    fct = (r, sz, sigma)-> exp(-r^2/(2*sigma^2))
    offset = (2.2, -2.2)  ; scale = (1.1, 1.2); factor = 1.0;
    @time gauss_sep = calculate_separables(fct, sz, (0.5,1.0), pos = (0.1,0.2), offset=offset, scale=scale, factor=factor)
    @test size(.*(gauss_sep...)) == sz
    # test with preallocated array
    all_axes = zeros(Float32, prod(sz))
    @time gauss_sep = calculate_separables(fct, sz, (0.5,1.0), all_axes = all_axes, pos = (0.0,0.0))
    @test all_axes[7] ≈ 1.0
    @test all_axes[13+8] ≈ 1.0
end

@testset "gaussian" begin
    sz = (11,22)
    sigma = (11.2, 5.5)
    mygaussian = gaussian(sz, sigma=sigma)
    test_fct_t((mygaussian, gaussian_col), sz; sigma=sigma);    
    test_fct_t((mygaussian, SeparableFunctions.gaussian_lz), sz; sigma=sigma);    
    test_fct_t((mygaussian, gaussian_sep), sz; sigma=sigma);    
    offset = sz.÷2 .+1 ; scale = (1.0, 1.0); factor = 1.0;
    test_fct_t((mygaussian, gaussian_nokw_sep), sz, offset, scale, factor, sigma);    

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
    factor = 2.0;
    test_fct_t((2 .*myrr2, rr2_nokw_sep), sz, offset, scale, factor);

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
    test_fct_t((myxy, ramp_nokw_sep), sz, nothing, nothing, nothing, slope);
end

@testset "exp_ikx" begin
    sz = (11, 22, 4)
    shift_by = (1.1, 0.2, 2.2)
    myexp_ikx = exp_ikx(sz; shift_by = shift_by)
    # scale leads to problems! Since  exp_ikx(sz) ≈ exp_ikx(sz, scale=(1.0,1.0,1.0))   -> false
    test_fct(ComplexF32, (myexp_ikx, exp_ikx_col), sz; shift_by=shift_by);
    test_fct(ComplexF32, (myexp_ikx, SeparableFunctions.exp_ikx_lz), sz; shift_by=shift_by);
    test_fct(ComplexF32, (myexp_ikx, exp_ikx_sep), sz; shift_by=shift_by);
    test_fct(ComplexF32, (myexp_ikx, exp_ikx_nokw_sep), sz, nothing, nothing, nothing, shift_by);

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

return
