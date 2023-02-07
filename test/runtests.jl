using Test
using IndexFunArrays
using SeparableFunctions

function test_fct(T, fcts, sz, args...; kwargs...)
    ifa, col, lz, sep, op = fcts
    a = let 
        if typeof(ifa) <: AbstractArray
            ifa
        else
            ifa(T, sz, args...; kwargs...)
        end
    end
    b = col(Array{T}, sz, args...; kwargs...)
    c = lz(Array{T}, sz, args...; kwargs...)
    res = sep(Array{T}, sz, args...; kwargs...)
    d = op.(res...)
    @test a≈b
    @test eltype(b)==T
    @test a≈c
    @test eltype(c)==T
    @test a≈d
    @test eltype(d)==T
end

function test_fct_t(fcts, sz, args...; kwargs...)
    test_fct(Float32, fcts, sz, args...;kwargs...)
    test_fct(Float64, fcts, sz, args...;kwargs...)
end

@testset "gaussian" begin
    sz = (11,22)
    test_fct_t((gaussian, gaussian_col, SeparableFunctions.gaussian_lz, gaussian_sep, *), sz; sigma=(11.2, 5.5));
end

@testset "rr2" begin
    sz = (11,22, 3)
    offset = (2,3,1)
    test_fct_t((rr2, rr2_col, SeparableFunctions.rr2_lz, rr2_sep, +), sz; scale=(2.2, 3.3, 1.0), offset=offset);
end

@testset "box" begin
    sz = (11,22, 3)
    offset = (2,3,1)
    test_fct_t((box, box_col, SeparableFunctions.box_lz, box_sep, *), sz; scale=(2.2, 3.3, 1.0), offset=offset);
end

@testset "ramp" begin
    sz = (11,22)
    test_fct_t((xx(sz) .+ yy(sz), ramp_col, SeparableFunctions.ramp_lz, ramp_sep, +), sz; slope=(1.0,1.0));
end

@testset "exp_ikx" begin
    sz = (11, 22, 4)
    # scale leads to problems! Since  exp_ikx(sz) ≈ exp_ikx(sz, scale=(1.0,1.0,1.0))   -> false
    test_fct(ComplexF32, (exp_ikx, exp_ikx_col, SeparableFunctions.exp_ikx_lz, exp_ikx_sep, *), sz; shift_by=(1.1,0.2,2.2));
    myshift = (0.1,0.2,0.3)
    a = ones(ComplexF64,sz)
    SeparableFunctions.mul_exp_ikx!(a; shift_by=myshift)
    @test exp_ikx(sz; shift_by = myshift) ≈ a
end

@testset "sinc" begin
    sz = (12, 23)
    scale = (1.1, 2.2)
    mysinc = sinc.(xx(sz; scale=scale)) .* sinc.(yy(sz; scale=scale));
    test_fct(Float32, (mysinc, sinc_col, SeparableFunctions.sinc_lz, sinc_sep, *), sz; scale=scale);
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
