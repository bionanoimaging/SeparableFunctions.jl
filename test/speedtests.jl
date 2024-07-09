using Test
using IndexFunArrays
using SeparableFunctions
using BenchmarkTools
using CUDA

function speedt_test()
    sz = (256, 256, 256)
    sigma = (11.2f0, 5.5f0, 2.2f0)
    offset = (2.2f0, 100.0f0, 100.0f0)
    @time mygaussian = gaussian(Float32, sz, sigma=sigma, offset=offset);
    @btime collect($mygaussian);  # 67ms (256, 256, 256)

    function get_exp(ci, sigma, offset)
        exp(-sum(abs2.((Tuple(ci) .- offset)./sigma)))
    end
    res = get_exp.(CartesianIndices(sz), Ref(Float32(sqrt(2)).*sigma), Ref(offset));  
    @btime get_exp.(CartesianIndices($sz), Ref($sigma)); # 47.7 ms (2 allocations, 64 Mb) , but 243 ms with offset!
    @btime get_exp.(CartesianIndices($sz), Ref($sigma), Ref(offset)); # 47.7 ms, but 243 ms with offset (7 allocations, 64 Mb)!

    res2 = similar(res);
    ress = gaussian_sep(sz, sigma=sigma, offset=offset);
    @btime $ress = gaussian_sep($sz, sigma=$sigma, offset=$offset); # 12.3 µs
    resns = gaussian_nokw_sep(sz, offset, 1f0, 1f0, sigma); 
    @btime $resns = gaussian_nokw_sep($sz, $offset, 1f0, 1f0, $sigma); # 12 µs
    res2 .= ress; 
    res2 ≈ res
    @btime $res2 .= $ress; # 8.35 ms
    @btime $res2 = similar($res); # 0.04 ms
    @btime $res2 .= gaussian_sep($sz, sigma=$sigma, offset=$offset); # 8.4 ms

    # res3 = gaussian_col(sz, sigma=sigma, offset=offset);
    t_col = @btime $res3 = gaussian_col($sz, sigma=$sigma, offset=$offset); # 14 ms

    @btime $res2 .= SeparableFunctions.gaussian_lz($sz, sigma=$sigma, offset=$offset); # 8.47 ms

    resc = CuArray(res);
    res3c = gaussian_col(typeof(resc), sz, sigma=sigma, offset=offset); # 
    @btime CUDA.@sync $res3 = gaussian_col(typeof(resc), $sz, sigma=$sigma, offset=$offset); # 0.983 ms
     
    ids = CuArray(CartesianIndices(sz))
    resc = get_exp.(ids, Ref(sigma), Ref(offset)); 
    @btime CUDA.@sync $resc = get_exp.($ids, Ref($sigma)); # 2.83 ms


    t_in_place = @belapsed get_exp.(CartesianIndices($sz), Ref($sigma), Ref(offset)); # 47.7 ms, but 243 ms with offset (7 allocations, 64 Mb)!
    t_gaussian_col = @belapsed $res3 = gaussian_col($sz, sigma=$sigma, offset=$offset)
    t_gaussian_lz = @belapsed $res2 .= SeparableFunctions.gaussian_lz($sz, sigma=$sigma, offset=$offset)
    # t_gaussian_nokw_sep = @belapsed $resns = gaussian_nokw_sep($sz, $offset, 1f0, 1f0, $sigma)
    t_res2_assign = @belapsed $res2 .= $ress
    # t_similar_res = @belapsed $res2 = similar($res)
    t_gaussian_sep = @belapsed res_gs = gaussian_sep($sz, sigma=$sigma, offset=$offset)


    tc_get_exp = @belapsed CUDA.@sync $resc = get_exp.($ids, Ref($sigma))
    tc_gaussian_col = @belapsed CUDA.@sync $res3 = gaussian_col(typeof(resc), $sz, sigma=$sigma, offset=$offset)

    # NOT working: resc .= SeparableFunctions.gaussian_lz(typeof(resc), sz, sigma=sigma, offset=offset)
    tc_gaussian_lz = NaN # @belapsed collect(SeparableFunctions.gaussian_lz(typeof($resc), $sz, sigma=$sigma, offset=$offset))
    tc_gaussian_sep = @belapsed CUDA.@sync res_gsc = gaussian_sep(typeof(resc), $sz, sigma=$sigma, offset=$offset)
    tc_gaussian_col = @belapsed CUDA.@sync gaussian_col(typeof(resc), $sz, sigma=$sigma, offset=$offset)

    res3_sep = gaussian_sep(typeof(resc), sz, sigma=sigma, offset=offset); # 
    tc_res2_assign = @belapsed CUDA.@sync $resc .= $res3_sep
    
    using PlotlyJS
    method = ["Compute In Place", "Collect Separables", "Lazy Arrays", "Collect Precomputed", "Precompute"]
    dat_no_cuda = 1000 .*[t_in_place, t_gaussian_col, t_gaussian_lz, t_res2_assign, t_gaussian_sep]
    dat_cuda = 1000 .*[tc_get_exp, tc_gaussian_col, tc_gaussian_lz, tc_res2_assign, tc_gaussian_sep]
    
    p = plot([
        bar(name="With CUDA", x=method, y=dat_cuda),
        bar(name="No CUDA", x=method, y=dat_no_cuda),
    ], Layout(title="3D Gaussian (512x512x256)", yaxis=attr(title="Time [ms]", type="log"))) # barmode="stack", 


    # now som speed comparison for propagator_col!
    sz = (1024, 1024)
    Δz = 1f0
    scale = 0.5f0 ./ (max.(sz ./ 2, 1))
    arr = zeros(ComplexF32, sz)
    k_max=0.5f0
    k2_max = real(eltype(arr))(k_max .^2)
    fac = real(eltype(arr))(4pi * Δz)
    myzero = zero(real(eltype(arr)))
    # f(x) = cis(sqrt(max(myzero, k2_max - x)) * fac)
    # CUDA has problems with (global) Clojures..., and julia allocates in this case 
    g(x) = cis(sqrt(max(0f0, 0.25f0 - x)) * 12.566371f0)

    myrr2 = collect(rr2_sep(sz; scale=scale))
    @time res .= g.(rr2_sep(sz; scale=scale)); # 11.7 kB
    t_no_rad = @belapsed res .= g.($myrr2);  # 7 ms

    @time propagator_col!(arr; Δz=Δz, k_max=k_max, scale=scale); # 10.3 kB
    res ≈ arr
    t_rad_speedup = @belapsed  propagator_col!($arr; Δz=$Δz, k_max=$k_max, scale=$scale); # 2.4 ms
    # @time  propagator_col!(arr; Δz=Δz, k_max=k_max, scale=scale); # 2.4 ms, 10 kB

    myrr2c = CuArray(myrr2)
    resc = CuArray(res)
    arrc = CuArray(arr)
    tc_no_rad = @belapsed CUDA.@sync $resc .= g.($myrr2c);  # 0.1ms
    tc_rad_speedup = @belapsed  CUDA.@sync propagator_col!($arrc; Δz=$Δz, k_max=$k_max, scale=$scale); # 0.26 msec ms

    method = ["Compute In Place", "Radial Speedup"]
    dat_no_cuda = 1000 .*[t_no_rad, t_rad_speedup]
    dat_cuda = 1000 .*[tc_no_rad, tc_rad_speedup]
    
    p = plot([
        bar(name="With CUDA", x=method, y=dat_cuda),
        bar(name="No CUDA", x=method, y=dat_no_cuda),
    ], Layout(title="2D Propagator (1024 x 1024)", yaxis=attr(title="Time [ms]", type="log"))) # barmode="stack", 
    
end

# TODO:
# gaussian_nokw_sep((10,10), (2.2,3.3), 1.0, 1.0, 1.0)[4, 7] # works
# gaussian_nokw_sep((10,10), (2.2,3.3), 1.0, 1.0, 1.0)[3:5, 3:7] # does not work, but should ideally give a 3x5 array, and support @view

if (false)
    using Zygote
    using SeparableFunctions
    @testset "gradients" begin
        sz = (1000, 1000)
        offset = (2.2,3.3)
        # collect(gaussian_nokw_sep(sz, offset, 1.0, 1.0, 1.0))
        # loss(x) = sum(abs2.(gaussian_nokw_sep(sz, x, 1.0, 1.0, 1.0)))
        dat = rand(sz...)
        bw = gaussian_nokw_sep(sz, offset, 1.0, 1.0)
        bw .+ 0
        # https://fluxml.ai/Zygote.jl/latest/limitations
        # Zygote.buffer
        tmp_mem = similar(dat, prod(sz))
        function loss(off)
            # g, op = gaussian_nokw_sep(sz, off, 1.0, 1.0; all_axes=tmp_mem)
            # bw = Broadcast.instantiate(Broadcast.broadcasted(op, g...))
            bw = gaussian_nokw_sep(sz, off, 1.0, 1.0; all_axes=tmp_mem)
            # sum(abs2.(op.(g...) .- dat))
            sum(abs2.(bw .- dat))
        end
        @time loss(offset)
        # loss(x) = gaussian_nokw_sep(sz, x, 1.0, 1.0, 1.0)[3,3] # problems
        @time gradient(loss, offset)[1]

        sz = (10,10)
        offset = (2.2,3.3)
        # f(x) = ([(1:size(dat,1))...] .- x[1]) .* (reshape([(1:size(dat,1))...],(1, size(dat,1))) .- x[2])
        f(x) = Broadcast.instantiate(Broadcast.broadcasted(*, [(1:size(dat,1))...] .- x[1], reshape([(1:size(dat,1))...],(1, size(dat,1))) .- x[2]))
        # f(x) = collect(gaussian_nokw_sep(sz, x, 1.0, 1.0, 1.0))
        # f(x) = gaussian_nokw_sep(sz, x, 1.0, 1.0, 1.0)
        f(offset)
        collect(f(offset))
        function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(f), offset)
            @show "in rrule"
            function f_pullback(dy)
                @show "in f_pullback"
                # @show dy .* f(offset)
                @show size(dy)
                @show size(f(offset))
                return NoTangent(), (42.0, 55.0)
            end
            return collect(f(offset)), f_pullback
        end
        loss(x) = sum(abs2.(f(x) .- dat))
        loss(offset)
        gradient(loss , offset)[1]


        using Zygote
        using PreallocationTools
        sz = (10, 10)
        offset = (2.2,3.3)
        dat = rand(sz...)

        cache1 = DiffCache(@view (dat[:,1]))
        cache2 = DiffCache(@view (dat[1,:]))
        function uses_tmp(x)
            tmp1 = get_tmp(cache1, x)
            tmp2 = get_tmp(cache2, x)
            # @show size(tmp1)
            # @show size(tmp2)
            tmp1 .= 1:size(dat,1) .- x[1]
            tmp2 .= 1:size(dat,2) .- x[2]
            return (tmp1, tmp2)
        end
        loss(x) = sum(abs2.(.*(uses_tmp(x)...) .- dat))
        @time loss(offset)
        gradient(loss , offset)[1]


        function save_assign!(x, y)
            x = y
        end

        q = ones(10)
        z = zeros(10)
        save_assign!.(q, z)
        q


        y(x) = ([(1:size(dat,1))...] .- x[1], reshape([(1:size(dat,1))...],(1, size(dat,1))) .- x[2])
        loss(x) = sum(abs2.(.*(y(x)...) .- dat))
        loss(offset)
        gradient(loss , offset)[1]


    end
end
return
