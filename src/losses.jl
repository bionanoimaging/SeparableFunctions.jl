# from InverseModeling.jl
loss_gaussian(data::AbstractArray{T}, fwd, bg=zero(T)) where {T} = mapreduce(abs2, +, fwd .- data ; init=zero(T))
function get_fgC!(data::AbstractArray{T,N}, ::typeof(loss_gaussian), bg=zero(T)) where {T,N}
    function fg!(F, G, fwd)
        if !isnothing(G)
            G .= (fwd .- data)
            return (!isnothing(F)) ? mapreduce(abs2, +, G; init=zero(T)) :  T(0);
        end
        return (!isnothing(F)) ? mapreduce(abs2, +, (fwd .- data); init=zero(T)) : T(0);
    end
    return fg!, 2
end

loss_poisson(data::AbstractArray{T}, fwd, bg=zero(T)) where {T} = sum((fwd.+bg) .- (data.+bg).*log.(fwd.+bg))
function get_fgC!(data::AbstractArray{T,N}, ::typeof(loss_poisson), bg=zero(T)) where {T,N}
    function fg!(F, G, fwd)
        if !isnothing(G)
            G .= one(T) .- (data .+ bg)./max.(fwd .+ bg, max(b, T(1f-10)))
        end
        return (!isnothing(F)) ? reduce(+, (fwd.+bg) .- (data.+bg).*log.(fwd.+bg); init=zero(T)) : T(0);
    end
    return fg!, 1
end

loss_poisson_pos(data::AbstractArray{T}, fwd, bg=zero(T)) where {T} = sum(max.(eltype(fwd)(0),fwd.+bg) .- max.(eltype(fwd)(0),data.+bg).*log.(max.(eltype(fwd)(0),fwd.+bg)))
function get_fgC!(data::AbstractArray{T,N}, ::typeof(loss_poisson_pos), bg=zero(T)) where {T,N}
    function fg!(F, G, fwd)
        if !isnothing(G)
            G .= one(T) .- (data .+ bg)./max.(fwd .+ bg, T(1f-10))
        end
        return (!isnothing(F)) ? reduce(+, max.(eltype(fwd)(0),(fwd.+bg)) .- max.(eltype(fwd)(0),(data.+bg)).*log.(max.(eltype(fwd)(0),fwd.+bg)); init=zero(T)) : T(0);
    end
    return fg!, 1
end

loss_anscombe(data::AbstractArray{T}, fwd, bg=zero(T)) where {T} = sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg)))
function get_fgC!(data::AbstractArray{T,N}, ::typeof(loss_anscombe), bg=zero(T)) where {T,N}
    function fg!(F, G, fwd)
        if !isnothing(G)
            G .= one(T) .- sqrt.(data.+bg)./sqrt.(max.(fwd .+ bg, max(bg, T(1f-10))))
        end
        return (!isnothing(F)) ? mapreduce(abs2, +, sqrt.(fwd.+bg) .- sqrt.(data.+bg); init=zero(T)) : T(0);
    end
    return fg!, 1
end

loss_anscombe_pos(data::AbstractArray{T}, fwd, bg=zero(T)) where {T} = sum(abs2.(sqrt.(max.(eltype(fwd)(0),fwd.+bg)) .- sqrt.(max.(eltype(fwd)(0),data.+bg))))

function get_fgC!(data::AbstractArray{T,N}, ::typeof(loss_anscombe_pos), bg=zero(T)) where {T,N}
    function fg!(F, G, fwd)
        if !isnothing(G)
            G .= one(T) .- sqrt.(max.(eltype(fwd)(0), data.+bg))./sqrt.(max.(fwd .+ bg, max(bg, T(1f-10))))
        end
        return (!isnothing(F)) ? mapreduce(abs2, +, sqrt.(max.(eltype(fwd)(0), fwd.+bg)) .- sqrt.(max.(eltype(fwd)(0),data.+bg)); init=zero(T)) : T(0);
    end
    return fg!, 1
end

