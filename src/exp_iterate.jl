struct JuliaIter{T}
    c::T
end

function Base.iterate(j::JuliaIter, z)
    r = z .* j.c
    (prod(r), r)
end
