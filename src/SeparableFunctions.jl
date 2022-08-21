module SeparableFunctions
using NDTools, LazyArrays

export separable_view, separable_create
export gaussian_sep, gaussian_sep_lz

export @define_separable # macro for defining new functions

DefaultResElType = Float32
DefaultArrType = Array{DefaultResElType}

include("general.jl")
include("specific.jl")

end # module SeparableFunctions
