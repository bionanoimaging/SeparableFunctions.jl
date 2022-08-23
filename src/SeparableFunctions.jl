module SeparableFunctions
using NDTools, LazyArrays

export separable_view, separable_create
export gaussian_sep, gaussian_sep_lz
export propagator_col, propagator_col!
export @define_separable # macro for defining new functions

export calc_radial_symm!, calc_radial_symm, get_corner_ids

DefaultResElType = Float32
DefaultArrType = Array{DefaultResElType}
DefaultComplexArrType = Array{complex(DefaultResElType)}

include("general.jl")
include("specific.jl")
include("radial.jl")
include("docstrings.jl")

end # module SeparableFunctions
