using SeparableFunctions, Documenter 

 # set seed fixed for documentation
DocMeta.setdocmeta!(SeparableFunctions, :DocTestSetup, :(using SeparableFunctions); recursive=true)
makedocs(modules = [SeparableFunctions], 
         sitename = "SeparableFunctions.jl", 
         pages = Any[
            "SeparableFunctions.jl" => "index.md",
            "Concrete Separable Functions" => "separables.md",
            "Radial Functions" => "radial.md",
            "Utilities" => "utilities.md",
         ]
        )

deploydocs(repo = "github.com/bionanoimaging/SeparableFunctions.jl.git", devbranch="main")
