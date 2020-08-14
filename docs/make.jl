# https://juliadocs.github.io/Documenter.jl/stable/man/guide/#Package-Guide
# push!(LOAD_PATH,"../src/") 

# Run these locally to build docs/build folder:
# julia --color=yes --project=docs/ -e 'using Pkg; pkg\"add https://github.com/bankofcanada/TimeSeriesEcon.jl.git; add https://github.com/bankofcanada/ModelBaseEcon.jl.git\"'
# julia --color=yes --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'       
# julia --project=docs/ docs/make.jl

using Documenter, StateSpaceEcon

# Workaround for JuliaLang/julia/pull/28625
if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

makedocs(sitename = "StateSpaceEcon.jl",
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
         modules = [StateSpaceEcon],
         doctest = false,
         pages = [
        "Reference" => "index.md",

        
        # "Examples" => "examples.md"
    ]
)

deploydocs(
    repo = "github.com/bankofcanada/StateSpaceEcon.jl.git",
)