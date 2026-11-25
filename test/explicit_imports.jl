##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2026, Bank of Canada
# All rights reserved.
##################################################################################

# Import hygiene checks, https://github.com/JuliaTesting/ExplicitImports.jl
using ExplicitImports

@testset "ExplicitImports" begin
    test_explicit_imports(StateSpaceEcon;
        # Plans keeps `using TimeSeriesEcon`: it evaluates strings read from plan files
        # (ranges such as 2020Q1:2021Q4, frequencies) in its own namespace
        no_implicit_imports=(ignore=(StateSpaceEcon.Plans,),),
        # mean is used in `√mean(...)` (dfm/em.jl), which the analyzer does not see
        no_stale_explicit_imports=(ignore=(:mean,),),
        # libblastrampoline is owned by libblastrampoline_jll, which is not a dependency
        all_qualified_accesses_via_owners=(ignore=(:libblastrampoline,),),
        # use of non-public names is checked later
        all_explicit_imports_are_public=false,
        all_qualified_accesses_are_public=false,
    )
end
