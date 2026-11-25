##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2026, Bank of Canada
# All rights reserved.
##################################################################################

# Package quality checks, https://github.com/JuliaTesting/Aqua.jl
using Aqua

@testset "Aqua" begin
    Aqua.test_all(StateSpaceEcon;
        # StateSpaceEcon extends Base indexing of MVTSeries with model variables, and
        # transform/inverse_transform for matrices; these types are from the sibling packages
        piracies=(treat_as_own=[TimeSeriesEcon.MVTSeries, ModelBaseEcon.Model],),
    )
end
