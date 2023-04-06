##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################


### API

sf_factorize(A::SparseMatrixCSC)::Factorization = lu(A)::Factorization
sf_update!(F::Factorization, A::SparseMatrixCSC) = lu!(F, A)
sf_solve!(F::Factorization, b) = ldiv!(F, b)
sf_solve(F::Factorization, b) = (cb = copy(b); ldiv!(F, cb); cb)

