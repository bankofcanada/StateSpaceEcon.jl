##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################


module Kalman

using LinearAlgebra
using StaticArrays
using SparseArrays

using TimeSeriesEcon
using ModelBaseEcon
using ..StateSpaceEcon

using UnPack
using MacroTools

include("kalman/api.jl")
include("kalman/data.jl")
include("kalman/types.jl")
include("kalman/helpers.jl")
include("kalman/derbin_koopman.jl")
include("kalman/main.jl")
# include("kalman/filter.jl")
# include("kalman/smoother.jl")

# the model API 
export kf_length_x, kf_length_y, kf_is_linear
export kf_linear_model
export KFLinearModel
export @kfd_get, @kfd_set!, @kfd_view
export KFDataFilter, KFDataFilterEx, KFDataSmoother, KFDataSmootherEx
export KFilter
export kf_filter, kf_smoother


"""
API for using the functionality of the `$(@__MODULE__)` module with your model.

The model is most generally assumed to have the following structure.

x - vector of latent state variables
v - vector of state noise variables ~ N(0, Q)
y - vector of observed variables
w - vector of observation noise variables ~ N(0, R)

  y[t]   = H(x[t], w[t])
  x[t+1] = F(x[t], v[t])

We assume that the model is time-invariant, meaning that functions H and F, and
matrices Q and R do not depend on time `t`.

Users of the algorithms provided in this module must define their own model type
and implement the functions declared below. Make sure your methods specialize on
your model type.

* `kf_length_x(model, user_data...)` returns the number of state variables
* `kf_length_y(model, user_data...)` returns the number of observed variables
* `kf_is_linear(model, user_data...)` returns `true` if H and F are linear functions of `x` and `w`/`v`.

The linear version of the model is in the form.
  y[t]   = mu + H*x[t] + w[t]
  x[t+1] = F*x[t] + G*v[t]

Implementations specialized for linear models can use model type `KFLinearModel`
which has fields for `mu`, `H`, `F` and `G`. Users can define the following method
```
function kf_linear_model(model, user_data...)
    #
    # prepare mu, H, F, G, Q, and R
    #
    return KFLinearModel(mu, H, F, G, Q, R)
end


"""
Kalman


end
