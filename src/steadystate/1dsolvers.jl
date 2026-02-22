##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

"""
    1dsolvers.jl 

This file contains solvers used in SteadyStateSolver. The functions here solve a
problem where a multivariate function is solved in a single coordinate, keeping
the other coordinates fixed. We have 1d- Newton and a variant of the bisection
method.

### Important note: 
These functions are used in pre-solving the steady state. They have a peculiar
feature that they fail if the first derivative is zero. Normally, if the
residual is zero, that would be a solution regardless of the first derivative.
However, in the context of presolving the steady state we have to consider that
the first derivative might be zero because the variable actually cancels out of
the equation. In that case, any value would result in zero residual, but we
can't consider that to be a solution. For this reason, the implementations in
this file deliberately fail if the first derivative is zero, even if the
residual is zero. """

# Above line intentionally left blank



"""
    newton1!(F, x, i; <options>)

Solve the equation `F(x) = 0` for `x[i]` keeping the other values of `x` fixed.
The input value of `x[i]` is used as the initial guess and it is updated in
place. Other entries of `x` are not accessed at all. Return `true` upon success,
or `false` otherwise.

### Function `F`
The function `F` must accept a single argument `x`, which is an array, and must
return a tuple of two things: the value of `F` (numeric scalar) and the gradient
of `F` (an array of the same shape as x). Only the i-th index of the gradient
array is used.

### Options
  * `maxiter = 5` - maximum number of iterations. The default is small (5)
    because the Newton method either converges very fast or doesn't converge at
    all.
  * `tol = 1e-8` - desired tolerance of the solution.

### Notes
The `tol` value is used for a stopping criterion and also for diagnosing
problems.

!!! warning
    This is an internal function used by the steady state solver. In the future
    it might be removed or modified.
"""
function newton1!(F::Function, vals::AbstractVector{T}, ind::Integer; maxiter=5, tol=sqrt(eps(T)))::Bool where {T<:AbstractFloat}
    fval, Jval = F(vals)
    err = tol * abs(fval) + tol
    for iter = 1:maxiter
        if abs(Jval[ind]) < tol || !isfinite(Jval[ind])
            return false
        end
        if abs(fval) < err
            return true
        end
        dx = fval / Jval[ind]
        vals[ind] -= dx
        fval, Jval = F(vals)
    end
    return abs(fval) < err
end


"""
    bisect!(F, x, i, dF; <options>)

Solve the equation `F(x) = 0` for `x[i]` keeping the other values of `x` fixed.
The input value of `x[i]` is used as the initial guess and it is updated in
place. Other entries of `x` are not accessed at all. Return `true` upon success,
or `false` otherwise.

### Arguments
  * `F` - A function that must accept a single argument `x`, which is an array,
    and must return the value of `F` (numeric scalar).
  * `x` - An array.
  * `i` - The index identifying the dimension in which we're solving the
    problem.
  * `dF` - A numeric value. This must equal the partial derivative of F with
    respect to `x[i]` at the input value of `x[i]`. This value is used to
    construct the initial interval in which the bisection method will be
    applied.

### Options
  * `maxiter = 500` - maximum number of iterations. The default is large (500)
    because this method sometimes converges slowly.
  * `tol = 1e-8` - desired tolerance of the solution.

!!! warning
    This is an internal function used by the steady state solver. In the future
    it might be removed or modified.
"""
function bisect!(F::Function, vals::AbstractVector{T}, ind::Int64, deriv::T; maxiter=500, tol=sqrt(eps(T)))::Bool where {T<:AbstractFloat}
    # This code is a mess. Someone please clean it up!

    # Convenience wrapper - evaluate F as if it were a univariate function
    # and trap any errors (e.g. inadmissible arguments) returning NaN.
    # With each call, the current value of vals[i] is updated in place.
    f(x) = (vals[ind] = x; try
        F(vals)
    catch
        NaN
    end)

    """
        _do_bisect(x0, f0, x1, f1)

    Perform the iteration of the bisection method.

    The two initial points are assumed to bracket the solution, i.e. f0 and f1
    have different signs. On each iteration we replace one of the endpoints with
    the midpoint, always maintaining that property.

    Return `true` if either the length of the interval shrinks to the machine
    accuracy of if the function value is within `tol` of zero. Return `false`
    otherwise.

    Note that here we call the function f(x) defined above. In particular, the
    value of x that solves the problem is written directly into `vals[i]`.
    """
    function _do_bisect(x0, f0, x1, f1)
        for it = 1:maxiter
            abs(x0 - x1) <= eps(x0) && return true
            xp = 0.5 * (x1 + x0)
            fp = f(xp)
            abs(fp) < tol && return true
            if fp * f0 < 0.0
                x1 = xp
                f1 = fp
            else
                x0 = xp
                f0 = fp
            end
        end
        return f(0.5 * (x0 + x1)) < tol
    end

    #######################################################################
    # The problem is to find two points that bracket the solution, i.e. such 
    # that f(x0) and f(x1) have opposite signs. 

    # With the sign of f0 and the known derivative at x0 we can search for x1 
    # in the direction in which the function approaches zero.)
    # if the derivative is zero at the initial point, that's an argument error
    abs(deriv) < 1e-10 && return false
    !isfinite(deriv) && return false

    # First point - that's easy
    x0 = vals[ind]
    f0 = f(x0)
    # is x0 the solution? 
    abs(f0) < tol && return true
    isnan(f0) && return false

    # Try a line search in the Newton direction
    step = -f0 / deriv
    x1 = x0 + step
    f1 = f(x1)
    while isnan(f1) || abs(f1) > 1e5 * abs(f0)
        # If x1 is inadmissible or too far off to infinity try a smaller step
        step *= 0.5
        step < eps(T) && return false
        x1 = x0 + step
        f1 = f(x1)
    end
    # is x1 the solution?
    abs(f1) < tol && return true
    # does (x0, x1) bracket the solution? 
    f1 * f0 < 0.0 && return _do_bisect(x0, f0, x1, f1)

    # We have two points (x0, f0) and (x1, f1), but f0 and f1 are of the same sign. 
    # It may be that there's a root between them, but we overshot, or that 
    # the root is beyond x1 (from x0)

    # try the midpoint
    x2 = 0.5 * (x0 + x1)
    f2 = f(x2)
    # if x2 the solution? 
    abs(f2) < tol && return true
    # does (x0, x2) bracket the solution?
    if f0 * f2 < 0.0
        # yes, then (x1, x2) also brackets the solution. Hmmm!
        return abs(f0) < abs(f1) ? _do_bisect(x0, f0, x2, f2) : _do_bisect(x1, f1, x2, f2)
    end

    # We have three points with the same sign of f. 
    # Desperately try a few iterations of inverse quadratic interpolation.
    for i = 1:20
        # Approximate f-inverse with a quadratic through the three points we have and set x3 = f_inverse(0)
        x3 = f0 * f1 * x2 / (f2 - f0) / (f2 - f1) + f0 * x1 * f2 / (f1 - f0) / (f1 - f2) + x0 * f1 * f2 / (f0 - f1) / (f0 - f2)
        f3 = f(x3)
        # is it a bad point? 
        (isnan(f3) || isinf(f3)) && break
        # is x3 a solution? 
        abs(f3) < tol && return true
        # is x3 on the opposite side of the root?
        (f0 * f3 < 0.0) && return _do_bisect(x0, f0, x3, f3)
        f0, f1, f2 = f1, f2, f3
        x0, x1, x2 = x1, x2, x3
    end
    # all else failed, try brute-force interval search
    step = -sign(f0) * sign(deriv) * 0.01
    for it = 1:30
        step *= 2.0
        x1 = x0 + step
        f1 = f(x1)
        abs(f1) < tol && return true
        f0 * f1 < 0.0 && return _do_bisect(x0, f0, x1, f1)
    end
    # we failed miserably! :-(
    return false
end

