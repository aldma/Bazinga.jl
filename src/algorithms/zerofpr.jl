# ZEROFPR
#
# PART OF BAZINGA.jl

using Base.Iterators
using ProximalAlgorithms.IterationTools
using LinearAlgebra: norm
using OptiMo
using Printf

###########################################################################
# Options
###########################################################################
struct ZEROFPR_options{R<:Real}
    alpha::R                        # ∈ (0, 1), e.g.: 0.95
    beta::R                         # ∈ (0, 1), e.g.: 0.5
    sigma::R                        # = beta * (1 - alpha) / 2 ∈ (0, 0.5)
    adaptive::Bool                  # enforce adaptive stepsize even if L is provided
    tau_min::R                      # minimum linesearch stepsize
    gamma_min::R                    # minimum forward-backward stepsize
    bigeps::R                       # small value used for tolerances
    grad_scaling::Bool              # perform gradient-based scaling?
    scaling_grad_max::R             # maximum gradient after scaling
    scaling_grad_target::Maybe{R}   # target value of gradient after scaling
    scaling_min_value::R            # minimum value of gradient-based scaling factor
    warnings::Bool                  # can print warnings?
end

###########################################################################
# Iterable
###########################################################################
struct ZEROFPR_iterable{R<:Real,Tx<:AbstractArray{R},TP,TH,TO<:ZEROFPR_options{R}}
    prob::TP                        # problem
    x0::Tx                          # initial point
    gamma::Maybe{R}                 # forward-backward stepsize
    H::TH                           # quasi-Newton object
    opts::TO                        # options
end

Base.IteratorSize(::Type{<:ZEROFPR_iterable}) = Base.IsInfinite()

###########################################################################
# State
###########################################################################
mutable struct ZEROFPR_state{R<:Real,Tx,TH}
    x::Tx                   # iterate
    f_x::R                  # (unscaled) value of smooth term
    grad_f_x::Tx            # (unscaled) gradient of smooth term
    gamma::R                # stepsize parameter of forward and backward steps
    y::Tx                   # forward point
    xbar::Tx                # forward-backward point
    g_xbar::R               # (unscaled) value of nonsmooth term (at xbar)
    res::Tx                 # fixed-point residual at iterate (= x - xbar)
    optim::R                # optimality measure
    H::TH                   # variable metric
    tau::Maybe{R}           # linesearch stepsize
    objec::R                # (unscaled) objective function at `xbar`
    scalobj::R              # scaling factor (> 0)
    fbe_x::R                # (scaled) forward-backward envelope at `x`
    # additional allocation
    grad_f_xbar::Tx
    xbarbar::Tx
    res_xbar::Tx
    d::Tx                   # search direction at `xbar`
    xbar_old::Maybe{Tx}
    res_xbar_old::Maybe{Tx}
end

###########################################################################
# Quadratic model
###########################################################################
f_model(state::ZEROFPR_state) = f_model(
    state.scalobj * state.f_x,
    state.scalobj .* state.grad_f_x,
    state.res,
    state.gamma,
)

###########################################################################
# Initialize
###########################################################################
function Base.iterate(iter::ZEROFPR_iterable{R}) where {R}
    # initial point
    x = copy(iter.x0)
    g_xbar, x = objprox(iter.prob, x, iter.opts.bigeps)
    f_x, grad_f_x = objgrad(iter.prob, x)
    # scaling
    if iter.opts.grad_scaling
        scalobj = objgradscaling(
            grad_f_x,
            iter.opts.scaling_grad_target,
            iter.opts.scaling_grad_max,
            iter.opts.scaling_min_value,
        )
    else
        scalobj = R(1)
    end
    # forward-backward stepsize
    gamma = iter.gamma
    if gamma === nothing
        # compute lower bound to Lipschitz constant of x ↦ ∇f(x)
        xh = x .+ 0.01 .* (x .+ R(1))
        grad_f_xh = grad(iter.prob, xh)
        L = norm(scalobj .* (grad_f_xh - grad_f_x)) / R(sqrt(length(x)))
        L = max(1.0, L) # this avoids zero-division
        gamma = max(iter.opts.alpha / L, sqrt(iter.opts.gamma_min))
    end

    # initial forward-backward step
    y = x - gamma .* (scalobj .* grad_f_x)
    g_xbar, xbar = objprox(iter.prob, y, gamma * scalobj)

    # initial fixed-point residual and optimality
    res = x - xbar
    #optim = norm( res, Inf ) / gamma
    f_xbar, tmp = objgrad(iter.prob, xbar) # grad_f_xbar
    tmp .+= (res ./ gamma) .- grad_f_x
    optim = norm(tmp, Inf)

    # initial objective
    objec = f_xbar + g_xbar

    # forward-backward envelope
    f_xbar_upp = f_model(scalobj * f_x, scalobj .* grad_f_x, res, gamma)
    fbe_x = f_xbar_upp + scalobj * g_xbar

    state = ZEROFPR_state(
        x,
        f_x,
        grad_f_x,
        gamma,
        y,
        xbar,
        g_xbar,
        res,
        optim,
        iter.H,
        nothing,
        objec,
        scalobj,
        fbe_x,
        zero(x),
        zero(x),
        zero(x),
        zero(x),
        nothing,
        nothing,
    )

    return state, state
end

###########################################################################
# Step
###########################################################################
function Base.iterate(iter::ZEROFPR_iterable{R}, state::ZEROFPR_state{R,Tx}) where {R,Tx}

    # (scaled) quadratic model of `f` around `x` at `xbar`
    f_xbar_upp = f_model(state)

    # `f` at `xbar`
    f_xbar = objgrad!(iter.prob, state.xbar, state.grad_f_xbar)
    f_xbar *= state.scalobj

    # check and possibly backtrack stepsize `gamma`
    while iter.gamma === nothing || iter.opts.adaptive == true
        if state.gamma < iter.opts.gamma_min
            if iter.opts.warnings
                @warn "stepsize `gamma` too small ($(state.gamma))"
            end
            return nothing
        end
        tol = iter.opts.bigeps * (1 + abs(f_xbar))
        if f_xbar <= f_xbar_upp + tol
            break
        end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* (state.scalobj .* state.grad_f_x)
        state.g_xbar = objprox!(iter.prob, state.y, state.gamma * state.scalobj, state.xbar)
        state.res .= state.x .- state.xbar
        reset!(state.H)
        f_xbar_upp = f_model(state)
        f_xbar = objgrad!(iter.prob, state.xbar, state.grad_f_xbar)
        f_xbar *= state.scalobj
    end

    # (scaled) forward-backward envelope
    state.fbe_x = f_xbar_upp + state.scalobj * state.g_xbar

    # residual at `xbar`
    state.y .= state.xbar .- state.gamma .* (state.scalobj .* state.grad_f_xbar)
    g_xbarbar = objprox!(iter.prob, state.y, state.gamma * state.scalobj, state.xbarbar)
    state.res_xbar .= state.xbar .- state.xbarbar

    if state.xbar_old === nothing
        # store vectors for next update
        state.xbar_old = copy(state.xbar)
        state.res_xbar_old = copy(state.res_xbar)
    else
        # update variable metric
        update!(state.H, state.xbar - state.xbar_old, state.res_xbar - state.res_xbar_old)
        # store vectors for next update
        copyto!(state.xbar_old, state.xbar)
        copyto!(state.res_xbar_old, state.res_xbar)
    end

    # search direction
    mul!(state.d, state.H, -state.res_xbar)

    # linesearch over the forward-backward envelope
    tau = R(1)
    sigma = iter.opts.sigma / state.gamma
    tol = iter.opts.bigeps * (1 + abs(state.fbe_x))
    rhs = state.fbe_x - sigma * norm(state.res)^2 + tol

    while true
        state.x .= state.xbar_old .+ tau .* state.d
        state.f_x = objgrad!(iter.prob, state.x, state.grad_f_x)
        state.y .= state.x .- state.gamma .* (state.scalobj .* state.grad_f_x)
        state.g_xbar = objprox!(iter.prob, state.y, state.gamma * state.scalobj, state.xbar)
        state.res .= state.x .- state.xbar
        f_xbar_upp = f_model(state)
        state.fbe_x = f_xbar_upp + state.scalobj * state.g_xbar

        f_xbar = obj(iter.prob, state.xbar)
        f_xbar *= state.scalobj
        tol = iter.opts.bigeps * (1 + abs(f_xbar))
        is_gamma_ok = f_xbar <= f_xbar_upp + tol

        if (state.fbe_x <= rhs) && is_gamma_ok
            break

        elseif tau < iter.opts.tau_min
            if !is_gamma_ok
                state.gamma *= 0.5
                reset!(state.H)
            end
            objec_t = f_xbar + state.g_xbar
            tol = iter.opts.bigeps * (1 + abs(state.objec))
            if objec_t <= state.objec + tol
                # small linesearch stepsize : taking tau > 0, which gives decrease
                # in the objective function.
                break
            end
            # small linesearch stepsize : taking `tau = 0`, which gives decrease in
            # the FBE.
            tau = 0.0
            copyto!(state.x, state.xbar_old)
            state.f_x = objgrad!(iter.prob, state.x, state.grad_f_x)
            state.y .= state.x .- state.gamma .* (state.scalobj .* state.grad_f_x)
            state.g_xbar =
                objprox!(iter.prob, state.y, state.gamma * state.scalobj, state.xbar)
            state.res .= state.x .- state.xbar
            state.fbe_x = f_model(state) + state.scalobj * state.g_xbar
            break

        else
            tau *= 0.5
        end
    end

    state.tau = tau
    f_xbar, tmp = objgrad(iter.prob, state.xbar) # grad_f_xbar
    tmp .+= (state.res ./ state.gamma) .- state.grad_f_x
    state.optim = norm(tmp, Inf)
    state.objec = f_xbar + state.g_xbar
    return state, state
end

###########################################################################
# Solver
###########################################################################
mutable struct ZEROFPR{R<:Real}
    memory::Int
    max_iter::Int
    tol_optim::R
    verbose::Bool
    freq::Int
    tol_diverging::R
    opts::ZEROFPR_options

    function ZEROFPR{R}(;
        alpha::R = R(0.95),
        beta::R = R(0.5),
        adaptive::Bool = true,
        memory::Int = 8,
        max_iter::Int = 10000,
        tol_optim::R = R(1e-8),
        verbose::Bool = false,
        freq::Int = 10,
        tau_min::R = R(1e-8),
        gamma_min::R = R(1e-12),
        grad_scaling::Bool = true,
        scaling_grad_max::R = R(100),
        scaling_grad_target::Maybe{R} = nothing,
        scaling_min_value::R = R(1e-8),
        tol_diverging::R = R(1e20),
        bigeps::R = R(1e-12),
        warnings::Bool = true,
    ) where {R}
        @assert 0 < alpha < 1
        @assert 0 < beta < 1
        @assert 0 <= memory
        @assert 0 < max_iter
        @assert 0 < freq
        @assert 0 < bigeps < 1
        @assert 0 < tol_optim < 1 < tol_diverging
        @assert 0 < scaling_grad_max
        @assert scaling_grad_target === nothing || 0 < scaling_grad_target
        @assert 0 < scaling_min_value < 1
        @assert 0 < gamma_min < 1
        @assert 0 < tau_min < 1
        sigma = beta * 0.5 * (1.0 - alpha)
        opts = ZEROFPR_options(
            alpha,
            beta,
            sigma,
            adaptive,
            tau_min,
            gamma_min,
            bigeps,
            grad_scaling,
            scaling_grad_max,
            scaling_grad_target,
            scaling_min_value,
            warnings,
        )
        new(memory, max_iter, tol_optim, verbose, freq, tol_diverging, opts)
    end
end

###########################################################################
# Solver call
###########################################################################
function (solver::ZEROFPR{R})(
    prob::TP;
    x0::AbstractArray{R} = prob.meta.x0,
    L::Maybe{R} = nothing,
) where {R,TP<:AbstractOptiModel}

    tstart = time()

    stop_solved(state::ZEROFPR_state) = state.optim <= solver.tol_optim
    stop_unbound(state::ZEROFPR_state) = state.objec < -solver.tol_diverging
    is_illegal(x) = x === nothing ? false : (any(isnan.(x)) || any(isinf.(x)))
    stop_illegal(state::ZEROFPR_state) = is_illegal(state.x) || is_illegal(state.xbar)
    stop(state::ZEROFPR_state) =
        stop_solved(state) || stop_unbound(state) || stop_illegal(state)

    if solver.verbose
        @printf(
            "%6s | %8s | %8s  %9s | %9s  %9s\n",
            "iter",
            "objec",
            "fbe",
            "optim",
            "gamma",
            "tau"
        )
        disp((i, state)) = @printf(
            "%6d | %+.1e | %+.1e  %.3e | %.3e  %.3e \n",
            i,
            state.objec,
            state.fbe_x,
            state.optim,
            state.gamma,
            (state.tau === nothing ? 0.0 : state.tau)
        )
    end

    gamma = (L === nothing ? nothing : solver.opts.alpha / L)

    iter = ZEROFPR_iterable(prob, x0, gamma, LBFGS(x0, solver.memory), solver.opts)
    iter = halt(iter, stop)
    iter = take(iter, solver.max_iter)
    iter = enumerate(iter)
    if solver.verbose
        iter = sample(iter, solver.freq)
        iter = tee(iter, disp)
    end

    num_iters, state = loop(iter)

    status = if stop_solved(state)
        :first_order
    elseif stop_unbound(state)
        :unbounded
    elseif stop_illegal(state)
        :ieee_nan_inf
    elseif (num_iters >= solver.max_iter)
        :max_iter
    elseif (state.gamma <= solver.opts.gamma_min)
        :stalled
    else
        :unknown
    end

    return OptiOutput(
        status,
        x = state.xbar,
        objective = state.objec,
        optimality = state.optim,
        iterations = num_iters,
        time = time() - tstart,
        solver_name = "ZEROFPR",
        solver = Dict(:gx => state.g_xbar),
    )
end

###########################################################################
# Outer constructors
###########################################################################
"""
    ZEROFPR([gamma, adaptive, memory, max_iter, tol_optim, verbose, freq, ...])

Instantiate the ZeroFPR algorithm [TSP18] for solving optimization problems
of the form

    minimize f(x) + g(x),

where `f` is smooth and `g` is proximable. If `solver = ZEROFPR(args...)`, then the
above problem is solved with

    solver( prob, [x0, L] )

Optional keyword arguments:

* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `alpha/L` if not set (but `L` is).
* `adaptive::Bool` (default: `true`), if true, forces the method stepsize to be adaptively adjusted even
    if `L` is provided (this behaviour is always enforced if `L` is not provided).
* `memory::Integer` (default: `5`), memory parameter for L-BFGS.
* `max_iter::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol_optim::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.
* `alpha::Real` (default: `0.95`), stepsize to inverse-Lipschitz-constant ratio; should be in (0, 1).
* `beta::Real` (default: `0.5`), sufficient decrease parameter; should be in (0, 1).
* `grad_scaling::Bool`
* `scaling_grad_max::Real`
* `scaling_grad_target::Maybe{Real}`
* `scaling_min_value::Real`

If `gamma` is not specified at construction time, the following keyword
argument can be used to set the stepsize parameter:

* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).

References:

[TSP18] Themelis, Stella, Patrinos, ``Forward-backward envelope for the sum of two
        nonconvex functions: Further properties and nonmonotone line-search algorithms``,
        SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274–2303 (2018).
        DOI: 10.1137/16M1080240
"""
ZEROFPR(::Type{R}; kwargs...) where {R} = ZEROFPR{R}(; kwargs...)
ZEROFPR(; kwargs...) = ZEROFPR(Float64; kwargs...)
