# PANOC
#
# PART OF BAZINGA.jl

using Base.Iterators
using ProximalAlgorithms.IterationTools
using LinearAlgebra
using OptiMo
using Printf

###########################################################################
# Options
###########################################################################
struct PANOC_options{R<:Real}
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
struct PANOC_iterable{R<:Real,Tx<:AbstractArray{R},TP,TH,TO<:PANOC_options{R}}
    prob::TP                        # problem
    x0::Tx                          # initial point
    gamma::Maybe{R}                 # forward-backward stepsize
    H::TH                           # quasi-Newton object
    opts::TO                        # options
end

Base.IteratorSize(::Type{<:PANOC_iterable}) = Base.IsInfinite()

###########################################################################
# State
###########################################################################
mutable struct PANOC_state{R<:Real,Tx,TH}
    x::Tx                           # iterate
    f_x::R                          # value of smooth term at `x`
    grad_f_x::Tx                    # gradient of smooth term at `x`
    gamma::R                        # forward-backward stepsize
    y::Tx                           # forward point at `x`
    xbar::Tx                        # forward-backward point at `x`
    g_xbar::R                       # value of nonsmooth term at `xbar`
    res::Tx                         # fixed-point residual at `x`
    optim::R                        # optimality measure at `x`
    H::TH                           # quasi-Newton object
    tau::Maybe{R}                   # linesearch stepsize
    objec::R                        # (unscaled) objective function at `xbar`
    scalobj::R                      # scaling factor (> 0)
    fbe_x::R                        # (scaled) forward-backward envelope at `x`
    # additional allocation
    x_d::Tx                         # search direction at `x`, full step update
    x_old::Tx
    res_old::Tx
    xbar_old::Tx
end

###########################################################################
# Quadratic model
###########################################################################
f_model(state::PANOC_state) = f_model(
    state.scalobj * state.f_x,
    state.scalobj .* state.grad_f_x,
    state.res,
    state.gamma,
)

###########################################################################
# Initialize
###########################################################################
function Base.iterate(iter::PANOC_iterable{R}) where {R}
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
        xh = x .+ R(1)
        grad_f_xh = grad(iter.prob, xh)
        L = norm(scalobj .* (grad_f_xh - grad_f_x)) / R(sqrt(length(x)))
        L = max(1.0, L) # this avoids zero-division
        gamma = iter.opts.alpha / L
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
    objec = obj(iter.prob, xbar) + g_xbar

    # (scaled) forward-backward envelope
    f_xbar_upp = f_model(scalobj * f_x, scalobj .* grad_f_x, res, gamma)
    fbe_x = f_xbar_upp + scalobj * g_xbar

    state = PANOC_state(
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
    )

    return state, state
end

###########################################################################
# Step
###########################################################################
function Base.iterate(iter::PANOC_iterable{R}, state::PANOC_state{R,Tx}) where {R,Tx}

    # (scaled) quadratic model of `f` around `x` at `xbar`
    f_xbar_upp = f_model(state)

    # check and possibly backtrack stepsize `gamma`
    while iter.gamma === nothing || iter.opts.adaptive == true
        if state.gamma < iter.opts.gamma_min
            if iter.opts.warnings
                @warn "parameter `gamma` too small ($(state.gamma))"
            end
            return nothing
        end
        # (scaled) `f` at `xbar`
        f_xbar = state.scalobj * obj(iter.prob, state.xbar)
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
    end

    # (scaled) forward-backward envelope
    state.fbe_x = f_xbar_upp + state.scalobj * state.g_xbar

    # compute direction
    mul!(state.x_d, state.H, -state.res)

    # store iterate and residual for metric update later on
    state.x_old .= state.x
    state.res_old .= state.res

    # backtracking linesearch over the forward-backward envelope
    tau = R(1)

    state.x_d .+= state.x
    copyto!(state.x, state.x_d)
    copyto!(state.xbar_old, state.xbar)
    state.f_x = objgrad!(iter.prob, state.x, state.grad_f_x)

    sigma = iter.opts.sigma / state.gamma
    tol = iter.opts.bigeps * (1 + abs(state.fbe_x))
    rhs = state.fbe_x - sigma * norm(state.res)^2 + tol

    while true
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
            update!(state.H, state.x - state.x_old, state.res - state.res_old)
            break

        elseif tau < iter.opts.tau_min
            if !is_gamma_ok
                # small linesearch stepsize, with inappropriate stepsize `gamma`:
                # decrease `gamma`, which likely helps taking full steps.
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
            state.x .= tau .* state.x_d .+ (1 - tau) .* state.xbar_old
            state.f_x = objgrad!(iter.prob, state.x, state.grad_f_x)
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
mutable struct PANOC{R<:Real}
    memory::Int
    max_iter::Int
    tol_optim::R
    verbose::Bool
    freq::Int
    tol_diverging::R
    opts::PANOC_options

    function PANOC{R}(;
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
        @assert 0 < bigeps
        @assert 0 < tol_optim < 1 < tol_diverging
        @assert 0 < scaling_grad_max
        @assert scaling_grad_target === nothing || 0 < scaling_grad_target
        @assert 0 < scaling_min_value
        @assert 0 < gamma_min
        @assert 0 < tau_min
        sigma = beta * 0.5 * (1.0 - alpha)
        opts = PANOC_options(
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
function (solver::PANOC{R})(
    prob::TP;
    x0::AbstractArray{R} = prob.meta.x0,
    L::Maybe{R} = nothing,
) where {R,TP<:AbstractOptiModel}

    tstart = time()

    stop_solved(state::PANOC_state) = state.optim <= solver.tol_optim
    stop_unbound(state::PANOC_state) = state.objec < -solver.tol_diverging
    is_illegal(x) = x === nothing ? false : (any(isnan.(x)) || any(isinf.(x)))
    stop_illegal(state::PANOC_state) = is_illegal(state.x) || is_illegal(state.xbar)
    stop(state::PANOC_state) =
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

    iter = PANOC_iterable(prob, x0, gamma, LBFGS(x0, solver.memory), solver.opts)
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
        solver_name = "PANOC",
        solver = Dict(:gx => state.g_xbar),
    )
end

###########################################################################
# Outer constructors
###########################################################################
"""
    PANOC([gamma, adaptive, memory, max_iter, tol_optim, verbose, freq, ...])

Instantiate the PANOC algorithm [STS17] for solving optimization problems
of the form

    minimize f(x) + g(x),

where `f` is smooth and `g` is proximable. If `solver = PANOC(args...)`, then the
above problem is solved with

    solver( prob, [x0, L])

where `prob` is an OptiModel <: AbstractOptiModel.

Optional keyword arguments:

* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `alpha/L` if not set (but `L` is).
* `adaptive::Bool` (default: `false`), if true, forces the method stepsize to be adaptively adjusted even if `L` is provided (this behaviour is always enforced if `L` is not provided).
* `memory::Integer` (default: `5`), memory parameter for L-BFGS.
* `max_iter::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol_optim::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.
* `alpha::Real` (default: `0.95`), stepsize to inverse-Lipschitz-constant ratio; should be in (0, 1).
* `beta::Real` (default: `0.5`), sufficient decrease parameter; should be in (0, 1).

If `gamma` is not specified at construction time, the following keyword
argument can be used to set the stepsize parameter:

* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).

References:

[STS17] Stella, Themelis, Sopasakis, Patrinos, ``A simple and efficient algorithm
        for nonlinear model predictive control``, 56th IEEE Conference on Decision
        and Control (2017).
        DOI: 10.1109/CDC.2017.8263933
"""
PANOC(::Type{R}; kwargs...) where {R} = PANOC{R}(; kwargs...)
PANOC(; kwargs...) = PANOC(Float64; kwargs...)
