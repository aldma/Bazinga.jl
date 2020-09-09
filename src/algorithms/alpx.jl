# ALPX
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
struct ALPX_options{R<:Real}
    tol_cviol::R
    tol_cviol_sqrt::R
    tol_optim_sqrt::R
    dual_reset::Bool
    dual_grow::R
    y_min::R
    y_max::R
    y_min_min::R
    y_max_max::R
    μ_min::R
    μ_max::R
    μ_est::R
    μ_down::R
    μ_grow::R
    μ_slow::R
    θ::R
    ϵ_min::R
    ϵ_init::R
    ϵ_down::R
    ϵ_rel::R
    max_sub_iter::Int
    subsolver::Symbol
    subsolver_verbose::Bool
    subsolver_freq::Int
    subsolver_gamma_min::R
    subsolver_tau_min::R
    bigeps::R
    warnings::Bool
end

###########################################################################
# Iterable
###########################################################################
struct ALPX_iterable{R<:Real,Tx<:AbstractArray{R},TP,TO<:ALPX_options{R}}
    prob::TP                        # problem
    x0::Tx                          # initial primal point
    y0::Tx                          # initial dual point
    μ::Maybe{Tx}                    # penalty parameters
    opts::TO                        # options
end

Base.IteratorSize(::Type{<:ALPX_iterable}) = Base.IsInfinite()

###########################################################################
# State
###########################################################################
mutable struct ALPX_state{R<:Real,Tx<:AbstractArray{R},TS,TA<:AugLagOptiModel{R}}
    x::Tx                   # primal iterate
    fx::R                   # value of smooth term
    gx::R                   # value of proximable term
    objec::R                # value of objective function
    cx::Tx                  # value of constraints
    y::Tx                   # dual iterate
    μ::Tx                   # penalty parameter vector
    optim::Maybe{R}         # optimality/criticality
    cviol::R                # constraint violation
    cslack::R               # compl. slackness
    # additional allocation
    px::Tx                  # constraint violation vector
    w::Tx                   # compl. slackness vector : cx - proj_S( cx + μ .* y )
    z::Tx                   # dual step vector : w ./ μ
    μ_zero::R               # slowly vanishing penalty parameter
    y_min::R                # lower dual safeguard
    y_max::R                # upper dual safeguard
    ϵsub::R                 # inner tolerance
    sx::Tx                  # infeasibility stationarity vector
    infsta::R               # infeasibility stationarity
    cslack_old::Maybe{R}    # previous compl. slackness
    sub_iter::Int           # subsolver iterations
    sub_flag::Bool          # subproblem successful
    subsolver::TS           # subsolver
    alprob::TA              # augmented Lagrangian subproblem
end

###########################################################################
# Initialize
###########################################################################
function Base.iterate(iter::ALPX_iterable{R}) where {R}
    # initial point
    x = copy(iter.x0)
    y = copy(iter.y0)
    gx, x = objprox(iter.prob, x, iter.opts.bigeps)
    fx = obj(iter.prob, x)
    objec = fx + gx
    y_min = iter.opts.y_min
    y_max = iter.opts.y_max
    dual_safeguard!(y, y_min, y_max, iter.opts)
    # constraint violation
    cx = cons(iter.prob, x)
    w = proj(iter.prob, cx) # tmp
    px = cx - w
    cviol = norm(px, Inf)
    # allocation
    z = similar(cx)
    sx = similar(x)
    # infeasibility stationarity
    infeas_station!(iter.prob, x, px, iter.opts.bigeps, sx)
    infsta = norm(sx, Inf)
    # penalty parameter vector [BiM12]
    μ_zero = copy(iter.opts.μ_max)
    μ = iter.μ
    if μ === nothing
        μ = similar(px)
        penalty_estimate!(objec, px, μ, iter.opts.μ_est)
    end
    μ .= max.(iter.opts.μ_min, min.(μ, iter.opts.μ_max))
    # complementarity slackness
    compl_slackness!(iter.prob, cx, μ, y, w, z)
    cslack = norm(w, Inf)
    # subproblem tolerance
    ϵsub = max(iter.opts.ϵ_min, iter.opts.ϵ_init)
    # subproblem solver
    if iter.opts.subsolver == :zerofpr
        subsolver = Bazinga.ZEROFPR(
            max_iter = iter.opts.max_sub_iter,
            verbose = iter.opts.subsolver_verbose,
            freq = iter.opts.subsolver_freq,
            gamma_min = iter.opts.subsolver_gamma_min,
            tau_min = iter.opts.subsolver_tau_min,
        )
    else
        subsolver = Bazinga.PANOC(
            max_iter = iter.opts.max_sub_iter,
            verbose = iter.opts.subsolver_verbose,
            freq = iter.opts.subsolver_freq,
            gamma_min = iter.opts.subsolver_gamma_min,
            tau_min = iter.opts.subsolver_tau_min,
        )
    end

    # augmented Lagrangian subproblem
    alprob = AugLagOptiModel(iter.prob, μ, y)

    state = ALPX_state(
        x,
        fx,
        gx,
        objec,
        cx,
        y,
        μ,
        nothing,
        cviol,
        cslack,
        px,
        w,
        z,
        μ_zero,
        y_min,
        y_max,
        ϵsub,
        sx,
        infsta,
        nothing,
        -1,
        false,
        subsolver,
        alprob,
    )

    return state, state
end

###########################################################################
# Step
###########################################################################
function Base.iterate(iter::ALPX_iterable{R}, state::ALPX_state{R,Tx}) where {R,Tx}
    # update the augmented Lagrangian subproblem
    AugLagUpdate!(state.alprob, state.μ, state.y)
    # update and call subsolver
    state.subsolver.tol_optim = state.ϵsub
    sub_out = state.subsolver(state.alprob, x0 = state.x)
    state.sub_iter = sub_out.iterations
    state.sub_flag = (sub_out.optimality <= state.ϵsub)
    # primal update
    copyto!(state.x, sub_out.x)
    # objective
    state.gx = sub_out.solver[:gx]
    state.fx = obj(iter.prob, state.x)
    state.objec = state.fx + state.gx
    # (outer) optimality coincides with subproblem optimality because of the
    # first-order dual update [LeV20]
    state.optim = sub_out.optimality
    # constraint violations
    cons!(iter.prob, state.x, state.cx)
    proj!(iter.prob, state.cx, state.w) # tmp
    state.px .= state.cx .- state.w
    state.cviol = norm(state.px, Inf)
    # infeasibility stationarity
    infeas_station!(iter.prob, state.x, state.px, iter.opts.bigeps, state.sx)
    state.infsta = norm(state.sx, Inf)
    # complementarity slackness
    compl_slackness!(iter.prob, state.cx, state.μ, state.y, state.w, state.z)
    state.cslack = norm(state.w, Inf)
    # safeguarded dual update
    state.y .+= state.z
    dual_safeguard!(state.y, state.y_min, state.y_max, iter.opts)
    # penalty parameters
    state.μ_zero *= iter.opts.μ_slow
    if state.cslack_old === nothing
        if state.cviol > 0
            penalty_estimate!(state.objec, state.px, state.μ, iter.opts.μ_est)
        end
    elseif (state.cslack <= iter.opts.tol_cviol) && (state.cviol <= iter.opts.tol_cviol)
        state.μ *= iter.opts.μ_grow
        state.μ_zero = min(state.μ_zero, maximum(state.μ))
    elseif (state.cslack <= iter.opts.θ * state.cslack_old)
        # do nothing
    else
        state.μ .*= iter.opts.μ_down
    end
    state.μ .= max.(iter.opts.μ_min, min.(state.μ, state.μ_zero))
    state.cslack_old = copy(state.cslack)
    # subproblem tolerance
    if state.sub_flag &&
       (state.optim <= iter.opts.tol_optim_sqrt) &&
       (state.cslack <= iter.opts.tol_cviol_sqrt) &&
       (state.cviol <= iter.opts.tol_cviol_sqrt)
        state.ϵsub *= iter.opts.ϵ_down
        state.ϵsub = max(iter.opts.ϵ_min, min(state.ϵsub, iter.opts.ϵ_rel * state.optim))
    end

    return state, state
end

"""
    penalty_estimate!( objec, cviol, μ, scale )
Estimates suitable penalty parameters balancing objective `objec` and constraint
violations `cviol` [BiM12].
"""
function penalty_estimate!(
    objec::R,
    cviol::AbstractVector{R},
    μ::AbstractVector{R},
    scale::R,
) where {R<:Real}
    μ .= scale .* max.(1.0, 0.5 .* abs2.(cviol)) ./ max(1.0, abs(objec))
end

"""
    infeas_station!( prob, x, px, bigeps, sx )
Computes stationarity of the infeasibility measure.
"""
function infeas_station!(
    prob::TP,
    x::AbstractVector{R},
    px::AbstractVector{R},
    bigeps::R,
    sx::AbstractVector{R},
) where {R<:Real,TP}
    jtprod!(prob, x, px, sx) # tmp
    sx .= x .- bigeps .* sx # tmp
    prox!(prob, sx, bigeps, sx) # tmp
    sx .= (x .- sx) ./ bigeps
end

"""
    compl_slackness!( prob, cx, μ, y, w, z )
"""
function compl_slackness!(
    prob::TP,
    cx::AbstractVector{R},
    μ::AbstractVector{R},
    y::AbstractVector{R},
    w::AbstractVector{R},
    z::AbstractVector{R},
) where {R<:Real,TP}
    w .= cx .+ μ .* y
    proj!(prob, w, z)
    w .= cx .- z
    z .= w ./ μ
end

"""
    dual_safeguard!( y, ymin, ymax, opts )
If an entry of the dual estimate `y` is out of bounds `ymin` and `ymax`, projects
it, or reset it to zero, and expand bounds, within their limit [BiM14].
"""
function dual_safeguard!(
    y::AbstractVector{R},
    ymin::R,
    ymax::R,
    opts::TO,
) where {R<:Real,TO<:ALPX_options}
    idy = (y .<= ymin)
    if any(idy)
        y[idy] .= opts.dual_reset ? R(0) : ymin
        ymin *= opts.dual_grow
        ymin = max(opts.y_min_min, ymin)
    end
    idy = (y .>= ymax)
    if any(idy)
        y[idy] .= opts.dual_reset ? R(0) : ymax
        ymax *= opts.dual_grow
        ymax = min(opts.y_max_max, ymax)
    end
end

###########################################################################
# Solver
###########################################################################
struct ALPX{R<:Real}
    max_iter::Int
    max_sub_iter::Int
    tol_optim::R
    tol_cviol::R
    tol_cviol_sqrt::R
    tol_inf_mu::R
    tol_diverging::R
    bigeps::R
    verbose::Bool
    freq::Int
    opts::ALPX_options{R}

    function ALPX{R}(;
        max_iter::Int = 100,
        max_sub_iter::Int = 10000,
        tol_optim::R = R(1e-8),
        tol_cviol::R = R(1e-8),
        theta::R = R(0.5),
        mu_down::R = R(0.25),
        eps_down::R = R(0.1),
        verbose::Bool = false,
        freq::Int = 1,
        dual_reset::Bool = true,
        dual_grow::R = R(1.1),
        mu_min::R = R(1e-12),
        mu_max::R = R(1e6),
        mu_est::R = R(0.1),
        mu_grow::R = R(10),
        mu_slow::R = R(0.99),
        y_min::R = R(-1e3),
        y_max::R = R(1e3),
        y_min_min::R = R(-1e6),
        y_max_max::R = R(1e6),
        eps_min::R = min(tol_optim, tol_cviol),
        eps_init::R = sqrt(tol_optim),
        eps_rel::R = R(0.5),
        tol_diverging::R = R(1e20),
        subsolver::Symbol = :zerofpr,
        subsolver_verbose::Bool = false,
        subsolver_freq::Int = 10,
        subsolver_gamma_min::R = 1e-16,
        subsolver_tau_min::R = 1e-8,
        bigeps::R = R(1e-12),
        warnings::Bool = true,
    ) where {R}
        @assert 0 < max_iter
        @assert 0 < max_sub_iter
        @assert 0 < bigeps
        @assert 0 < tol_optim < 1 < tol_diverging
        @assert 0 < tol_cviol < 1
        @assert 0 < theta < 1
        @assert 0 < mu_down < 1
        @assert 0 < freq
        @assert 1 <= dual_grow
        @assert 0 < mu_min < 1
        @assert mu_min < mu_max
        @assert 0 < mu_est
        @assert 0 < mu_slow < 1 <= mu_grow
        @assert y_min_min <= y_min <= 0 <= y_max <= y_max_max
        @assert 0 < eps_min <= tol_optim
        @assert eps_min <= eps_init
        @assert 0 < eps_rel < 1
        @assert 0 < eps_down < 1
        @assert subsolver ∈ [:zerofpr, :panoc]
        @assert 0 < subsolver_freq
        @assert 0 < subsolver_gamma_min
        @assert 0 < subsolver_tau_min
        tol_optim_sqrt = sqrt(tol_optim)
        tol_cviol_sqrt = sqrt(tol_cviol)
        tol_inf_mu = cbrt(mu_min^2)
        opts = ALPX_options(
            tol_cviol,
            tol_cviol_sqrt,
            tol_optim_sqrt,
            dual_reset,
            dual_grow,
            y_min,
            y_max,
            y_min_min,
            y_max_max,
            mu_min,
            mu_max,
            mu_est,
            mu_down,
            mu_grow,
            mu_slow,
            theta,
            eps_min,
            eps_init,
            eps_down,
            eps_rel,
            max_sub_iter,
            subsolver,
            subsolver_verbose,
            subsolver_freq,
            subsolver_gamma_min,
            subsolver_tau_min,
            bigeps,
            warnings,
        )

        new(
            max_iter,
            max_sub_iter,
            tol_optim,
            tol_cviol,
            tol_cviol_sqrt,
            tol_inf_mu,
            tol_diverging,
            bigeps,
            verbose,
            freq,
            opts,
        )
    end
end

###########################################################################
# Solver call
###########################################################################
function (solver::ALPX{R})(
    prob::TP;
    x0::AbstractArray{R} = prob.meta.x0,
    y0::AbstractArray{R} = prob.meta.y0,
    mu::Maybe{AbstractArray{R}} = nothing,
) where {R,TP<:AbstractOptiModel}

    tstart = time()

    stop_solved(state::ALPX_state) =
        (state.optim === nothing ? false : state.optim <= solver.tol_optim) &&
        (state.cslack === nothing ? false : state.cslack <= solver.tol_cviol) &&
        (state.cviol <= solver.tol_cviol)
    stop_unbound(state::ALPX_state) =
        (state.cviol <= solver.tol_cviol) &&
        (norm(state.x, Inf) > solver.tol_diverging || state.objec < -solver.tol_diverging)
    is_illegal(x) = (x === nothing ? false : (any(isnan.(x)) || any(isinf.(x))))
    stop_illegal(state::ALPX_state) =
        is_illegal(state.x) ||
        is_illegal(state.y) ||
        is_illegal(state.optim) ||
        is_illegal(state.cx)
    stop_infeas(state::ALPX_state) =
        state.infsta <= solver.tol_optim &&
        state.cviol > solver.tol_cviol_sqrt &&
        maximum(state.μ) <= solver.tol_inf_mu
    stop(state::ALPX_state) =
        stop_solved(state) ||
        stop_unbound(state) ||
        stop_illegal(state) ||
        stop_infeas(state)

    disp((i, state)) = begin
        @printf("%5d | %+.3e | ", i, state.objec)
        state.optim === nothing ? @printf("%9s  ", "") : @printf("%.3e  ", state.optim)
        state.cslack === nothing ? @printf("%9s  ", "") :
            @printf("%.3e  ", state.cslack)
        @printf("%.3e | ", state.cviol)
        @printf("%.1e  %.1e  %.1e | ", norm(state.y, Inf), minimum(state.μ), state.ϵsub)
        if (state.sub_iter >= 0)
            state.sub_flag ? @printf("(+) ") : @printf("(-) ")
            @printf("%5d | ", state.sub_iter)
            #state.idf ? @printf("F | ") : (state.idp ? @printf("P | ") : @printf("U | "))
        end
        @printf("\n")
    end

    iter = ALPX_iterable(prob, x0, y0, mu, solver.opts)
    iter = halt(iter, stop)
    iter = take(iter, solver.max_iter)
    iter = enumerate(iter)
    if solver.verbose
        @printf(
            "%5s | %10s | %9s  %9s  %9s | %7s  %7s  %7s | \n",
            "iter",
            "objec",
            "optim",
            "cslack",
            "cviol",
            "|y|",
            "|μ|",
            "sub-ϵ"
        )
        iter = sample(iter, solver.freq)
        iter = tee(iter, disp)
    end

    num_iters, state = loop(iter)

    status = if stop_solved(state)
        :first_order
    elseif stop_infeas(state)
        :infeasible
    elseif stop_unbound(state)
        :unbounded
    elseif stop_illegal(state)
        :ieee_nan_inf
    elseif (num_iters >= solver.max_iter)
        :max_iter
    else
        :unknown
    end

    return OptiOutput(
        status,
        x = state.x,
        y = state.y,
        objective = state.objec,
        optimality = state.optim,
        cviolation = state.cviol,
        iterations = num_iters,
        time = time() - tstart,
        solver_name = "ALPX",
        solver = Dict(:cslackness => state.cslack, :infstation => state.infsta),
    )

end

###########################################################################
# Outer constructors
###########################################################################
"""
    ALPX([max_iter, tol_optim, tol_cviol, verbose, freq, ...])

Instantiate the ALPX algorithm [DeM21] for solving optimization problems
of the form

    minimize    f(x) + g(x)
    subject to  c(x) ∈ S

where `f` and `c` are smooth, `g` is proximable, and `S` is projectable. If
`solver = ALPX(args...)`, then the above problem is solved with

    solver( prob, [x0, y0] )

Optional keyword arguments:

* `max_iter::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol_optim::Real` (default: `1e-8`), absolute tolerance on the dual residual.
* `tol_cviol::Real` (default: `1e-8`), absolute tolerance on the primal residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.
* `grad_scaling::Bool`
* `scaling_grad_max::Real`
* `scaling_min_value::Real`

References:

[BiM12] Birgin, Martinez, ``Augmented Lagrangian method with nonmonotone penalty
        parameters for constrained optimization``, Computational Optimization and
        Applications, 51(3), (2012).
        DOI: 10.1007/s10589-011-9396-0

[BiM14] Birgin, Martinez, ``Practical Augmented Lagrangian Methods for Constrained
        Optimization``, SIAM, Philadelphia, (2014).
        ISBN: 978-1611973358

[LeV20] Leyffer, Vanaret, ``An Augmented Lagrangian Filter Method``
        Mathematical Methods of Operations Research, (2020).
        DOI: 10.1007/s00186-020-00713-x

[SFP20] Sopasakis, Fresk, Patrinos, ``OpEn: Code Generation for Embedded Nonconvex
        Optimization``, 21st IFAC World Congress Proceedings, (2020).
        arXiv: 2003.00292v1

[DeM21] De Marchi, ``Augmented Lagrangian Proximal solver for constrained structured
        optimization``, (2021).
"""
ALPX(::Type{R}; kwargs...) where {R} = ALPX{R}(; kwargs...)
ALPX(; kwargs...) = ALPX(Float64; kwargs...)
