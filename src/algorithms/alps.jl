#=mutable struct alpsState{Tx,Ta,R <: Real}
    x::Tx
    y::Tx
    s::Tx
    mu::Tx
    cx::Tx
    norm_res_prim::R
    epsilon::R
    al::Ta
end
function alpsState(x0,y0)
    x = copy(x0)
    y = copy(y0)
    return alpsState(x, y )
end=#

function default_dual_safeguard!(y)
    y .= max.(-1e20, min.(y, 1e20))
    return nothing
end

function alps(
    f::ProximalOperators.ProximableFunction,
    g::ProximalOperators.ProximableFunction,
    c::SmoothFunction,
    D::ClosedSet,
    x0::AbstractArray,
    y0::AbstractArray;
    tol::Real = eltype(x0)(1e-6),
    maxit::Int = 100,
    verbose::Bool = false,
    dual_safeguard::Function = default_dual_safeguard!,
)
    start_time = time()
    T = eltype(x0)
    tol_prim = tol
    tol_dual = tol
    theta = 0.8
    kappa = 0.5
    kappaepsilon = 0.1
    epsilonmin = tol_dual
    mumin = 1e-8

    default_stop_criterion(tol_prim, tol_dual, res_prim, res_dual) = (res_prim <= tol_prim) && (res_dual <= tol_dual)
    default_penalty_parameter!(mu, cx, proj_cx, objx) = begin
        distsq = 0.5 * (cx - proj_cx).^2
        mu .= max.(1, distsq) ./ max(1, abs(objx))
        mu .= max.(1e-4, min.(mu, 1e4))
        return nothing
    end
    default_subsolver = ProximalAlgorithms.PANOCplus
    stop_criterion = default_stop_criterion

    ############################################################################
    # initialize
    x = similar(x0)
    y = similar(y0)
    cx = similar(y)
    s = similar(y)
    mu = similar(y)
    gx = prox!(x, g, x0, eps(T))
    objx = f(x) + gx
    eval!(cx, c, x)
    proj!(s, D, cx)
    default_penalty_parameter!(mu, cx, s, objx)
    y .= y0
    dual_safeguard(y)
    proj!(s, D, cx .+ mu .* y)
    norm_res_prim = norm(cx .- s, Inf)
    norm_res_prim_old = nothing
    al = AugLagFun(f, c, D, mu, y, x)
    tot_it = 0
    tot_inner_it = 0
    epsilon = sqrt(tol_dual)
    solved = false
    tired = false
    broken = false
    if verbose
        @info "initial penalty parameters μ ∈ [$(minimum(mu)), $(maximum(mu))]"
        @info "initial primal residual $(norm_res_prim)"
        @info "initial inner tolerance $(epsilon)"
    end
    subsolver_maxit = 1_000

    ###############################################################################
    while !(solved || tired)
        tot_it += 1
        # dual estimate
        dual_safeguard(y)
        # inner tolerance
        epsilon *= kappaepsilon
        epsilon = max(epsilon, epsilonmin)
        # solve subproblem
        subsolver = default_subsolver(tol=epsilon, verbose=verbose, maxit=subsolver_maxit, freq=subsolver_maxit, minimum_gamma=1e-12)
        AugLagUpdate!(al, mu, y)
        sub_sol, sub_it = subsolver(f=al, g=g, x0=x)
        x .= sub_sol
        tot_inner_it += sub_it
        cx .= al.cx
        # s
        s .= al.s
        # dual update
        y .= al.yupd
        # penalty parameters
        if norm_res_prim_old === nothing
            #
        elseif norm_res_prim > max(theta * norm_res_prim_old, tol_prim)
            mu .*= kappa
            mu .= max.(mu, mumin)
        end
        # residuals
        norm_res_prim_old = norm_res_prim
        norm_res_prim = norm(cx .- s, Inf)

        solved = stop_criterion(tol_prim, tol_dual, norm_res_prim, epsilon)
        tired = tot_it > maxit
    end
    elapsed_time = time() - start_time

    status = if solved
        :first_order
    elseif tired
        :max_iter
    elseif broken
        :exception
    else
        :unknown
    end

    #=return GenericExecutionStats(
        status,
        solution = x,
        multipliers = y,
        iter = tot_it,
        elapsed_time = elapsed_time,
        dual_feas = epsilon,
        primal_feas = norm_res_prim,
        solver_specific = Dict(
          :sub_iter => tot_inner_it,
        ),
      )=#
      return x, y, tot_it, tot_inner_it, elapsed_time, epsilon, norm_res_prim

end
