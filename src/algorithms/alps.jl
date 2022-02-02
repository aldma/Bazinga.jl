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

function alps(
    f::ProximalOperators.ProximableFunction,
    g::ProximalOperators.ProximableFunction,
    c::SmoothFunction,
    D::ClosedSet,
    x0::AbstractArray,
    y0::AbstractArray,
    tol::Real = eltype(x0)(1e-6),
    maxit::Int = 100,
)
    start_time = time()
    T = eltype(x0)
    tol_prim = tol
    tol_dual = tol
    theta = 0.8
    kappa = 0.5
    kappaepsilon = 0.25
    epsilonmin = 1e-12
    ymax = 1e20

    default_dual_safeguard!(y) = begin
        y .= max.(-ymax, min.(y, ymax))
        return nothing
    end
    default_stop_criterion(tol_prim, tol_dual, res_prim, res_dual) = (res_prim <= tol_prim) && (res_dual <= tol_dual)
    default_penalty_parameter!(mu, cx, proj_cx, objx) = begin
        distsq = (cx - proj_cx).^2
        mu .= 0.1 * max.(1, distsq) ./ max(1, objx)
        mu .= max.(1e-3, min.(mu, 1e3))
        return nothing
    end
    default_subsolver = ProximalAlgorithms.PANOCplus

    ############################################################################
    # initialize
    x = similar(x0)
    y = similar(y0)
    cx = similar(y)
    s = similar(y)
    mu = similar(y)
    gx = prox!(x, g, x0, eps(T))
    y .= y0
    eval!(cx, c, x)
    proj!(s, D, cx)
    objx = f(x) + gx
    default_penalty_parameter!(mu, cx, s, objx)
    default_dual_safeguard!(y)
    proj!(s, D, cx .+ mu .* y)
    norm_res_prim = norm(cx .- s, Inf)
    norm_res_prim_old = nothing
    al = AugLagFun(f, c, D, mu, y, x)
    tot_it = 0
    tot_inner_it = 0
    epsilon = sqrt(tol_dual)
    first_order = false
    solved = false
    tired = false

    ###############################################################################
    while !(solved || tired)
        tot_it += 1
        # dual estimate
        default_dual_safeguard!(y)
        # inner tolerance
        epsilon *= kappaepsilon
        epsilon = max(epsilon, epsilonmin)
        # solve subproblem
        subsolver = default_subsolver(tol=epsilon, verbose=true, freq=100)
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
        end
        # residuals
        norm_res_prim_old = norm_res_prim
        norm_res_prim = norm(cx .- s, Inf)

        first_order = default_stop_criterion(tol_prim, tol_dual, norm_res_prim, epsilon)
        solved = first_order
        tired = tot_it > maxit
    end
    elapsed_time = time() - start_time

    status = if first_order
        :first_order
    elseif tired
        :max_iter
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
