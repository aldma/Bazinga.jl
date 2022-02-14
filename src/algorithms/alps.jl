# alps.jl
#
# part of Bazinga.jl

default_subsolver = ProximalAlgorithms.PANOCplus

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
    subsolver_minimum_gamma::Real = eps(eltype(x0)),
    subsolver::TS = default_subsolver,
) where {TS}
    start_time = time()
    T = eltype(x0)
    tol_prim = tol
    tol_dual = tol
    theta = 0.8
    kappamu = 0.5
    kappaepsilon = 0.1

    # allocation
    x = similar(x0)
    y = similar(y0)
    cx = similar(y0)
    s = similar(y0)
    mu = similar(y0)
    # initialize
    epsilon = sqrt(tol_dual)
    gFun = NonsmoothCostFun(g)
    prox!(x, gFun, x0, eps(T))
    objx = f(x) + gFun.gz
    eval!(cx, c, x)
    proj!(s, D, cx)
    default_penalty_parameter!(mu, cx, s, objx)
    y .= y0
    norm_res_prim = nothing
    norm_res_prim_old = nothing
    alFun = AugLagFun(f, c, D, mu, y, x)
    tot_it = 0
    tot_inner_it = 0
    solved = false
    tired = tot_it >= maxit
    broken = isnan(objx)
    if verbose
        @info "initial penalty parameters μ ∈ [$(minimum(mu)), $(maximum(mu))]"
        @info "initial inner tolerance $(epsilon)"
    end

    # loop
    while !(solved || tired || broken)
        tot_it += 1
        # safeguarded dual estimate
        y .= alFun.yupd
        dual_safeguard(y)
        # inner tolerance
        epsilon = max(kappaepsilon * epsilon, tol_dual)
        # solve subproblem
        sub_solver = subsolver(
            tol = epsilon,
            minimum_gamma = subsolver_minimum_gamma,
        )
        AugLagUpdate!(alFun, mu, y)
        sub_sol, sub_it = sub_solver(f = alFun, g = gFun, x0 = x)
        x .= sub_sol
        tot_inner_it += sub_it
        cx .= alFun.cx
        s .= alFun.s
        objx = alFun.fx + gFun.gz
        broken = isnan(objx)
        # dual estimate update
        #y .= alFun.yupd
        # residuals
        norm_res_prim_old = norm_res_prim
        norm_res_prim = norm(cx .- s, Inf)
        # penalty parameters
        if norm_res_prim_old === nothing
            #
        elseif norm_res_prim > max(theta * norm_res_prim_old, tol_prim)
            mu .*= kappamu
        end
        if verbose
            @info "$(tot_it) | $(tot_inner_it) | $(objx) | $(epsilon) | $(norm_res_prim)"
        end

        solved = epsilon <= tol_dual && norm_res_prim <= tol_prim # check subsolver_maxit?
        tired = tot_it >= maxit
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

    return x, y, tot_it, tot_inner_it, elapsed_time, epsilon, norm_res_prim, s

end

function default_dual_safeguard!(y)
    y .= max.(-1e20, min.(y, 1e20))
    return nothing
end

function default_penalty_parameter!(mu, cx, proj_cx, objx)
    mu .= max.(1.0, 0.5 * (cx .- proj_cx) .^ 2) ./ max(1.0, objx)
    mu .= max.(1e-8, min.(mu, 1e8))
    return nothing
end
