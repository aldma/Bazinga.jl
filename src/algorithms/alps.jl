# alps.jl
#
# part of Bazinga.jl

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
    subsolver_maxit::Int = 1_000,
)
    start_time = time()
    T = eltype(x0)
    tol_prim = tol
    tol_dual = tol
    theta = 0.8
    kappamu = 0.5
    kappaepsilon = 0.1
    epsilon = sqrt(tol_dual)
    default_subsolver = ProximalAlgorithms.PANOCplus

    # initialize
    x = similar(x0)
    y = similar(y0)
    cx = similar(y)
    s = similar(y)
    mu = similar(y)
    gFun = NonsmoothCostFun(g)
    prox!(x, gFun, x0, eps(T))
    fx = f(x)
    objx = fx + gFun.gz
    eval!(cx, c, x)
    proj!(s, D, cx)
    default_penalty_parameter!(mu, cx, s, objx)
    y .= y0
    dual_safeguard(y)
    proj!(s, D, cx .+ mu .* y)
    norm_res_prim = norm(cx .- s, Inf)
    norm_res_prim_old = nothing
    alFun = AugLagFun(f, c, D, mu, y, x)
    tot_it = 0
    tot_inner_it = 0
    solved = false
    tired = false
    if verbose
        @info "initial penalty parameters μ ∈ [$(minimum(mu)), $(maximum(mu))]"
        @info "initial primal residual $(norm_res_prim)"
        @info "initial inner tolerance $(epsilon)"
    end

    # loop
    while !(solved || tired)
        tot_it += 1
        # dual estimate
        dual_safeguard(y)
        # inner tolerance
        epsilon = max(kappaepsilon * epsilon, tol_dual)
        # solve subproblem
        subsolver = default_subsolver(tol=epsilon, verbose=verbose, maxit=subsolver_maxit, freq=subsolver_maxit, minimum_gamma=1e-12)
        AugLagUpdate!(alFun, mu, y)
        sub_sol, sub_it = subsolver(f=alFun, g=gFun, x0=x)
        x .= sub_sol
        tot_inner_it += sub_it
        cx .= alFun.cx
        fx = alFun.fx
        gx = gFun.gz
        objx = fx + gx
        # s
        s .= alFun.s
        # dual update
        y .= alFun.yupd
        # penalty parameters
        if norm_res_prim_old === nothing
            default_penalty_parameter!(mu, cx, s, objx)
            if verbose
                @info "restarted penalty parameters μ ∈ [$(minimum(mu)), $(maximum(mu))]"
            end
        elseif norm_res_prim > max(theta * norm_res_prim_old, tol_prim)
            mu .*= kappamu
        end
        # residuals
        norm_res_prim_old = norm_res_prim
        norm_res_prim = norm(cx .- s, Inf)

        solved = sub_it < subsolver_maxit && epsilon <= tol_dual && norm_res_prim <= tol_prim
        tired = tot_it > maxit
    end
    elapsed_time = time() - start_time

    status = if solved
        :first_order
    elseif tired
        :max_iter
    else
        :unknown
    end

    return x, y, tot_it, tot_inner_it, elapsed_time, epsilon, norm_res_prim

end

function default_dual_safeguard!(y)
    y .= max.(-1e20, min.(y, 1e20))
    return nothing
end

function default_penalty_parameter!(mu, cx, proj_cx, objx)
    mu .= max.(1.0, 0.5 * (cx .- proj_cx).^2) ./ max(1.0, objx)
    mu .= max.(1e-4, min.(mu, 1e4))
    return nothing
end
