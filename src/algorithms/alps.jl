# alps.jl
#
# part of Bazinga.jl

default_subsolver = ProximalAlgorithms.PANOCplus

function alps(
    f::Tf,
    g::Tg,
    c::SmoothFunction,
    D::ClosedSet,
    x0::AbstractArray,
    y0::AbstractArray;
    tol::Real = eltype(x0)(1e-6),
    tol_prim::Real = tol,
    tol_dual::Real = tol,
    inner_tol::Real = cbrt(tol_dual),
    maxit::Int = 100,
    theta_penalty::Real = 0.8,
    kappa_penalty::Real = 0.5,
    kappa_tol::Real = 0.1,
    verbose::Bool = false,
    dual_safeguard::Function = default_dual_safeguard!,
    subsolver::TS = default_subsolver,
    subsolver_maxit::Int = 1_000_000_000,
) where {Tf,Tg,TS}
    start_time = time()
    T = eltype(x0)

    # allocation
    x = similar(x0)
    y = similar(y0)
    cx = similar(y0)
    s = similar(y0)
    mu = similar(y0)
    # initialize
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
        @info "initial inner tolerance $(inner_tol)"
    end

    # loop
    can_stop = solved || tired || broken
    while !can_stop
        tot_it += 1
        # safeguarded dual estimate
        dual_safeguard(y, cx) # in-place
        # solve subproblem
        sub_solver = subsolver(tol = inner_tol, verbose = verbose)
        AugLagUpdate!(alFun, mu, y)
        sub_sol, sub_it = sub_solver(f = alFun, g = gFun, x0 = x)
        x .= sub_sol
        objx = alFun.fx + gFun.gz
        tot_inner_it += sub_it
        sub_solved = sub_it < subsolver_maxit
        #cx .= alFun.cx # cx from alFun ?
        eval!(cx, c, x)
        #s .= alFun.s # s from alFun ?
        y .= cx .+ alFun.muy # temporary
        proj!(s, D, y)
        # NB: taking s as simply the projection assumes that the proj operator is deterministic, 
        # so that s is in fact a certificate associated to x for the subproblem.
        # dual estimate update
        #y .= alFun.yupd # dual estimate from alFun ?
        y .-= s # temporary
        y ./= mu
        # residuals
        norm_res_prim_old = norm_res_prim
        norm_res_prim = norm(cx .- s, Inf)

        # termination checks
        solved = (inner_tol <= tol_dual && sub_solved) && (norm_res_prim <= tol_prim)
        tired = tot_it >= maxit
        broken = isnan(objx)
        can_stop = solved || tired || broken

        if !can_stop
            # update penalty parameters
            if norm_res_prim_old === nothing
                #
            elseif norm_res_prim > max(theta_penalty * norm_res_prim_old, tol_prim)
                mu .*= kappa_penalty
            end
            # update inner tolerance
            inner_tol = max(kappa_tol * inner_tol, tol_dual)
        end
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

    return x, y, tot_it, tot_inner_it, elapsed_time, status, inner_tol, norm_res_prim, s, mu

end
