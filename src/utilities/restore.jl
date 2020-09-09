using OptiMo
using LinearAlgebra: norm
using Printf

"""
    restore( prob, [x0, with_indicator, verbose, ...] )
"""
function restore(
    prob::TP;
    x0::Tx = prob.meta.x0,
    with_indicator::Bool = false,
    verbose::Bool = false,
    max_out_iter::Int = 100,
    max_sub_iter::Int = 1000,
    ifactor::Real = 0.1,
    wfactor::Real = 0.1,
    wmin::Real = 1e-16,
    tol_cviol::Real = 1e-8,
    tol_inf_cviol::Real = 1e-5,
    tol_infst::Real = 1e-10,
    tol_optim::Real = 1e-14,
    loss::Symbol = :l2sq,
    huber_rho::Real = 1.0,
    huber_mu::Real = 1.0,
) where {TP<:AbstractOptiModel,Tx<:AbstractVector}

    @assert 0 < tol_optim
    @assert 0 < tol_cviol
    @assert 0 < tol_inf_cviol
    @assert 0 < tol_infst
    @assert 0 < wmin
    @assert 0 < wfactor < 1
    @assert 0 < ifactor
    @assert 0 < max_sub_iter
    @assert 0 < max_out_iter

    solver = Bazinga.ZEROFPR( verbose = false,
                              max_iter = max_sub_iter,
                              tol_optim = tol_optim)

    # initial point = base point
    x = copy(x0)
    R = eltype(x)
    # allocation
    s = similar(x)
    infst = -1.0
    if with_indicator
        bigeps = 1e-10
    end
    # diagonal scaling
    d0 = min.(1.0, 1.0 ./ abs.(x0))
    # initial infeasibility
    fprob = FeasOptiModel(
        prob,
        x0 = x,
        with_indicator = with_indicator,
        loss = loss,
        huber_rho = huber_rho,
        huber_mu = huber_mu,
    )
    infea = infeasibility(fprob, x)
    cviol = cviolation(fprob, x)
    # initial weight estimate
    w = max(wmin, ifactor * infea)

    if verbose
        @printf(
            "%4s | %7s | %4s  %7s | %7s  %7s | %7s | \n",
            "o.it",
            "w",
            "i.it",
            "optim",
            "infea",
            "pdist",
            "cviol"
        )
    end

    niter = 0
    while true

        fprob = FeasOptiModel(
            prob,
            x0 = x,
            with_indicator = with_indicator,
            xprox = x0,
            wprox = w,
            dprox = d0,
            loss = loss,
            huber_rho = huber_rho,
            huber_mu = huber_mu,
        )
        out = solver(fprob)
        copyto!(x, out.x)
        infea = infeasibility(fprob, x)
        cviol = cviolation(fprob, x)
        w = max(wmin, min(wfactor * w, ifactor * infea))

        # infeasibility stationarity
        infeasibilitygrad!(fprob, x, s)
        if with_indicator
            s .= x .- bigeps .* s
            prox!(fprob, s, bigeps, s)
            s .= (x .- s) ./ bigeps
        end
        infst = norm(s, Inf)

        if verbose
            proxd = proxdistance(fprob, x)
            @printf(
                "%4d | %.1e | %4d  %.1e | %.1e  %.1e | %.1e | %.1e \n",
                niter,
                w,
                out.iterations,
                out.optimality,
                infea,
                proxd,
                cviol,
                infst
            )
        end

        niter += 1
        if cviol <= tol_cviol
            if verbose
                @printf "cviolation satisfied"
            end
            break
        elseif cviol >= tol_inf_cviol && infst <= tol_infst
            if verbose
                @printf "local infeasibility detected"
            end
            break
        elseif w <= wmin && out.iterations == 1
            if verbose
                @printf "tired restoration"
            end
            break
        elseif niter >= max_out_iter
            if verbose
                @printf "max iterations reached"
            end
            break
        end

    end
    return x, cviol, infst
end
