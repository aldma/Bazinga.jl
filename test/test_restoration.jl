push!(LOAD_PATH,"/home/alberto/Documents/optimo/src");
push!(LOAD_PATH,"/home/alberto/Documents/bazinga.jl/src");

using OptiMo
using LinearAlgebra
using CUTEst
using Bazinga
using Printf

solver = Bazinga.ZEROFPR( verbose=false, freq=100, tol_optim=1e-12 )

# problem
problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 );
problems_infeas = ["HIMMELBD", "JUNKTURN", "LUBRIF", "LUBRIFC", "MODEL"]
                  # medium ["FLOSP2HH", "FLOSP2HL", "FLOSP2HM", "WOODSNE", ]
                  # large ["CONT6-QQ", "DRCAVTY3"]
problem = problems[45]
nlp = CUTEstModel( problem )
prob = NLPOptiModel( nlp );

x0 = copy( prob.meta.x0 )

xk = x0 .+ 0.2

@printf "\n========== L1    =================\n"
@time x, cviol, infst = Bazinga.restore( prob, x0=xk, with_indicator=true, verbose=true, loss=:l1 )

@printf "\n========== L2    =================\n"
@time x, cviol, infst = Bazinga.restore( prob, x0=xk, with_indicator=true, verbose=true, loss=:l2 )

@printf "\n========== L2SQ  =================\n"
@time x, cviol, infst = Bazinga.restore( prob, x0=xk, with_indicator=true, verbose=true, loss=:l2sq )

@printf "\n========== HUBER =================\n"
@time x, cviol, infst = Bazinga.restore( prob, x0=xk, with_indicator=true, verbose=true, loss=:huber )





finalize( nlp )

#=
xk = x0 .+ 0.2
dk = min.( 1.0, 1.0 ./ abs.(xk) )

x = copy( xk )
cx = cons(prob, x)
px = proj(prob, cx)
px .= cx .- px
global infeas = 0.5 * norm( px, 2 )^2

wfac = 1e-2 # default = 1.0
maxriter = 10

@printf "===========================\n"
@printf "with adaptive prox regularization \n"
global w = wfac * 2 * infeas
@printf "%3d | w = %.3e | \n" 0 w
for i = 1:maxriter

    xold = copy( x )

    # proximal subproblem with base point at `xk`
    local fprob = FeasOptiModel( prob, x0=x, xprox=xk, wprox=w, dprox=dk, with_indicator=true )

    local out = solver( fprob )

    copyto!( x, out.x )
    cons!( prob, x, cx )
    proj!( prob, cx, px )
    px .= cx .- px

    local iters  = out.iterations
    local optim  = out.optimality
    local proxim = 0.5 * sum( dk .* (x .- xold).^2 )
    local prdist = 0.5 * sum( dk .* (x .- xk).^2 )
    global infeas = 0.5 * norm( px, 2 )^2
    global w = wfac * 2 * infeas
    @printf "%3d | w = %.3e | %4d it : %.2e | inf = %.3e | %.3e %.3e | \n" i w iters optim infeas proxim prdist
end

@printf "===========================\n"
@printf "with adaptive prox-point regularization \n"
x = copy( xk )
cx = cons(prob, x)
px = proj(prob, cx)
px .= cx .- px
infeas = 0.5 * norm( px, 2 )^2
w = wfac * 2 * infeas
@printf "%3d | w = %.3e | \n" 0 w
for i = 1:maxriter

    xold = copy( x )

    # proximal subproblem with base point at `x`
    local fprob = FeasOptiModel( prob, x0=x, xprox=x, wprox=w, dprox=dk, with_indicator=true )

    local out = solver( fprob )

    copyto!( x, out.x )
    cons!( prob, x, cx )
    proj!( prob, cx, px )
    px .= cx .- px

    local iters  = out.iterations
    local optim  = out.optimality
    local proxim = 0.5 * sum( dk .* (x .- xold).^2 )
    local prdist = 0.5 * sum( dk .* (x .- xk).^2 )
    global infeas = 0.5 * norm( px, 2 )^2
    global w = wfac * 2 * infeas
    @printf "%3d | w = %.3e | %4d it : %.2e | inf = %.3e | %.3e %.3e | \n" i w iters optim infeas proxim prdist
end

@printf "===========================\n"
@printf "with adaptively scaled prox-point regularization \n"
x = copy( xk )
cx = cons(prob, x)
px = proj(prob, cx)
px .= cx .- px
infeas = 0.5 * norm( px, 2 )^2
w = wfac * 2 * infeas
@printf "%3d | w = %.3e | \n" 0 w
for i = 1:maxriter

    xold = copy( x )
    local d = min.( 1.0, 1.0 ./ abs.(x) )
    # proximal subproblem with base point at `x`
    local fprob = FeasOptiModel( prob, x0=x, xprox=x, wprox=w, dprox=d, with_indicator=true )

    local out = solver( fprob )

    copyto!( x, out.x )
    cons!( prob, x, cx )
    proj!( prob, cx, px )
    px .= cx .- px

    local iters  = out.iterations
    local optim  = out.optimality
    local proxim = 0.5 * sum( dk .* (x .- xold).^2 )
    local prdist = 0.5 * sum( dk .* (x .- xk).^2 )
    global infeas = 0.5 * norm( px, 2 )^2
    global w = wfac * 2 * infeas
    @printf "%3d | w = %.3e | %4d it : %.2e | inf = %.3e | %.3e %.3e | \n" i w iters optim infeas proxim prdist
end

@printf "===========================\n"
@printf "without regularization \n"
# direct method, without prox regularization
fprob = FeasOptiModel( prob, x0=xk, with_indicator=true )
out = solver( fprob )
copyto!( x, out.x )
cons!( prob, x, cx )
proj!( prob, cx, px )
px .= cx .- px

iters  = out.iterations
optim  = out.optimality
prdist = 0.5 * sum( dk .* (x .- xk).^2 )
infeas = 0.5 * norm( px, 2 )^2
@printf "%3d | w = %.3e | %4d it : %.2e | inf = %.3e | %.3e %.3e | \n" 0 w iters optim infeas prdist prdist

=#
