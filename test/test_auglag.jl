push!(LOAD_PATH,"/home/albertodm/Documents/optimo/src");
push!(LOAD_PATH,"/home/albertodm/Documents/bazinga.jl/src");

using OptiMo
using Bazinga
using CUTEst
using Printf

VERBOSE = true
FREQ = 10
TOL = 1e-8
MAXIT = 1000

problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 )
problem = problems[6]

@printf "\n========================================\n"
@printf "%s\n" problem
nlp = CUTEstModel( problem )
prob = NLPOptiModel( nlp )
x0 = copy( prob.meta.x0 )
T = eltype( x0 )
nvar = prob.meta.nvar
ncon = prob.meta.ncon

# augmented lagrangian
fx0 = obj( prob, x0 )
cx0 = cons( prob, x0 )
px0 = proj( prob, cx0 )
px0 .= cx0 .- px0

ye = zeros(T,ncon)
mu = zeros(T,ncon)
Bazinga.penalty_estimate!( fx0, px0, mu, 0.1 )

alprob = AugLagOptiModel( prob, mu, ye )

# solver
solver = Bazinga.ZEROFPR( verbose=VERBOSE, freq=FREQ, tol_optim=TOL, max_iter=MAXIT )
@time out = solver( alprob )
print( out )

solver = Bazinga.PANOC( verbose=VERBOSE, freq=FREQ, tol_optim=TOL, max_iter=MAXIT )
@time out = solver( alprob )
print( out )

finalize( nlp )
