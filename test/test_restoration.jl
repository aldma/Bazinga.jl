using OptiMo, Bazinga
using Printf, LinearAlgebra
using CUTEst

solver = Bazinga.ZEROFPR( verbose=false, freq=100, tol_optim=1e-12 )

# problem
problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 );
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
