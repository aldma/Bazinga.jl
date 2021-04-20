using OptiMo, Bazinga
using NLPModelsIpopt, CUTEst
using DataFrames, Query
using CSV, Printf

data = DataFrame();

problem = "HIMMELBI"
nlp = CUTEstModel( problem )
prob = NLPOptiModel( nlp )

TOL_OPTIM = 1e-6
TOL_CVIOL = TOL_OPTIM

# warm-up
out = ipopt( nlp, tol=sqrt(TOL_OPTIM) )
solver = Bazinga.ALPX(  verbose=true,
                        tol_optim=sqrt(TOL_OPTIM),
                        tol_cviol=sqrt(TOL_CVIOL),
                        subsolver=:zerofpr,
                        max_iter=50,
                        max_sub_iter=10 )
out = solver( prob )

# solve!
for maxsubiter in [1000; 2500; 5000; 10000; 12500; 15000; 20000]

    @printf "\n\n max sub iters %d \n" maxsubiter
    solver = Bazinga.ALPX(  verbose=true,
                            tol_optim=TOL_OPTIM,
                            tol_cviol=TOL_CVIOL,
                            subsolver=:zerofpr,
                            max_iter=50,
                            max_sub_iter=maxsubiter )
    out = solver( prob )
    @printf "sub iters %d \n" out.solver[:sub_iterations]
    @printf "time      %f \n" out.time

end

@printf "\n\n IPOPT \n"
out = ipopt( nlp, tol=TOL_OPTIM )
print(out)
@printf "time      %f \n\n" out.elapsed_time


@printf "\n\n"
maxsubiter = 5000
mu_est=1.0
solver = Bazinga.ALPX(  verbose=true,
                        tol_optim=TOL_OPTIM,
                        tol_cviol=TOL_CVIOL,
                        mu_est=mu_est,
                        max_sub_iter=maxsubiter )
out = solver( prob )
@printf "sub iters %d \n" out.solver[:sub_iterations]
@printf "time      %f \n" out.time

mu_est=10.0
solver = Bazinga.ALPX(  verbose=true,
                        tol_optim=TOL_OPTIM,
                        tol_cviol=TOL_CVIOL,
                        mu_est=mu_est,
                        max_sub_iter=maxsubiter )
out = solver( prob )
@printf "sub iters %d \n" out.solver[:sub_iterations]
@printf "time      %f \n" out.time

# finalize
finalize( nlp )
