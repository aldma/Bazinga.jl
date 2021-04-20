using OptiMo, Bazinga
using CUTEst, NLPModelsIpopt
using CSV, Printf
using DataFrames, Query
using BenchmarkProfiles

problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 )
nprob = length( problems )

data_alpx = DataFrame()
data_ipopt = DataFrame()

TOL_OPTIM = 1e-4
TOL_CVIOL = TOL_OPTIM

alpx = Bazinga.ALPX(  verbose=false,
                      tol_optim=TOL_OPTIM,
                      tol_cviol=TOL_CVIOL,
                      subsolver=:zerofpr,
                      max_iter=50,
                      max_sub_iter=1000 )

problem = problems[2]
nlp = CUTEstModel( problem )
out = ipopt( nlp, tol=1e-3 )
prob = NLPOptiModel( nlp )
out = alpx( prob )
finalize( nlp )

for id in 1:nprob
    local problem, nlp, out, prob

    problem = problems[id]
    @printf( "********************  %s  ********************\n", problem )
    nlp = CUTEstModel( problem )

    ##########
    # IPOPT
    ##########
    out = ipopt( nlp, tol=TOL_OPTIM,print_level=0 )
    etime = out.elapsed_time
    iters = out.iter
    optim = out.dual_feas
    cviol = out.primal_feas
    solved = out.solver_specific[:internal_msg] == :Solve_Succeeded ? 1 : 0
    push!( data_ipopt, (id=id,
                  time=etime,
                  iter=iters,
                  optim=optim,
                  cviol=cviol,
                  solved=solved) )

    ##########
    # ALPX
    ##########
    prob = NLPOptiModel( nlp )
    out = alpx( prob )
    etime = out.time
    iters = out.iterations
    optim = out.optimality
    cviol = out.cviolation
    solved = out.status == :first_order ? 1 : 0
    push!( data_alpx, (id=id,
                  time=etime,
                  iter=iters,
                  optim=optim,
                  cviol=cviol,
                  solved=solved) )


    finalize( nlp )
end

n_tot = size( data_alpx, 1 )
@printf(" %d problems \n", n_tot)
data_solved_alpx = data_alpx |> @filter(_.solved == 1) |> DataFrame
n_solved_alpx = size( data_solved_alpx, 1 )
@printf(" ALPX     %4.1f/100 first order\n", 100*n_solved_alpx/n_tot)
data_solved_ipopt = data_ipopt |> @filter(_.solved == 1) |> DataFrame
n_solved_ipopt = size( data_solved_ipopt, 1 )
@printf(" IPOPT    %4.1f/100 first order\n", 100*n_solved_ipopt/n_tot)

# store results
inttol = -Int(log10(TOL_OPTIM))
foldername = "/home/alberto/Documents/"
filename = "cutest_" * "alpx" * "_$(inttol).csv"
CSV.write( foldername * "Bazinga.jl/test/data/" * filename, data_alpx, header=false )
filename = "cutest_" * "ipopt" * "_$(inttol).csv"
CSV.write( foldername * "Bazinga.jl/test/data/" * filename, data_ipopt, header=false )

# plot profiles
cost(data) = data.time + Inf .* (data.solved .== 0)
T = [cost(data_ipopt) cost(data_alpx)]
performance_profile(T, ["IPOPT","ALPX"],title="CUTEst, ϵ=$TOL_OPTIM")
