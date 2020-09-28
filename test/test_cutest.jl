push!(LOAD_PATH,"/home/alberto/Documents/OptiMo.jl/src");
push!(LOAD_PATH,"/home/alberto/Documents/Bazinga.jl/src");

using OptiMo, Bazinga
using CUTEst

using DataFrames, Query, CSV
using Printf

using NLPModels
using NLPModelsIpopt
#using Percival

problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 )
#problems_to_exclude = [ "PALMER7ANE", "PALMER1BNE", "MGH17", "MGH09", "S365",
#                        "POWERSUMNE", "ERRINRSMNE", "HIMMELBFNE", "WEEDSNE",
#                        "PALMER3BNE", "PALMER1ENE", "MISRA1B", "MESH" ]
#deleteat!(problems, findall(x->x ∈ problems_to_exclude, problems))
#problems = [ "TENBARS2" ]
nprob = length( problems )
#nprob = min( nprob, 25 )
#problems = problems[1:nprob]

data = DataFrame()

TOL_OPTIM = 1e-8
TOL_CVIOL = TOL_OPTIM

solver_flag = :alpx
subsolver_flag = :zerofpr

for id in 1:1
    problem = problems[id]
    nlp = CUTEstModel( problem )
    if solver_flag == :ipopt
        out = ipopt( nlp, tol=1e-3 )
    elseif solver_flag == :alpx
        prob = NLPOptiModel( nlp )
        solver = Bazinga.ALPX(  verbose=false,
                                tol_optim=1e-3,
                                tol_cviol=1e-3,
                                subsolver=subsolver_flag )
        out = solver( prob )
    end
    finalize( nlp )
end

for id in 1:nprob
    problem = problems[id]
    @printf( "********************  %s  ********************\n", problem )
    nlp = CUTEstModel( problem )

    #out = percival( nlp )
    if solver_flag == :ipopt
        out = ipopt( nlp, tol=TOL_OPTIM )
        status = out.status
        etime = out.elapsed_time
        iters = out.iter
        optim = out.dual_feas
        cviol = out.primal_feas
        mssg = out.solver_specific[:internal_msg]
    elseif solver_flag == :alpx
        prob = NLPOptiModel( nlp )
        solver = Bazinga.ALPX(  verbose=false,
                                tol_optim=TOL_OPTIM,
                                tol_cviol=TOL_CVIOL,
                                subsolver=subsolver_flag,
                                max_iter=50,
                                max_sub_iter=1000 )
        out = solver( prob )
        status = out.status
        etime = out.time
        iters = out.iterations
        optim = out.optimality
        cviol = out.cviolation
        mssg = out.status
    end

    if solver_flag == :alpx && status != :first_order
        print( out )
    end
    push!( data, (id=id, name=problem, status=status, time=etime,
                  iter=iters, optim=optim, cviol=cviol, mssg=mssg) )

    finalize( nlp )
end

n_tot = size( data, 1 )
datatmp = data |> @filter(_.status == :first_order) |> DataFrame
n_first_order = size( datatmp, 1 )
datatmp = data |> @filter(_.status == :max_iter) |> DataFrame
n_max_iter = size( datatmp, 1 )
datatmp = data |> @filter(_.status == :infeasible) |> DataFrame
n_infeas = size( datatmp, 1 )
datatmp = data |> @filter(_.status == :unknown) |> DataFrame
n_unknown = size( datatmp, 1 )
datatmp = data |> @filter(_.status == :acceptable) |> DataFrame
n_acceptable = size( datatmp, 1 )
@printf(" %d problems \n", n_tot)
@printf("      %4.1f/100 first order\n", 100*n_first_order/n_tot)
@printf("      %4.1f/100 acceptable\n", 100*n_acceptable/n_tot)
@printf("      %4.1f/100 max iterations\n", 100*n_max_iter/n_tot)
@printf("      %4.1f/100 infeasible\n", 100*n_infeas/n_tot)
@printf("      %4.1f/100 unknown\n", 100*n_unknown/n_tot)

filename = (solver_flag == :alpx ? "alpx_" : "ipopt")
#filename = "cutest_tmp_" * filename * ".csv"
filename = "cutest_" * filename * "_8red.csv"
CSV.write( "/home/alberto/Documents/Bazinga.jl/test/data/" * filename, data )
