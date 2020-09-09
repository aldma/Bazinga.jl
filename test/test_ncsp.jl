push!(LOAD_PATH,"/home/albertodm/Documents/optimo/src");
push!(LOAD_PATH,"/home/albertodm/Documents/bazinga.jl/src");

using OptiMo, Bazinga

using CUTEst
using NLPModelsIpopt
using Printf

problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 )
problem = problems[2]
#problem = "TENBARS2"
nlp = CUTEstModel( problem )

# IPOPT
ipopt_out = ipopt( nlp )
print( ipopt_out )

# ALPX
prob = NLPOptiModel( nlp )
solver = Bazinga.ALPX( verbose=true, subsolver=:zerofpr )
out = solver( prob )
print( out )

finalize( nlp )
