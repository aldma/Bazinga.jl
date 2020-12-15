using OptiMo, Bazinga
using Printf, LinearAlgebra
using CUTEst

#problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 )
#problem = problems[3]
problem = "HS93"
@printf "========================================\n"
@printf "%s\n" problem
@printf "========================================\n"

nlp = CUTEstModel( problem )
prob = NLPOptiModel( nlp )
x = copy( prob.meta.x0 )
T = eltype( x )
nvar = prob.meta.nvar
ncon = prob.meta.ncon

# augmented lagrangian initialization
bigeps = T(1e-10)
gx, x = objprox( prob, x, bigeps )
fx = obj( prob, x )
cx = cons( prob, x )
px = proj( prob, cx )
px .= cx .- px

ye = zeros(T,ncon)
mu = zeros(T,ncon)
mu_est = T(0.1)
mu .= mu_est .* max.( 1.0, 0.5 .* abs2.( px ) ) ./ max( 1.0, abs( fx ) )
mu .= max.( T(1e-6), min.( T(1e6), mu ) )

w = similar(cx)
z = similar(cx)

# subsolver
VERBOSE = true
FREQ = 100
TOL = 1e-8

zerofpr = Bazinga.ZEROFPR( verbose=VERBOSE, freq=FREQ, tol_optim=TOL, max_iter=1000 )
panoc = Bazinga.PANOC( verbose=VERBOSE, freq=FREQ, tol_optim=TOL, max_iter=1000 )
alpx = Bazinga.ALPX( verbose=VERBOSE, freq=1, tol_optim=TOL, max_iter=20, max_sub_iter=1000 )

# main loop
for i in 1:2

    alprob = AugLagOptiModel( prob, mu, ye )

    out = zerofpr( alprob, x0=x )

    x .= out.x # primal update
    fx = obj( prob, x )
    gx = out.solver[:gx]
    objective = fx + gx
    optimality = out.optimality
    subiter = out.iterations
    subflag = (out.optimality <= TOL)
    cons!( prob, x, cx )
    proj!( prob, cx, px )
    px .= cx .- px
    cviolation = norm( px, Inf )
    w .= cx .+ mu .* ye
    proj!( prob, w, z )
    w .= cx .- z
    cslackness = norm( w, Inf )
    z .= w ./ mu
    ye .+= z # dual update

    @printf "%4d  %+.3e  %.3e  %.3e  %.3e | " i objective optimality cviolation cslackness
    subflag ? @printf("(+)") : @printf("(-)")
    @printf " %5d \n" subiter
    @printf "\n\n"

end

@printf "\n\n"
out = alpx( prob )
print( out )


@printf "\n\n"
out = zerofpr( prob )
print(out)

@printf "feasibility problem \n"
fprob = FeasOptiModel( prob )
fout = zerofpr( fprob )
xf = copy(fout.x)

@printf "aug lag problem \n"
gx, x = objprox( prob, xf, bigeps )
fx = obj( prob, x )
cx = cons( prob, x )
px = proj( prob, cx )
px .= cx .- px
ye = zeros(T,ncon)
mu = zeros(T,ncon)
mu_est = T(0.1)
mu .= mu_est .* max.( 1.0, 0.5 .* abs2.( px ) ) ./ max( 1.0, abs( fx ) )
mu .= max.( T(1e-6), min.( T(1e6), mu ) )
alprob = AugLagOptiModel( prob, mu, ye )
aout = zerofpr( alprob, x0=xf )

finalize( nlp )
