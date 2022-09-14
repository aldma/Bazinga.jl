using ScSTO
using ProximalAlgorithms
using Bazinga
using Printf
using Plots
using CSV
using DataFrames

################################################################################
# smooth cost function
struct ScSTOSmoothCost <: ProximableFunction
    model::ScSTOModel
end
function (f::ScSTOSmoothCost)(x)
    return ScSTO.obj(f.model, x)
end
function Bazinga.gradient!(dfx, f::ScSTOSmoothCost, x)
    return ScSTO.objgrad!(f.model, x, dfx)
end

# nonsmooth cost function
struct NonsmoothCostFreeTimeLO <: ProximableFunction
    swcost::Real
end
function (f::NonsmoothCostFreeTimeLO)(x)
    return f.swcost * sum(x .!== 0)
end
function Bazinga.prox!(z, f::NonsmoothCostFreeTimeLO, x, gamma)
    if f.swcost == 0
        # nonnegative
        z .= max.(0, x)
        return eltype(x)(0)
    else
        a = f.swcost * gamma
        z .= x
        z[.!(x .> sqrt(2 * a))] .= 0
        return f.swcost * sum(z .> 0)
    end
end

# constraint function
struct ConstraintFreeTime <: SmoothFunction end
function Bazinga.eval!(cx, c::ConstraintFreeTime, x)
    cx[1] = sum(x)
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintFreeTime, x, v)
    jtv[1] = sum(v)
    return nothing
end

# constraint set
struct SetFreeTimeOutBox <: ClosedSet
    lowerlower::Real
    lowerupper::Real
    upperlower::Real
    upperupper::Real
end
function Bazinga.proj!(z, f::SetFreeTimeOutBox, cx)
    zlower = max(f.lowerlower, min(cx[1], f.lowerupper))
    zupper = max(f.upperlower, min(cx[1], f.upperupper))
    dlower = abs(zlower - cx[1])
    dupper = abs(zupper - cx[1])
    z[1] = if dlower < dupper
        zlower
    elseif dlower > dupper
        zupper
    else # set-valued case
        zlower
        #z = zupper
    end
    return nothing
end

################################################################################
# make switching times vectors comparable and compare results

using LinearAlgebra

function cleandelta(delta::Vector)
    # !!! for binary controls only !!!
    # uvec = 0, 1, 0, 1, 0, 1, ...
    deltanew = copy(delta)
    n = length(delta)
    for k = 1:n-2
        if deltanew[k] > 0.0 && deltanew[k+1] == 0.0
            deltanew[k+2] += deltanew[k]
            deltanew[k] = 0.0
        end
    end
    return deltanew
end
function cleandelta(delta::Matrix)
    deltanew = copy(delta)
    m = size(delta, 2)
    for k = 1:m
        deltanew[:, k] .= cleandelta(delta[:, k])
    end
    return deltanew
end

function store_csv_file(filepath, data)
    CSV.write(filepath * ".csv", data, header = true)
    return nothing
end

################################################################################
T = Float64

# time interval
t0 = 0.0
tf = 12.0

# control sequence [nu x N]
nurepetitions = 12 # default 12
uvec = collect(repeat([0.0; 1.0], nurepetitions, 1)')
nu = size(uvec, 1)
N = size(uvec, 2)

# cost matrix
C = Matrix([1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0])
Q = Matrix(C' * C)

# initial state
state0 = [0.5; 0.7; 1; 1]
nstate = length(state0)

# system dynamics
function dynam(x::Vector{T}, u::Vector{T})
    n = length(x)
    f = zeros(n)
    f[1] = x[1] - x[1] * x[2] - 0.4 * x[1] * u[1]
    f[2] = -x[2] + x[1] * x[2] - 0.2 * x[2] * u[1]
    return f
end
function d_dynam(x::Vector{T}, u::Vector{T})
    n = length(x)
    df = zeros(n, n)
    df[1, 1] = 1.0 - x[2] - 0.4 * u[1]
    df[1, 2] = -x[1]
    df[2, 1] = x[2]
    df[2, 2] = -1.0 + x[1] - 0.2 * u[1]
    return df
end

# solver build
subsolver_directions = ProximalAlgorithms.LBFGS(5)
subsolver_maxit = 1_000_000
subsolver_minimum_gamma = 1e-32
subsolver(; kwargs...) = ProximalAlgorithms.PANOCplus(
    directions = subsolver_directions,
    maxit = subsolver_maxit,
    freq = subsolver_maxit,
    minimum_gamma = subsolver_minimum_gamma;
    kwargs...,
)
solver(f, g, c, D, x0, y0; kwargs...) = Bazinga.als(
    f,
    g,
    c,
    D,
    x0,
    y0,
    verbose = true,
    subsolver = subsolver,
    subsolver_maxit = subsolver_maxit;
    kwargs...,
)

problem_name = "scsto_box" # scsto_free, scsto_box
basepath = joinpath(@__DIR__, "results", "scsto")

# number of points on the fixed grid
ngrid = 200 # default 200

# switching cost
swc_vector = [1e-6; 1e-5; 1e-4; 0.001; 0.01; 0.1; 1.0; 10.0] # default 10.^[-6:1:1]
ntests = length(swc_vector)

# allocation
nt = ngrid; # default 2000
tfsol = Array{T}(undef, ntests);
swdelta = Array{T}(undef, N, ntests);
swtau = Array{T}(undef, N - 1, ntests);
tsim = Array{T}(undef, nt, ntests);
xsim = Array{T}(undef, nstate, nt, ntests);
usim = Array{T}(undef, nu, nt, ntests);

prob = scstoproblem(state0, dynam, d_dynam, uvec, ngrid = ngrid, t0 = t0, tf = tf, Q = Q)

f = ScSTOSmoothCost(prob)
c = ConstraintFreeTime()
if problem_name == "scsto_free"
    D = SetFreeTimeOutBox(0.0, 0.0, 0.0, 15.0)
elseif problem_name == "scsto_box"
    D = SetFreeTimeOutBox(5.0, 10.0, 13.0, 15.0)
else
    @error "Unknown problem"
end
nx = prob.meta.nvar
ny = 1

for k = 1:ntests

    # regularization
    swc = swc_vector[k]
    @info "switching cost = $(swc)"
    g = NonsmoothCostFreeTimeLO(swc)

    # initial guess
    x0 = zeros(T, nx)
    if k > 1
        x0 .= swdelta[:, k-1]
    else
        x0 .= prob.meta.x0
    end
    y0 = zeros(T, ny)

    # solve!
    out = solver(f, g, c, D, x0, y0)

    # store results
    xsol = out[1]
    swdelta[:, k] .= xsol
    swtau[:, k], _ = gettau(prob, swdelta[:, k])
    tfsol[k] = t0 + sum(xsol)
    tsim[:, k] = collect(range(t0, stop = tfsol[k], length = nt))
    xsim[:, :, k], _, _, _ = simulate(prob, swtau[:, k], tsim[:, k])
    usim[:, :, k], _ = simulateinput(prob, swtau[:, k], tsim[:, k])

    # csv files with trajectory (t,x1,x2,u)
    t = tsim[:, k]
    x1 = xsim[1, :, k]
    x2 = xsim[2, :, k]
    u = usim[1, :, k]
    df = DataFrame(t = t, x1 = x1, x2 = x2, u = u)
    filename = problem_name * "_" * string(swc_vector[k])
    filepath = joinpath(basepath, filename)
    store_csv_file(filepath, df)
end

# csv file with solutions (switching intervals)
df = DataFrame(swdelta, :auto)
filename = problem_name * "_" * "swdelta"
filepath = joinpath(basepath, filename)
store_csv_file(filepath, df)

# csv file with `cleaned` solutions (switching intervals)
swdelta_clean = cleandelta(swdelta)
df = DataFrame(swdelta_clean, :auto)
filename = problem_name * "_" * "swdelta_clean"
filepath = joinpath(basepath, filename)
store_csv_file(filepath, df)

df = DataFrame()
for k = 1:ntests
    swc = swc_vector[k]
    # swdelta
    xsol = swdelta[:, k]
    smoothcost = f(xsol)
    nonsmoothcost = swc * sum(xsol .> 0)
    obj = smoothcost + nonsmoothcost
    csol = zeros(T, ny)
    eval!(csol, c, xsol)
    psol = similar(csol)
    proj!(psol, D, csol)
    cviol = norm(csol - psol, Inf)
    # swdelta clean
    xsol = swdelta_clean[:, k]
    smoothcostclean = f(xsol)
    nonsmoothcostclean = swc * sum(xsol .> 0)
    objclean = smoothcostclean + nonsmoothcostclean
    eval!(csol, c, xsol)
    proj!(psol, D, csol)
    cviolclean = norm(csol - psol, Inf)
    # store
    push!(
        df,
        (
            swc = swc,
            f = smoothcost,
            g = nonsmoothcost,
            obj = obj,
            cviol = cviol,
            fclean = smoothcostclean,
            gclean = nonsmoothcostclean,
            objclean = objclean,
            cviolclean = cviolclean,
        ),
    )
end
filename = problem_name * "_" * "summary"
filepath = joinpath(basepath, filename)
store_csv_file(filepath, df)

################################################################################

# plot results
hplt1 = plot()
for k = 1:ntests
    plot!(hplt1, tsim[:, k], xsim[1, :, k], legend = false)
end
hplt2 = plot()
for k = 1:ntests
    plot!(hplt2, tsim[:, k], xsim[2, :, k], legend = false)
end
hplt3 = plot()
for k = 1:ntests
    plot!(hplt3, tsim[:, k], usim[1, :, k], legend = false)
end
hplt4 = plot()
for k = 1:ntests
    plot!(hplt4, tsim[:, k], usim[1, :, k], label = "swc = $(swc_vector[k])")
end
plot(hplt1, hplt2, hplt3, hplt4, show = true)

filename = problem_name
filepath = joinpath(basepath, filename)
savefig(filepath * ".pdf")
