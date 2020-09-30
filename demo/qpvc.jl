"""
	qpvc : example from Bazinga.jl

	Quadratic program with vanishing constraints, from [KPB11].

	Original formulations:

	minimize      1/2 x' Q x + x' q
	subject to    x[i] ≥ 0                             ∀ i ∈ [1:nvc]
                  x[i] (G[i,:] x - g[i]) ≥ 0           ∀ i ∈ [1:nvc]


	Reformulation as a constrained structured problem in the form

	minimize     f(x)
	subject to   c(x) in S

	where
	    f(x) = 1/2 x' Q x + x' q
	    c(x) = [ x[1:nvc] ]
       	       [ G x - g  ]
	    S    = { (a,b) | ∀i ∈ [1:nvc] a_i = 0  or  a_i, b_i ≥ 0 }

    References:
    [KPB11]     Kirches, Potschka, Bock,, Sager, "A parametric active set method
                for quadratic programs with vanishing constraints" (2011).
                Pacific Journal of Optimization
"""

foldername = "/home/alberto/Documents/"
push!(LOAD_PATH, foldername * "OptiMo.jl/src");
push!(LOAD_PATH, foldername * "Bazinga.jl/src");

using Bazinga, OptiMo
using Random, LinearAlgebra
using DataFrames, Query, CSV
using Printf, Plots

###################################################################################
# problem definition
###################################################################################
mutable struct QPVC <: AbstractOptiModel
    meta::OptiModelMeta
    Q::Matrix
    q::Vector
    G::Matrix
    g::Vector
    nvc::Int
end

function QPVC(; nx::Int = 100, nvc::Int = 25, x0::Vector = zeros(Float64, nx))
    @assert nx >= 2 * nvc
    name = "QpVcExample"
    ncon = 2 * nvc
    R = Float64
    Q = randn(R, nx, nx)
    Q .= (Q' * Q)
    q = randn(R, nx)
    G = randn(R, nvc, nx)
    g = randn(R, nvc)
    meta = OptiModelMeta(nx, ncon, x0 = x0, name = name)
    return QPVC(meta, Q, q, G, g, nvc)
end

# necessary methods:
# obj, grad!: cons!, jprod!, jtprod!, proj!, prox!, objprox!
function OptiMo.obj(prob::QPVC, x::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    return 0.5 * x' * prob.Q * x + x' * prob.q
end

function OptiMo.grad!(prob::QPVC, x::AbstractVector, dfx::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    dfx .= prob.Q * x .+ prob.q
    return nothing
end

function OptiMo.cons!(prob::QPVC, x::AbstractVector, cx::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x
    OptiMo.@lencheck prob.meta.ncon cx
    cx .= [
        x[1:prob.nvc]
        prob.G * x .- prob.g
    ]
    return nothing
end

function OptiMo.jprod!(prob::QPVC, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x v
    OptiMo.@lencheck prob.meta.ncon Jv
    Jv .= [
        v[1:prob.nvc]
        prob.G * v
    ]
    return nothing
end

function OptiMo.jtprod!(
    prob::QPVC,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    OptiMo.@lencheck prob.meta.nvar x Jtv
    OptiMo.@lencheck prob.meta.ncon v
    Jtv .= prob.G' * v[prob.nvc+1:prob.meta.ncon]
    Jtv[1:prob.nvc] .+= v[1:prob.nvc]
    return nothing
end

function OptiMo.prox!(prob::QPVC, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    z[1:prob.nvc] .= max.(x[1:prob.nvc], 0)
    return nothing
end

function OptiMo.objprox!(prob::QPVC, x::AbstractVector, a::Real, z::AbstractVector)
    OptiMo.@lencheck prob.meta.nvar x z
    z .= x
    z[1:prob.nvc] .= max.(x[1:prob.nvc], 0)
    return 0.0
end

function OptiMo.proj!(prob::QPVC, cx::AbstractVector, px::AbstractVector)
    OptiMo.@lencheck prob.meta.ncon cx px
    for i = 1:prob.nvc
        a = cx[i]
        b = cx[i+prob.nvc]
        a = max(a, 0)
        if a + b < 0
            a = 0
        else
            b = max(0, b)
        end
        px[i] = a
        px[i+prob.nvc] = b
    end
    return nothing
end

###################################################################################
# problem build
###################################################################################
problem = QPVC()

###################################################################################
# solver build
###################################################################################
solver = Bazinga.ALPX(max_sub_iter = 1000, verbose = true, subsolver = :zerofpr)
out = solver(problem)
print(out)

solver = Bazinga.ALPX(max_sub_iter = 1000, verbose = false, subsolver = :zerofpr)

data = DataFrame()
ntests = 100
ndots = round(Int, cbrt(ntests^2))

for i = 1:ntests
    problem = QPVC()
    out = solver(problem)
    print(out)
    push!(
        data,
        (
            id = i,
            time = out.time,
            iters = out.iterations,
            subiters = out.solver[:sub_iterations],
            cviol = out.cviolation,
            optim = out.optimality,
            cslack = out.solver[:cslackness],
            status = out.status,
        ),
    )
    @printf "."
    if mod(i, ndots) == 0
        @printf "\n"
    end
end
@printf "\n"

###################################################################################
# write
###################################################################################
filename = "qpvc"
CSV.write(foldername * "Bazinga.jl/demo/data/" * filename * ".csv", data)

###################################################################################
# plot
###################################################################################
pyplot()

figname = filename * "_time"
histogram( log10.(data[!,2]), bins=10, legend=false )
xlabel!("log10(elapsed time [s])")
savefig(foldername * "Bazinga.jl/demo/data/" * figname * ".pdf")

figname = filename * "_iters"
histogram( data[!,3], bins=10, legend=false )
xlabel!("iterations")
savefig(foldername * "Bazinga.jl/demo/data/" * figname * ".pdf")

figname = filename * "_subiters"
histogram( data[!,4], bins=10, legend=false )
xlabel!("tot sub iterations")
savefig(foldername * "Bazinga.jl/demo/data/" * figname * ".pdf")

max_cviol = maximum( data[!,5] )
max_optim = maximum( data[!,6] )
max_cslack = maximum( data[!,7] )
datatmp = data |> @filter(_.status == :first_order) |> DataFrame
n_first_order = size( datatmp, 1 )
