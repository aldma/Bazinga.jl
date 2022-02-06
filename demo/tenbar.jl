using ProximalOperators
using LinearAlgebra
using Bazinga

################################################################################
function computeBarLengths(nodeCoordinates, potentialBars, dimension)
    if dimension == 2
        barLengths = sqrt.((nodeCoordinates[potentialBars[:,1],1] - nodeCoordinates[potentialBars[:,2],1]).^2 +
                           (nodeCoordinates[potentialBars[:,1],2] - nodeCoordinates[potentialBars[:,2],2]).^2)
    else
        barLengths = sqrt.((nodeCoordinates[potentialBars[:,1],1] - nodeCoordinates[potentialBars[:,2],1]).^2 +
                           (nodeCoordinates[potentialBars[:,1],2] - nodeCoordinates[potentialBars[:,2],2]).^2 +
                           (nodeCoordinates[potentialBars[:,1],3] - nodeCoordinates[potentialBars[:,2],3]).^2)
    end
    return barLengths
end
function computeBarAngles(nodeCoordinates, potentialBars, barLengths, n_bars, n_nodes, dimension, fixedNodes)
    T = Float64
    barAngles = zeros(T, n_nodes, dimension, n_bars)
    for bar = 1:n_bars
        startNode = potentialBars[bar,1]
        endNode = potentialBars[bar,2]
        if !(startNode in fixedNodes)
            barAngles[startNode, :, bar] .= - (nodeCoordinates[endNode,:] - nodeCoordinates[startNode,:]) / barLengths[bar]
        end
        if !(endNode in fixedNodes)
            barAngles[endNode, :, bar] .= - (nodeCoordinates[startNode,:] - nodeCoordinates[endNode,:]) / barLengths[bar]
        end
    end
    return barAngles
end
function initStiffnessMatrix(n_displacements)
    T = Float64
    return zeros(T, n_displacements, n_displacements)
end
function computeStiffnessMatrix!(stiffnessMatrix, n_displacements, n_bars, barAngles, barLengths, barDiameters, freeNodes, Eyoung)
    stiffnessMatrix .= 0.0
    for bar = 1:n_bars
        gamma_bar = reshape(barAngles[freeNodes,:,bar], n_displacements, 1)
        stiffnessMatrix .+= (Eyoung * barDiameters[bar] / barLengths[bar]) .* (gamma_bar * gamma_bar')
    end
    return nothing
end
function initStress(n_bars)
    T = Float64
    return zeros(T, n_bars)
end
function computeStress!(stress, n_bars, barAngles, Eyoung, nodeDisplacements, n_displacements)
    u_load = reshape(nodeDisplacements,n_displacements,1)
    for bar = 1:n_bars
        gamma_bar = reshape(barAngles[freeNodes,:,bar],n_displacements,1)
        stress[bar] = (Eyoung / barLengths[bar]) * dot(gamma_bar, u_load)
    end
    return nothing
end

# structure of the truss
# |o --- o --- o
#     X  |  X  |
# |o --- o --- *
#
# numbering of the nodes
# |4 --- 5 --- 6
#     X  |  X  |
# |1 --- 2 --- 3
T = Float64
dimension = 2
nodeCoordinates = T.( [0 0; 1 0; 2 0; 0 1; 1 1; 2 1] )
n_nodes = size(nodeCoordinates, 1)
fixedNodes = [1; 4]
potentialBars = [1 2; 1 5; 4 5; 2 4; 2 5; 2 3; 2 6; 5 6; 3 5; 3 6]

# load cases
loadCase = zeros(T, n_nodes, dimension)
loadCase[3,:] .= [0; -1]

# parameters
compliance_max = T(1)
stress_min = T(-0.1)
stress_max = T(0.1)
barDiameter_max = T(100)
Eyoung = T(1)
barCardinalityCost = T(0)

n_fixedNodes = length(fixedNodes)
n_freeNodes = n_nodes - n_fixedNodes
freeNodes = setdiff(1:n_nodes, fixedNodes)
n_bars = size(potentialBars,1)
n_displacements = n_freeNodes * dimension
n_x = n_bars + n_displacements

barLengths = computeBarLengths(nodeCoordinates, potentialBars, dimension)
barAngles = computeBarAngles(nodeCoordinates, potentialBars, barLengths, n_bars, n_nodes, dimension, fixedNodes)

n_y = n_displacements + # elastic equilibrium
    1 + # max compliance
    2*n_bars # vanishing stress

# x = [barDiameters; nodeDisplacements]
#
# min f(x) + g(x)
# wrt x
#  st c(x) ∈ D
#
# f(x) := ℓᵀ a
# g(x) := δ(a) + λ \|a\|_0
# c(x) := [K(a) * u - F; Fᵀ u - C; a; stress(u)]
# D    := 0 × (-∞,0] × D_VBC^{n_bars}

###################################################################################
# problem definition
###################################################################################
struct SmoothCostTRUSS <: ProximalOperators.ProximableFunction
    n_bars::Int
    barLengths::Vector{Real}
end
function (f::SmoothCostTRUSS)(x)
    return sum(f.barLengths .* x[1:f.n_bars])
end
function ProximalOperators.gradient!(dfx, f::SmoothCostTRUSS, x)
    dfx .= 0.0
    dfx[1:f.n_bars] .= f.barLengths
    return sum(f.barLengths .* x[1:f.n_bars])
end

struct NonsmoothCostTRUSS <: ProximalOperators.ProximableFunction
    n_bars::Int
    diameter_max::Real
    cardinality_cost::Real
end
function ProximalOperators.prox!(y, g::NonsmoothCostTRUSS, x, gamma)
    y .= x
    y[1:g.n_bars] .= max.(0.0, min.(x[1:g.n_bars], g.diameter_max))
    return zero(eltype(x))
end

mutable struct ConstraintTRUSS <: SmoothFunction
    n_bars::Int
    n_displacements::Int
    dimension::Int
    n_nodes::Int
    barAngles
    barLengths
    freeNodes
    force::Vector{Real}
    compliance_max::Real
    Eyoung::Real
    stiffnessMatrix::Matrix{Real}
    stress::Vector{Real}
end
function ConstraintTRUSS(n_bars, dimension, n_nodes, barAngles, barLengths, freeNodes, loadCase, compliance_max, Eyoung)
    n_freeNodes = length(freeNodes)
    n_displacements = n_freeNodes * dimension
    stiffnessMatrix = initStiffnessMatrix( n_displacements )
    stress = initStress( n_bars )
    force = dropdims( reshape(loadCase[freeNodes,:], n_displacements, 1), dims=2)
    return ConstraintTRUSS(n_bars, n_displacements, dimension, n_nodes, barAngles, barLengths, freeNodes, force, compliance_max, Eyoung, stiffnessMatrix, stress)
end
function Bazinga.eval!(cx, c::ConstraintTRUSS, x)
    barDiameters = x[1:c.n_bars]
    nodeDisplacements = x[c.n_bars+1:end]
    # stiffnessMatrix( barDiameters ) nodeDisplacements = force
    computeStiffnessMatrix!(c.stiffnessMatrix, c.n_displacements, c.n_bars, c.barAngles, c.barLengths, barDiameters, c.freeNodes, c.Eyoung)
    cx[1:c.n_displacements] .= (c.stiffnessMatrix * nodeDisplacements)
    # force' * nodeDisplacements <= compliance_max
    cx[c.n_displacements+1] = dot(c.force, nodeDisplacements)
    # barDiameters
    # stress
    cx[c.n_displacements+2:c.n_displacements+1+c.n_bars] .= barDiameters
    computeStress!(c.stress, c.n_bars, c.barAngles, c.Eyoung, nodeDisplacements, c.n_displacements)
    cx[c.n_displacements+2+c.n_bars:c.n_displacements+1+2*c.n_bars] .= c.stress
    return nothing
end
function Bazinga.jtprod!(jtv, c::ConstraintTRUSS, x, v)
    barDiameters = x[1:c.n_bars]
    nodeDisplacements = x[c.n_bars+1:end]
    jtv .= 0.0
    # stiffnessMatrix( barDiameters ) nodeDisplacements = force
    v1 = v[1:c.n_displacements]
    computeStiffnessMatrix!(c.stiffnessMatrix, c.n_displacements, c.n_bars, c.barAngles, c.barLengths, barDiameters, c.freeNodes, c.Eyoung)
    u_load = reshape(nodeDisplacements, c.n_displacements, 1)
    for bar = 1:n_bars
        gamma_bar = reshape(barAngles[c.freeNodes,:,bar], c.n_displacements, 1)
        jtv[bar] += (c.Eyoung / c.barLengths[bar]) * dot((gamma_bar * gamma_bar') * u_load, v1)
    end
    jtv[c.n_bars+1:end] .+= ((c.stiffnessMatrix)' * v1)
    # force' * nodeDisplacements <= compliance_max
    jtv[c.n_bars+1:end] .+= (c.force .* v[c.n_displacements+1])
    # barDiameters
    # stress
    jtv[1:c.n_bars] .+= v[c.n_displacements+2:c.n_displacements+1+c.n_bars]
    v4 = v[c.n_displacements+2+c.n_bars:c.n_displacements+1+2*c.n_bars]
    dstressdu = zeros(Float64,c.n_bars,c.n_displacements)
    for bar = 1:c.n_bars
        gamma_bar = reshape(c.barAngles[c.freeNodes,:,bar],c.n_displacements,1)
        # stress[bar] = (Eyoung / barLengths[bar]) * dot(gamma_bar, u_load)
        dstressdu[bar, :] .= (c.Eyoung / c.barLengths[bar]) .* gamma_bar
    end
    jtv[c.n_bars+1:end] .+= dstressdu' * v4
    return nothing
end

struct SetTRUSS <: ClosedSet
    n_bars::Int
    n_displacements::Int
    diameter_max::Real
    stress_min::Real
    stress_max::Real
    force::Vector{Real}
    compliance_max::Real
end
function SetTRUSS(n_bars, n_displacements, freeNodes, diameter_max, stress_min, stress_max, loadCase, compliance_max)
    force = dropdims( reshape(loadCase[freeNodes,:], n_displacements, 1), dims=2)
    return SetTRUSS(n_bars, n_displacements, diameter_max, stress_min, stress_max, force, compliance_max)
end
function Bazinga.proj!(z, D::SetTRUSS, x)
    # elastic equilibrium
    z[1:D.n_displacements] .= D.force
    # max compliance
    z[D.n_displacements+1] = min( x[D.n_displacements+1], D.compliance_max )
    # (barDiameters, stress) ∈ VBC (vanishing box constraints)
    z[D.n_displacements+2:D.n_displacements+1+D.n_bars] .= max.(0.0, min.(x[D.n_displacements+2:D.n_displacements+1+D.n_bars], D.diameter_max))
    for i = 1:D.n_bars
        a = z[D.n_displacements+1+i]
        s = x[D.n_displacements+1+D.n_bars+i]
        if a < s - D.stress_max || a < D.stress_min - s
            z[D.n_displacements+1+i] = 0.0
        else
            z[D.n_displacements+1+D.n_bars+i] = max(D.stress_min, min(s, D.stress_max))
        end
    end
    return nothing
end

# iter
problem_f = SmoothCostTRUSS( n_bars, barLengths )
problem_g = NonsmoothCostTRUSS( n_bars, barDiameter_max, barCardinalityCost )
problem_c = ConstraintTRUSS( n_bars, dimension, n_nodes, barAngles, barLengths, freeNodes, loadCase, compliance_max, Eyoung )
problem_D = SetTRUSS(n_bars, n_displacements, freeNodes, barDiameter_max, stress_min, stress_max, loadCase, compliance_max)

problem_x0 = zeros(T, n_x)
problem_y0 = zeros(T, n_y)

#
x = similar(problem_x0)
y = similar(problem_y0)
dfx = similar(x)
jtv = similar(x)
cx = similar(y)
z = similar(y)
x .= problem_x0
x[1:n_bars] .= [1; 1.5; 2; 0; 0; 1; 0; 0; 1.5; 0]
x[n_bars+1:n_x] .= 0.01
y .= 0.1
fx = problem_f(x)
fx = gradient!(dfx, problem_f, x)
eval!(cx, problem_c, x)
proj!(z, problem_D, cx)
[cx z]
jtprod!(jtv, problem_c, x, y)

out = Bazinga.alps(problem_f, problem_g, problem_c, problem_D, problem_x0, problem_y0, tol = 1e-4, verbose=true)
xsol = out[1]
ysol = out[2]
tot_it = out[3]
tot_inner_it = out[4]
