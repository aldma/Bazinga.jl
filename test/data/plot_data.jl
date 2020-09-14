using CSV
using DataFrames, Query
using PyPlot, PyCall

ipopt6_filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_ipopt_6.csv"
alpx6__filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_alpx__6.csv"
ipopt8_filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_ipopt_8.csv"
alpx8__filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_alpx__8.csv"

stats = Dict( :ALPX6 => CSV.File(alpx6__filename) |> DataFrame,
              :IPOPT6 => CSV.File(ipopt6_filename) |> DataFrame,
              :ALPX8 => CSV.File(alpx8__filename) |> DataFrame,
              :IPOPT8 => CSV.File(ipopt8_filename) |> DataFrame, )

cost(df) = df.time + (df.status .!= "first_order") * Inf
solvers = keys(stats)

statuses = []
for s in solvers
    df = stats[s]
    global statuses = [statuses; unique(df.mssg)]
end
statuses = unique(statuses)

tabs = DataFrame()
for s in solvers
    println("==========  ",s,"  ==========")
    df = stats[s]
    for m in statuses
        df_m = df |> @filter(_.mssg == m) |> DataFrame
        n_m = size( df_m, 1 )
        println(m,"  ", n_m)
    end
end

#dfs = (stats[s] for s in solvers)
#P = hcat([cost(df) for df in dfs]...)
#logP = log10.(P)
#histogram(logP, label=hcat([string(s) for s in solvers]...), fillalpha=0.8)

solvedIPOPT6 = (stats[:IPOPT6].status .== "first_order")
solvedALPX6  = (stats[:ALPX6].status  .== "first_order")
solvedBOTH6  = solvedALPX6 .& solvedIPOPT6
solvedIPOPT8 = (stats[:IPOPT8].status .== "first_order")
solvedALPX8  = (stats[:ALPX8].status  .== "first_order")
solvedBOTH8  = solvedALPX8 .& solvedIPOPT8

timeIPOPT6 = stats[:IPOPT6].time[solvedBOTH6]
timeALPX6  = stats[:ALPX6].time[solvedBOTH6]
timeIPOPT8 = stats[:IPOPT8].time[solvedBOTH8]
timeALPX8  = stats[:ALPX8].time[solvedBOTH8]

nIPOPT6 = sum( timeIPOPT6 .< timeALPX6 )
nALPX6  = sum( timeIPOPT6 .> timeALPX6 )
println("tol 6 : IPOPT ", nIPOPT6," - ", nALPX6," ALPX")
nIPOPT8 = sum( timeIPOPT8 .< timeALPX8 )
nALPX8  = sum( timeIPOPT8 .> timeALPX8 )
println("tol 8 : IPOPT ", nIPOPT8," - ", nALPX8," ALPX")

ratios6 = timeIPOPT6 ./ timeALPX6
ratios8 = timeIPOPT8 ./ timeALPX8

figure()
hist( log10.(max.(1e-6,ratios6)), bins=20, histtype="step", label="ϵ = 1e-6 : IPOPT ($nIPOPT6) / ALPX ($nALPX6)", linewidth=2 )
hist( log10.(max.(1e-6,ratios8)), bins=20, histtype="step", label="ϵ = 1e-8 : IPOPT ($nIPOPT8) / ALPX ($nALPX8)", linewidth=2 )
legend(loc="upper left")
xlabel("log10(time ratio)")
xlim(-4,2)
ymin, ymax = ylim()
vlines(0,ymin,ymax)
ylim(ymin,ymax)
gcf()
savefig("/home/alberto/Documents/Bazinga.jl/test/data/cutest_time_ratio_hist.pdf")

#using BenchmarkProfiles
#performance_profile(P, string.(solvers))
