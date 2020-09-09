using CSV
using DataFrames, Query
using Plots

ipopt6_filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_ipopt_6.csv"
alpx6__filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_alpx__6.csv"
ipopt8_filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_ipopt_8.csv"
alpx8__filename = "/home/alberto/Documents/Bazinga.jl/test/data/cutest_alpx__8.csv"

df_ipopt = CSV.File(ipopt_filename) |> DataFrame
df_alpx  = CSV.File(alpx__filename) |> DataFrame

stats = Dict( :ALPX__6 => CSV.File(alpx6__filename) |> DataFrame,
              :IPOPT_6 => CSV.File(ipopt6_filename) |> DataFrame,
              #:ALPX__8 => CSV.File(alpx8__filename) |> DataFrame,
              #:IPOPT_8 => CSV.File(ipopt8_filename) |> DataFrame,
            )





cost(df) = (df.status .!= "first_order") * Inf + df.time

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

dfs = (stats[s] for s in solvers)
P = hcat([cost(df) for df in dfs]...)
logP = log10.(P)
histogram(logP, label=hcat([string(s) for s in solvers]...), fillalpha=0.8)

# using BenchmarkProfiles
#performance_profile(P, string.(solvers))
