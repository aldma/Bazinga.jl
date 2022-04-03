using JuliaFormatter
# https://github.com/domluna/JuliaFormatter.jl

# running this file should automatically format all *.jl files in the `folders`

the_formatter_function(path::String) = format_file(path, indent = 4, margin = 92)

basepath = @__DIR__
folders = [
    basepath,
    joinpath(basepath, "src"),
    joinpath(basepath, "src", "algorithms"),
    joinpath(basepath, "src", "projections"),
    joinpath(basepath, "src", "proxoperators"),
    joinpath(basepath, "src", "utilities"),
    joinpath(basepath, "test"),
    joinpath(basepath, "test", "definitions"),
    joinpath(basepath, "test", "problems"),
    joinpath(basepath, "demo"),
]

function format_jl_files_in_folder(folderpath::String)
    d = readdir(folderpath, join = true)
    bool = true
    for this in d
        if isfile(this) && this[end-2:end] == ".jl"
            @info "Formatting $(this)"
            tmp = the_formatter_function(this)
            bool = bool && tmp
        end
    end
    return bool
end

for f in folders
    format_jl_files_in_folder(f)
end
