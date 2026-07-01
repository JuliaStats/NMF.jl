using Documenter
using NMF

DocMeta.setdocmeta!(NMF, :DocTestSetup, :(using NMF); recursive=true)

makedocs(
    modules = [NMF],
    sitename = "NMF.jl",
    authors = "JuliaStats contributors",
    checkdocs = :public,
    pages = [
        "Home" => "index.md",
        "High-level interface" => "interface.md",
        "Initialization" => "initialization.md",
        "Algorithms" => "algorithms.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaStats/NMF.jl.git",
    push_preview = true,
)
