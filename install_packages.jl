using Pkg

dependencies = [
    "DifferentialEquations", "Plots", "ArbNumerics",
    "CSV", "DataFrames", "RecursiveArrayTools",
    "PlotlyJS", "Distributed", "DelimitedFiles", "StaticArrays",
    "DiffEqGPU", "OrdinaryDiffEq", "BenchmarkTools", "Revise", "SymPy"
]

Pkg.add(dependencies)

