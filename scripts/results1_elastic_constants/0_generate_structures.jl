using ASEconvert

output = joinpath(@__DIR__, "structures")

ase.io.write(joinpath(output, "C.extxyz"), ase.build.bulk("C", "diamond"))
