using JSON3
using Printf
using Unitful
using UnitfulAtomic
include("eos.jl")

elements = ["Ba", "C", "O", "Ti", "Zn"]
versions = [1, 4, 2, 1, 4]
structures = ["BCC", "Diamond", "FCC", "SC"]

AE_FITS = JSON3.read("meta-GGA-ae-reference.json")["BM_fit_data"]

for metric in ["epsilon", "nu"]
    println("Now computing metric $metric")
    for (element, version) in zip(elements, versions)
        @printf "%-2s" element
        for structure in structures
            ae_fit = AE_FITS["$element-X/$structure"]
            ae_eos = (; volume0 = ae_fit["min_volume"],
                        e0 = 0.0,
                        bulk_modulus0 = ae_fit["bulk_modulus_ev_ang3"],
                        bulk_deriv0 = ae_fit["bulk_deriv"])
            
            pp_data = JSON3.read("$element/data_$element-$(structure)_60.0_v$version.json")
            pp_eos = eos_birch_murnaghan_fit(pp_data["volumes"] ./ austrip(1u"Å^3"),
                                             pp_data["energies"] ./ austrip(1u"eV"))
        
            if metric == "epsilon"
                ε = epsilon_metric(ae_eos, pp_eos)
                @printf " & %.2f" ε
            else
                ν = nu_metric(ae_eos, pp_eos)
                @printf " & %.2f" ν
            end
        end
        print(" \\\\\n")
    end
end
