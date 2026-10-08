#
# Equation of state utilities
#

using Polynomials
using Polynomials: derivative

function eos_birch_murnaghan_fit(volumes, energies; root_method=Polynomials.roots)
    # Based on https://github.com/aiidateam/acwf-verification-scripts/blob/main/3-analyze/eos_utils/eosfit_31_adapted.py
    p = fit(volumes.^(-2.0/3.0), energies, 3)
    deriv1 = derivative(p)
    deriv2 = derivative(deriv1)
    deriv3 = derivative(deriv2)

    # Find extrema by first derivative condition
    p_extrema = root_method(deriv1)  

    # Select local minimum by a second derivative test
    idx = findfirst(x -> x > 0 && deriv2(x) > 0, p_extrema)
    isnothing(idx) && error("BM fit failed: No minimum could be found")
    x = p_extrema[idx]
    volume0 = x^(-3.0/2.0)
    
    e0 = p(x)

    derivV2 = 4. / 9. * x^5. * deriv2(x)
    derivV3 = (-20. /  9. * x^(13. / 2.) * deriv2(x)
               - 8. / 27. * x^(15. / 2.) * deriv3(x))
    bulk_modulus0 = derivV2 / x^(3. / 2.)
    bulk_deriv0 = -1 - x^(-3. / 2.) * derivV3 / derivV2

    (; volume0, e0, bulk_modulus0, bulk_deriv0)
end

function eos_birch_murnaghan_energy(v; volume0, e0, bulk_modulus0, bulk_deriv0)
    r = (volume0 / v)^(2.0/3.0)
    (e0 + 9.0/16.0 * bulk_modulus0 * volume0 * (
            (r-1.)^3 * bulk_deriv0 + 
            (r-1.)^2 * (6.0 - 4.0 * r)))
end

function nu_metric(eos1, eos2; prefactor=100.0, weight_volume=1.0,
                   weight_bulk_modulus=1/20, weight_bulk_deriv=1/400)
    term(w, Ya, Yb) = (w * (Ya - Yb) / (Ya + Yb) * 2)^2
    prefactor * sqrt(
        term(weight_volume, eos1.volume0, eos2.volume0)
        + term(weight_bulk_modulus, eos1.bulk_modulus0, eos2.bulk_modulus0)
        + term(weight_bulk_deriv, eos1.bulk_deriv0, eos2.bulk_deriv0)
    )
end

"""
    epsilon_metric(eos1, eos2; prefactor=1.0, npoints=1001)

Compute the ε metric between two Birch–Murnaghan EOS fits, following the
definition in the ACWF verification effort (aiidateam/acwf-verification-scripts,
`quantities_for_comparison.py`).

`eos1` and `eos2` are named tuples with fields `volume0`, `bulk_modulus0`,
and `bulk_deriv0`. Volumes and bulk moduli of the two EOS must be in
consistent units (ε itself is dimensionless).
"""
function epsilon_metric(eos1, eos2; prefactor=1.0, npoints=1001)
    # Birch–Murnaghan energy with E0 = 0 (energy relative to the curve's minimum)
    bm(eos, V) = begin
        r = (eos.volume0 / V)^(2/3)
        9/16 * eos.bulk_modulus0 * eos.volume0 *
            ((r - 1)^3 * eos.bulk_deriv0 + (r - 1)^2 * (6 - 4r))
    end

    # Volume window: ±6% around the average equilibrium volume
    Vavg = (eos1.volume0 + eos2.volume0) / 2
    Vi, Vf = 0.94 * Vavg, 1.06 * Vavg
    ΔV = Vf - Vi

    # Composite Simpson's rule (npoints must be odd)
    n = isodd(npoints) ? npoints : npoints + 1
    h = ΔV / (n - 1)
    Vs = range(Vi, Vf; length=n)
    w = [i == 1 || i == n ? 1.0 : (iseven(i) ? 4.0 : 2.0) for i in 1:n] .* (h / 3)
    integrate(f) = sum(w[i] * f(Vs[i]) for i in 1:n)

    E1 = V -> bm(eos1, V)
    E2 = V -> bm(eos2, V)

    intdiff2 = integrate(V -> (E1(V) - E2(V))^2)        # ∫ (E1 - E2)² dV

    Ē1 = integrate(E1) / ΔV
    Ē2 = integrate(E2) / ΔV
    var1 = integrate(V -> (E1(V) - Ē1)^2)               # ∫ (E1 - ⟨E1⟩)² dV
    var2 = integrate(V -> (E2(V) - Ē2)^2)

    # Guard against tiny negative values from floating-point cancellation
    eps2 = abs(intdiff2) / sqrt(var1 * var2)

    prefactor * sqrt(eps2)
end
