# Birch-Murnaghan equation-of-state fitting, following
# https://github.com/aiidateam/acwf-verification-scripts/blob/main/3-analyze/eos_utils/eosfit_31_adapted.py

using Polynomials
using Polynomials: derivative

function eos_birch_murnaghan_fit(volumes, energies)
    p = fit(volumes .^ (-2.0 / 3.0), energies, 3)
    deriv1 = derivative(p)
    deriv2 = derivative(deriv1)
    deriv3 = derivative(deriv2)

    # Select the local minimum among the extrema of deriv1 by a second derivative test.
    p_extrema = Polynomials.roots(deriv1)
    idx = findfirst(x -> x > 0 && deriv2(x) > 0, p_extrema)
    isnothing(idx) && error("BM fit failed: no minimum could be found")
    x = p_extrema[idx]
    volume0 = x^(-3.0 / 2.0)
    e0 = p(x)

    derivV2 = 4.0 / 9.0 * x^5.0 * deriv2(x)
    derivV3 = (-20.0 / 9.0 * x^(13.0 / 2.0) * deriv2(x)
               - 8.0 / 27.0 * x^(15.0 / 2.0) * deriv3(x))
    bulk_modulus0 = derivV2 / x^(3.0 / 2.0)
    bulk_deriv0 = -1 - x^(-3.0 / 2.0) * derivV3 / derivV2

    (; volume0, e0, bulk_modulus0, bulk_deriv0)
end

function eos_birch_murnaghan_energy(v; volume0, e0, bulk_modulus0, bulk_deriv0)
    r = (volume0 / v)^(2.0 / 3.0)
    e0 + 9.0 / 16.0 * bulk_modulus0 * volume0 * (
        (r - 1.0)^3 * bulk_deriv0 + (r - 1.0)^2 * (6.0 - 4.0 * r))
end
