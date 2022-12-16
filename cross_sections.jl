
using DecFP

# all formulas from "Imaging Dark Matter with High-Energy Cosmic Neutrinos" (Mar 2021)
# https://arxiv.org/pdf/1703.00451.pdf


abstract type Interaction end
abstract type ScalarScalar <: Interaction end
abstract type FermionVector <: Interaction end
abstract type ScalarFermion <: Interaction end

function get_σ(interaction_type::Type{T}) where (T <: ScalarScalar)
    return σ_ScalarScalar
end
function get_σ(interaction_type::Type{T}) where (T <: FermionVector)
    return σ_FermionVector
end
function get_σ(interaction_type::Type{T}) where (T <: ScalarFermion)
    return σ_ScalarFermion
end


function get_dσ_dE(interaction_type::Type{T}) where (T <: ScalarScalar)
    return dσ_dE_ScalarScalar
end
function get_dσ_dE(interaction_type::Type{T}) where (T <: FermionVector)
    return dσ_dE_FermionVector
end
function get_dσ_dE(interaction_type::Type{T}) where (T <: ScalarFermion)
    return dσ_dE_ScalarFermion
end


# ================ Cross-Sections ===================

# y = (1 - x) = 1 - cos(θ)
get_y(Eν_i, Eν_f; mχ) = mχ * (1/Eν_f - 1/Eν_i)

# dx/dEν_f 
get_dx_dE(Eν_f; mχ) = mχ / Eν_f^2

# ------------------------------------

# scalar dark matter + scalar mediator
# formula in the lab frame
# all inputs assumed to have units of eV. 
#   (!) includes the coupling g 
function σ_ScalarScalar(Eν; g, mϕ, mχ)

    # # convert to Dec128 for precision
    # Eν = Dec128(Eν)
    # g = Dec128(g)
    # mϕ = Dec128(mϕ)
    # mχ = Dec128(mχ)
    
    term_1 = mϕ^2 * (2Eν + mχ)
    term_2 = (2Eν * mϕ^2) + (4Eν^2 * mχ) + (mϕ^2 * mχ)    
    log_r = log(term_1) - log(term_2)

    num = - g^2 * ( (4Eν^2 * mχ) + (term_2 * log_r) )
    denom = 64π * Eν^2 * mχ^2 * term_2

    # return num/denom
    return Float64(num/denom)
end

function dσ_dE_ScalarScalar(Eν_i, Eν_f; g, mχ, mϕ)

    y = get_y(Eν_i, Eν_f; mχ)
    dx_dE = get_dx_dE(Eν_f; mχ) 

    # dσ/dx
    frac_2_num = g^2/16π * y * Eν_i^2 * mχ
    frac_2_denom = (y*Eν_i + mχ) * (y*Eν_i*mϕ^2 + mχ*(mϕ^2 + 2y*Eν_i^2))^2

    # dσ/dEν_f = dσ/dx * dx/dEν_f
    return dx_dE * frac_2_num/frac_2_denom
end

# ------------------------------------

# fermion dark matter + vector mediator
# formula in the lab frame

# g² = g²g'² is unitless
function σ_FermionVector(E; g, mϕ, mχ)

    A = 2E + mχ             # ~ E
    B = 4E^2*mχ + mϕ^2*A    # ~ E³

    term_1 = (mϕ^2 + mχ*A) * ( 2log(mϕ) + log(A) - log(B) )      # ~ E²
    term_2 = 4E^2 * (1 + mχ^2/mϕ^2 - 2E*(E*mχ^2 + B)/(A*B))      # ~ E²

    return g^2/16π * (term_1 + term_2)/(E^2 * mχ^2)              # ~ 1/E²
end

function dσ_dE_FermionVector(Eν_i, Eν_f; g, mχ, mϕ)
    y = get_y(Eν_i, Eν_f; mχ)
    dx_dE = get_dx_dE(Eν_f; mχ)   
    
    E = Eν_i
    num = E^2 * mχ^2 * ( y*(2-y)*E*mχ + y^2*E^2 + (2-y)*mχ^2 )
    denom = (y*E + mχ) * (mχ * 2y*E^2 + mϕ^2*(mχ + y*E))^2

    return dx_dE * g^2/4π * num/denom
end

# ------------------------------------

# scalar dark matter + fermion mediator
# formula in the lab frame

# g² = (g²)^2 is unitless
function σ_ScalarFermion(E; g, mϕ, mχ)

    A = mχ + 2E             # ~ E
    B = mχ - 2E             # ~ E

    term_1 = 8E^2 * mχ / A * (mχ*A - mϕ^2)^2                # ~ 1/E^2
    term_2 = 4/(mχ * B - mϕ^2)                              # ~ 1/E^2
    term_3 = 8/(mχ * A - mϕ^2)                              # ~ 1/E^2

    term_4 = 3/E^2 - (6mϕ^2 + 2mχ * B)/(Emχ * (mχ*A - mϕ^2))    # ~ 1/E^2 
    log_term = log1p( 4E^2 * mχ/(mϕ^2 * A - mχ^3) )

    return g^2/64π * (term_1 + term_2 + term_3 + term_4*log_term)
end

# g² = (g²)^2 is unitless
function dσ_dE_ScalarFermion(Eν_i, Eν_f; g, mχ, mϕ)
    y = get_y(Eν_i, Eν_f; mχ)
    dx_dE = get_dx_dE(Eν_f; mχ)                                     # ~ 1/E   
    
    E = Eν_i

    num = E^4 * mχ^5 * (2-y) * ( y*E + 2mχ )^2                      # ~ E^11

    denom_1 = (mχ*(2E + mχ) - mϕ^2)^2                               # ~ E^4
    denom_2 = (y*E + mχ)^3                                          # ~ E^3
    denom_3 = ( E*(-y*mϕ^2 - (2-y)*mχ^2) + mχ^3 - mχ*mϕ^2 )^2       # ~ E^6

    return g^2/4π * dx_dE * num/(denom_1 * denom_2 * denom_3)  

end