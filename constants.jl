

# ================ constants ===================

# using natural units --> everything has units of eV

# note: comes from setting hbar*c = 1 = ... m eV
const m_to_1oeV = 5.06773093741e6   # = [1/eV] / [m]
const cm_to_1oeV = 1e-2 * m_to_1oeV

# const kg_to_eV = 5.62e35

const pc_to_m = 3.0857e16
const kpc_to_1oeV = 1e3 * pc_to_m * m_to_1oeV
 
const GeV_to_eV = 1e9
const eV_to_GeV = 1e-9

# Avogadro's number
const N_A = 6.0221415e+23