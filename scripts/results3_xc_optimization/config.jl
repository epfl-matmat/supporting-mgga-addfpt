get_config() = (;
    ref_functional  = "hse06",
    base_functional = [:gga_c_pbe],   # fixed correlation added alongside the learned exchange
    trainset        = [("diamond", 0), ("diamond", 3), ("diamond", 6)],
    loss_weights    = (; E=10.0, ρ=1.0, τ=1.0),
    tol_scf         = 1e-10,
    hidden_sizes    = (8,),
)
