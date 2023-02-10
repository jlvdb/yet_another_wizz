class none:
    pass


class Scales:
    rweight = None
    rbin_num = 50


class AutoBinning:
    zbin_num = 30
    method = "linear"


class Backend:
    thread_num = None
    crosspatch = True  # check with 'init' parser
    rbin_slop = 0.01


class Configuration:
    scales = Scales
    binning = AutoBinning
    backend = Backend
    cosmology = None


class Resampling:
    method = "bootstrap"
    n_boot = 500
    global_norm = False  # check with 'nz' parser
    seed = 12345

backend = "scipy"
