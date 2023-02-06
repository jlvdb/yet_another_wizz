setup_types = """# yet_another_wizz setup configuration

# NOTES:
# - Parameters are annotated with their types.
# - Keys in [braces] are optional and may be omitted entirely including their
#   subkeys.

backend:                        <str>

configuration:
    [backend]:
        [crosspatch]:           <bool>
        [rbin_slop]:            <float>
        [thread_num]:           <int>
    binning:
        # EITHER:
        zmin:                   <float> / <list[float]>
        zmax:                   <float> / <list[float]>
        zbin_num:               <int>
        [method]:               <str>
        # OR:
        zbins:                  <list[float]>
    [cosmology]:                <str>
    scales:
        rmin:                   <float>
        rmax:                   <float>
        [rbin_num]:             <int>
        [rweight]:              <float>

data:
    [cachepath]:                <directory>
    catalogs:
        [reference]:
            data:
                filepath:       <file>
                ra:             <str>
                dec:            <str>
                redshift:       <str>
                [patches]:      <str>
                [cache]:        <bool>
            [rand]:
                filepath:       <file>
                ra:             <str>
                dec:            <str>
                redshift:       <str>
                [patches]:      <str>
                [cache]:        <bool>
        [unknown]:
            data:
                filepath:
                    <int>:      <file>
                    ...
                ra:             <str>
                dec:            <str>
                [redshift]:     <str>
                [patches]:      <str>
                [cache]:        <bool>
            [rand]:
                filepath:
                    <int>:      <file>
                    ...
                ra:             <str>
                dec:            <str>
                [redshift]:     <str>
                [patches]:      <str>
                [cache]:        <bool>
"""


setup_default = """# yet_another_wizz setup configuration

backend: scipy

configuration:
    backend:
        crosspatch: true
        rbin_slop: 0.01
        thread_num: null
    binning:
        method: linear
        zbin_num: 30
        zmax: 2.0
        zmin: 0.01
        zbins: null
    cosmology: Planck15
    scales:
        rmin: 100.0
        rmax: 1000.0
        rweight: null
        rbin_num: 50

data:
    cachepath: null
    catalogs:
        reference:
            data:
                filepath: ...
                ra: ra
                dec: dec
                redshift: z
                patches: patch
                cache: true
            rand:
                filepath:
                    0: ...
                    1: ...
                ra: ra
                dec: dec
                redshift: z
                patches: patch
                cache: true
        unknown:
            data:
                filepath: ...
                ra: ra
                dec: dec
                redshift: null
                patches: patch
                cache: true
            rand:
                filepath:
                    0: ...
                    1: ...
                ra: ra
                dec: dec
                redshift: null
                patches: patch
                cache: true
"""
