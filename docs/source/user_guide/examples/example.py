import os

import yaw
from yaw.examples import PATH  # included example data based on 2dFLenS

cache_dir = "cache_example"  # create as needed
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

reference_path = PATH.data
ref_rand_path = PATH.rand
unknown_path = PATH.data

patch_num = 11

cat_ref_rand = yaw.Catalog.from_file(
    cache_directory=f"{cache_dir}/ref_rand",
    path=ref_rand_path,
    ra_name="RA",
    dec_name="Dec",
    weight_name="wei",  # optional
    redshift_name="redshift",  # required for reference
    # patch_centers=None,
    # patch_name=None,
    patch_num=patch_num,
    # degrees=True,
    overwrite=True,  # aborts if cache exists, default: False
    progress=True,  # shows a progress bar, default: False
)

# extract the patch centers to use these for all following catalogs
patch_centers = cat_ref_rand.get_centers()

cat_reference = yaw.Catalog.from_file(
    cache_directory=f"{cache_dir}/reference",
    path=reference_path,
    ra_name="RA",
    dec_name="Dec",
    weight_name="wei",  # optional
    redshift_name="redshift",  # required for reference
    patch_centers=patch_centers,  # use previously computed centers
    # patch_name=None,
    # patch_num=None,
    # degrees=True,
    overwrite=True,  # aborts if cache exists, default: False
    progress=True,  # shows a progress bar, default: False
)

cat_unknown = yaw.Catalog.from_file(
    cache_directory=f"{cache_dir}/unknown",
    path=unknown_path,
    ra_name="RA",
    dec_name="Dec",
    weight_name="wei",  # optional
    # we don't know the redshifts here, so we skip the argument
    patch_centers=patch_centers,  # use previously computed centers
    # patch_name=None,
    # patch_num=None,
    # degrees=True,
    overwrite=True,  # aborts if cache exists, default: False
    progress=True,  # shows a progress bar, default: False
)

cat_unk_rand = None  # would be constructed same as cat_unknown

config = yaw.Configuration.create(
    rmin=100.0,  # can also be a list of lower scale limits
    rmax=1000.0,  # can also be a list of upper scale limits
    # unit="kpc"  # defaults to angular diameter distance, but angles and
    # comoving transverse distance are supported
    # rweight=None,     # if you want to weight pairs by scales
    # resolution=None,  # resolution of weights in no. of log-scale bins
    zmin=0.15,
    zmax=0.7,
    num_bins=11,
    # method="linear",
    # edges=None,  # provide your custom bin edges
)

cts_ss_list = yaw.autocorrelate(
    config,
    cat_reference,
    cat_ref_rand,
    progress=True,  # shows a progress bar, default: False
)

cts_sp_list = yaw.crosscorrelate(
    config,
    cat_reference,
    cat_unknown,
    ref_rand=cat_ref_rand,
    unk_rand=cat_unk_rand,
    progress=True,  # shows a progress bar, default: False
)

cts_ss = cts_ss_list[0]
cts_sp = cts_sp_list[0]

ncc = yaw.RedshiftData.from_corrfuncs(
    cross_corr=cts_sp,
    ref_corr=cts_ss,
    # unk_corr=None,
)

# or even with estimated normalisation
ax = ncc.normalised().plot()
ax.figure.savefig("test.png")
