#!/bin/bash

DATADIR=/net/fohlen12/home/jlvdb/CC/POINTINGS/MICE2_KV450_magnification_on/idealized
OUTPUTDIR=${HOME}/EXPORT/Hendrik/cc_pipe_test
# I don't want to mess with your home
test ! -e ${OUTPUTDIR} && exit 1;

RAname=ra_gal_mag
DECname=dec_gal_mag
Zname=z_cgal_v
IDXname=unique_gal_id
SCALESmin=100,500    # running two
SCALESmax=1000,1500  # scales
Zmin=0.07
Zmax=1.42
ZBname=Z_B
WEIGHTspec=dec_gal_mag  # NOTE: DON'T DO THIS AT HOME
WEIGHTrand=DEC  # NOTE: DON'T DO THIS AT HOME
WEIGHTphot=recal_weight
NBINS=45
BINtype=comoving
ZBbins="0.101,0.301,0.501,0.701,0.901,1.201;0.101,1.201"
PAIRweighting=Y

### for compatibilty with the KV450 deep fields ###
# RAname=ALPHA_J2000
# DECname=DELTA_J2000
# Zname=z_spec
# IDXname=SeqNr
# SCALESmin=100,500
# SCALESmax=1000,1500
# Zmin=0.01
# Zmax=4.01
# ZBname=Z_B
# WEIGHTphot=recal_weight
# NBINS=40
# BINtype=comoving
# ZBbins="0.101,0.301,0.501,0.701,0.901,1.201;0.101,1.201"
# PAIRweighting=Y

### yaw_crosscorr ###
# run cross-correlation on pointings
mkdir -p ${OUTPUTDIR}/pointings_w_sp
for pointing in KV450_37p5_14p4 KV450_37p5_15p2 KV450_37p5_16p0 KV450_38p5_14p4 KV450_38p5_15p2 KV450_38p5_16p0 KV450_39p5_14p4 KV450_39p5_15p2 KV450_39p5_16p0
do
    echo "#### pointing $pointing ####"
    echo
    indir=${DATADIR}/${pointing}
    outdir=${OUTPUTDIR}/pointings_w_sp/${pointing}
    test -e ${outdir} && rm -r ${outdir}
    # You might not want to use --rand-weight and --rand-z.
    # I included these because my random catalogues have Z_B and recal_weight
    # cloned from the data.
    yaw_crosscorr \
        ${outdir} \
        --ref-file ${indir}/pointing_spec.cat \
        --ref-ra ${RAname} --ref-dec ${DECname} \
        --ref-z ${Zname} \
        --test-file ${indir}/pointing_phot.cat \
        --test-ra ${RAname} --test-dec ${DECname} \
        --test-z ${ZBname} --test-weight ${WEIGHTphot} \
        --rand-file ${indir}/pointing_phot.rand \
        --rand-ra RA --rand-dec DEC \
        --rand-z ${ZBname} --rand-weight ${WEIGHTphot} \
        --scales-min ${SCALESmin} \
        --scales-max ${SCALESmax} \
        --z-min ${Zmin} --z-max ${Zmax} \
        --ref-bin-no ${NBINS} --ref-bin-type ${BINtype} \
        --test-bin-edges ${ZBbins} \
        --pair-weighting ${PAIRweighting}
    echo
done

### yaw_autocorr ###
# run auto-correlation on pointings
mkdir -p ${OUTPUTDIR}/pointings_w_ss
for pointing in KV450_37p5_14p4 KV450_37p5_15p2 KV450_37p5_16p0 KV450_38p5_14p4 KV450_38p5_15p2 KV450_38p5_16p0 KV450_39p5_14p4 KV450_39p5_15p2 KV450_39p5_16p0
do
    echo "#### pointing $pointing ####"
    echo
    indir=${DATADIR}/${pointing}
    outdir=${OUTPUTDIR}/pointings_w_ss/${pointing}
    test -e ${outdir} && rm -r ${outdir}
    # The important argument is --param-file which loads the yaw_crosscorr
    # parameters (file in JSON format). The binning file is also created
    # by yaw_crosscorr and defines the bin edges.
    param_file=${OUTPUTDIR}/pointings_w_sp/${pointing}/yet_another_wizz.param
    binning_file=${OUTPUTDIR}/pointings_w_sp/${pointing}/binning.dat
    yaw_autocorr \
        ${outdir} \
        --output-suffix spec \
        --param-file ${param_file} \
        --binning-file ${binning_file} \
        --rand-file ${indir}/pointing_spec.rand \
        --rand-ra RA --rand-dec DEC \
        --rand-z ${Zname} --rand-weight ${WEIGHTrand} \
        --which ref
    # You should specify --cat-file, --cat-z, --rand-file and --rand-z
    # explicitly, leave --output-suffix and --which as they are.
    echo
done

### yaw_merge_pickles ###
test -e ${OUTPUTDIR}/correlation_pickles && rm -r ${OUTPUTDIR}/correlation_pickles
# concatenate the output of the cross-correlation runs
yaw_merge_pickles \
    -i ${OUTPUTDIR}/pointings_w_sp/*/ \
    -o ${OUTPUTDIR}/correlation_pickles
echo
# concatenate the output of the auto-correlation runs into the same folder
yaw_merge_pickles \
    -i ${OUTPUTDIR}/pointings_w_ss/*/ \
    -o ${OUTPUTDIR}/correlation_pickles
echo

### yaw_pickles_to_redshift ###
# produce cross-correlation n(z) with spec. bias correction from the pickles
KEYORDER="0.101z0.301 0.301z0.501 0.501z0.701 0.701z0.901 0.901z1.201 0.101z1.201"
test -e ${OUTPUTDIR}/nz_bias_corr_spec && rm -r ${OUTPUTDIR}/nz_bias_corr_spec
yaw_pickles_to_redshift \
    ${OUTPUTDIR}/correlation_pickles \
    --seed KV450 \
    --cov-order $KEYORDER \
    -o ${OUTPUTDIR}/nz_bias_corr_spec
# --cov-order will trigger computing a global covariance between bins.
echo

### yaw_plot ###
# make a scale comparison check plot
yaw_plot --auto-offset \
    --plot ${OUTPUTDIR}/nz_bias_corr_spec/kpc500t1500 l:'500-1500 kpc' \
    --plot ${OUTPUTDIR}/nz_bias_corr_spec/kpc100t1000 l:'100-1000 kpc' \
    -o ${OUTPUTDIR}/scale_comparison.png
echo
