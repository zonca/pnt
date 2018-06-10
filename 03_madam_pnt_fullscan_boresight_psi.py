#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import ctypes
import os
import shutil
import sys

import quaternionarray as qa

_madam = ctypes.CDLL("libmadam.so")


def compute_pol_weigths(psi):
    spsi = np.sin(psi)
    cpsi = np.cos(psi)
    cf = 1. / (cpsi ** 2 + spsi ** 2)
    return (cpsi ** 2 - spsi ** 2) * cf, 2 * cpsi * spsi * cf


def dict2parstring(d):

    s = ""
    for key, value in d.items():
        if key in ["detset", "detset_nopol", "survey"]:
            for separate_value in value:
                s += "{} = {};".format(key, separate_value)
        else:
            s += "{} = {};".format(key, value)

    return s.encode("ascii")


def dets2detstring(dets):

    s = ""
    for d in dets:
        s += "{};".format(d)

    return s.encode("ascii")


if __name__ == "__main__":

    """
    When madam.py is ran directly, it performs a quick check of the code by making a hit map and a binned map
    """

    from mpi4py import MPI
    import numpy as np

    np.random.seed(123456)
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    try:
        import healpy as hp
    except:
        hp = None
        if itask == 0:
            print("Warning: unable to import healpy. Output maps are not checked.")

    fcomm = comm.py2f()

    if itask == 0:
        print("Running with ", ntask, " MPI tasks")

    print("Calling Madam")

    nside = 64
    npix = 12 * nside ** 2
    fsample = 1
    nsamp = 3600 * 2  # number of time ordered data samples
    nnz = 3  # number or non zero pointing weights, typically 3 for IQU

    timestamps = np.zeros(nsamp, dtype="double")
    timestamps[:] = np.arange(nsamp)

    dets = ["1A", "1B", "2A", "2B"]
    detstring = dets2detstring(dets)

    ndet = len(dets)

    x_axis, y_axis, z_axis = np.eye(3)

    # Earth

    angle_each_day = np.radians(360 / 365.25)
    angles = timestamps * angle_each_day / 3600 / 24
    rot_earth_orbit = qa.rotation(z_axis, angles)

    # Precession

    prec_period_seconds = 1 * 3600
    prec_ang_speed = 2 * np.pi / prec_period_seconds
    rot_prec_opening = qa.rotation(z_axis, -np.radians(40))
    prec_angles = (timestamps * prec_ang_speed) % (2 * np.pi)
    rot_prec = qa.mult(qa.rotation(x_axis, prec_angles), rot_prec_opening)

    # Spin

    spin_period_seconds = 60
    spin_ang_speed = 2 * np.pi / spin_period_seconds
    spin_angles = (timestamps * spin_ang_speed) % (2 * np.pi)
    rot_opening = qa.rotation(z_axis, -np.radians(10))
    rot_spin = qa.mult(qa.rotation(x_axis, spin_angles), rot_opening)

    # Total quaternions to boresight

    bore_quat = qa.norm(qa.mult(rot_earth_orbit, qa.mult(rot_prec, rot_spin)))

    bore_v = qa.rotate(bore_quat, x_axis)
    pix_1det = hp.vec2pix(nside, bore_v[:, 0], bore_v[:, 1], bore_v[:, 2], nest=True)

    pixels = np.tile(pix_1det, ndet)

    # polarization weights

    bore_v_proj_ortog = np.hstack(
        [-bore_v[:, [1]], bore_v[:, [0]], np.zeros(len(bore_v))[:, np.newaxis]]
    )
    bore_v_proj_ortog /= np.linalg.norm(bore_v_proj_ortog, axis=1)[:, np.newaxis]

    local_north_v = np.cross(bore_v, bore_v_proj_ortog)
    bore_quat_inv = qa.inv(bore_quat)
    local_north_v_fp = qa.rotate(bore_quat_inv, local_north_v)
    psi_ref = np.arccos(qa.arraylist_dot(local_north_v_fp, z_axis).flatten())
    psi_ref[np.isnan(psi_ref)] = 0
    psi = {
        "1A": psi_ref,
        "1B": psi_ref + np.pi / 2,
        "2A": psi_ref + np.pi / 4,
        "2B": psi_ref + np.pi * 3 / 4,
    }

    del pix_1det, bore_v, rot_spin, bore_quat_inv, local_north_v_fp, local_north_v, bore_v_proj_ortog

    pars = {}
    pars["base_first"] = 60.0
    pars["fsample"] = fsample
    pars["nside_map"] = nside
    pars["nside_cross"] = nside // 2
    pars["nside_submap"] = nside // 4
    pars["write_map"] = True
    pars["write_binmap"] = True
    pars["write_matrix"] = True
    pars["write_wcov"] = True
    pars["write_hits"] = True
    pars["write_leakmatrix"] = True
    pars["kfilter"] = True
    pars["diagfilter"] = 0
    pars["file_root"] = "madam_pytest"
    pars["path_output"] = "./pymaps/"
    pars["iter_max"] = 100
    # pars["nsubchunk"] = 2
    pars["allreduce"] = True

    # pars[ 'detset' ] = ['LFI27 : LFI27M, LFI27S',
    #                    'LFI28 : LFI28M, LFI28S']
    # pars["survey"] = [
    #    "hm1 : {} - {}".format(0, nsamp / 2),
    #    # 'hm2 : {} - {}'.format(nsamp/2, nsamp),
    #    # 'odd : {} - {}, {} - {}'.format(0, nsamp/4, nsamp/2, 3*nsamp/4),
    #    # 'even : {} - {}, {} - {}'.format(nsamp/4, nsamp/2, 3*nsamp/4, nsamp)
    # ]
    # pars["bin_subsets"] = True

    if itask == 0:
        shutil.rmtree("pymaps", ignore_errors=True)
        os.mkdir("pymaps")
        for fn in ["pymaps/madam_pytest_hmap.fits", "pymaps/madam_pytest_bmap.fits"]:
            if os.path.isfile(fn):
                print("Removing old {}".format(fn))
                os.remove(fn)

    parstring = dict2parstring(pars)

    weights = np.ones(ndet, dtype="double")

    qw, uw = {}, {}
    for det in dets:
        qw[det], uw[det] = compute_pol_weigths(psi[det])

    pixweights = np.ones(ndet * nsamp * nnz, dtype="double")
    signal = np.zeros(ndet * nsamp, dtype="double")

    for idet, det in enumerate(dets):
        pixweights[idet * nsamp * nnz + 1 : (idet + 1) * nsamp * nnz : nnz] = qw[det]
        pixweights[idet * nsamp * nnz + 2 : (idet + 1) * nsamp * nnz : nnz] = uw[det]

        signal[idet * nsamp : (idet + 1) * nsamp] = 2 + 1 * qw[det] + 3 * uw[det]

    nperiod = 1  # number of pointing periods

    periods = np.zeros(nperiod, dtype=np.int64)

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)
    psdstarts = np.zeros(npsdtot)
    npsdbin = 10
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)

    _madam.destripe(
        fcomm,
        ctypes.c_char_p(parstring),
        ctypes.c_long(ndet),
        ctypes.c_char_p(detstring),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_long(nsamp),
        ctypes.c_long(nnz),
        timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        pixweights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_long(nperiod),
        periods.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        npsd.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        ctypes.c_long(npsdtot),
        psdstarts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_long(npsdbin),
        psdfreqs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_long(npsdval),
        psdvals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    if itask == 0 and hp is not None:

        madam_hmap = hp.read_map("pymaps/madam_pytest_hmap.fits")
        assert madam_hmap.sum() == nsamp * ndet
        madam_bmap = hp.read_map("pymaps/madam_pytest_bmap.fits", (0, 1, 2))
        good = madam_hmap > 0
        ones = np.ones(good.sum(), dtype=np.double)
        np.testing.assert_allclose(madam_bmap[0][good], 2 * ones)
        np.testing.assert_allclose(madam_bmap[1][good], 1 * ones)
        np.testing.assert_allclose(madam_bmap[2][good], 3 * ones)
