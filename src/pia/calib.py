# -*- coding: utf-8 -*-

import os
import warnings
from astropy.utils.exceptions import AstropyWarning
import numpy as np
from scipy import stats as sstats
from astropy import units as u
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
import ccdproc as ccdp
from stats import Stats


def run_calib(
    light: str = None,
    flat: str = None,
    bias: str = None,
    dark: str = None,
    filt: str = None,
    gain: float = None,
    ron: float = None,
):

    # Initialize the Process class with available data
    processing = Process(
        light=light,
        flat=flat,
        bias=bias,
        dark=dark,
        im_filter=filt,
        gain=gain,
        ron=ron
    )

    # Compute master bias if bias images are available
    if bias is not None:
        mbias = processing.compute_master_bias(bias)
    else:
        mbias = bias

    # Compute master dark if dark images are available
    # Correction with bias images if existing
    if dark is not None:
        mdark = processing.compute_master_dark(dark, mbias)
    else:
        mdark = dark

    # Compute master flat if flat images are available
    # Correction with bias + dark images if existing
    if flat is not None:
        mflat = processing.compute_master_flat(flat, mbias, mdark)
    else:
        mflat = flat

    # Light images correction
    light = processing.compute_light_images(
        light,
        mbias,
        mdark,
        mflat,
    )


class Process(Stats):
    warnings.simplefilter('ignore', category=AstropyWarning)

    mem_limit = 1e10  # bytes

    def __init__(self,
                 light: str,
                 flat: str = None,
                 bias: str = None,
                 dark: str = None,
                 gain: float = None,
                 ron: float = None,
                 im_filter: str = None,
                 ):

        self.light = ccdp.ImageFileCollection(light)
        if flat is not None:
            self.flat = ccdp.ImageFileCollection(flat)
        elif all(item is None for item in [flat, gain]):
            raise ValueError(
                "Light images have to be provided to compute gain"
            )
        else:
            self.flat = flat

        if bias is not None:
            self.bias = ccdp.ImageFileCollection(bias)
        elif bias is None and ron or gain is None:
            raise ValueError(
                "Bias images have to be provided to compute gain or read noise"
            )
        else:
            self.bias = bias

        if dark is not None:
            self.dark = ccdp.ImageFileCollection(dark)
        else:
            self.dark = dark

        # Gain and read noise
        if all(item is None for item in [ron, gain]):
            self.gain = self.compute_gain()
            self.ron = self.compute_ron(gain=self.gain * u.adu / u.electron)
        elif gain is None and ron is not None:
            self.ron = ron * u.electron
            self.gain = self.compute_gain(ron / u.electron)
        elif ron is None and gain is not None:
            self.gain = gain * u.electron / u.adu
            self.ron = self.compute_ron(gain * u.adu / u.electron)
        elif all(item is not None for item in [ron, gain]):
            self.gain = gain * u.electron / u.adu
            self.ron = ron * u.electron

        # Filter used
        if im_filter is None:
            self.im_filter = 'no_filter'
        else:
            self.im_filter = im_filter

    def imstat(self, frame_id: str, ccd_data: str, sig_clip=None):

        if type(frame_id) != list:
            frame_id = [frame_id]

        try:
            len(ccd_data)
        except:
            ccd_data = [ccd_data]

        mean, median, mode = [], [], []
        std_dev, mad, unit = [], [], []

        for ccd in ccd_data:
            if sig_clip is None:
                mea = np.mean(ccd)
                med = np.median(ccd)
                std = np.std(ccd)
            else:
                mea, med, std = sigma_clipped_stats(ccd, sigma=sig_clip)

            mean.append(mea)
            median.append(med)
            mode.append(sstats.mode(ccd)[0][0][0])
            std_dev.append(std)
            mad.append(mad_std(ccd))
            unit.append(str(ccd.unit))

        stat_tab = Table(
            [frame_id, mean, median, mode, std_dev, mad, unit],
            names=('frame id', 'mean', 'median', 'mode', 'std dev', 'mad', 'unit')
            )

        for field in ['mean', 'median', 'mode', 'std dev', 'mad']:
            stat_tab[field].info.format = '8.3f'
        print(stat_tab)

        return stat_tab

    def compute_master_bias(
        self,
        output_path
    ):
        # Load bias
        bias_images = self.load_bias()

        # compute master bias and print statistics
        master_bias = ccdp.combine(
            bias_images,
            method='average',
            sigma_clip=True,
            sigma_clip_low_thresh=5.,
            sigma_clip_high_thresh=5.,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=self.mem_limit
            )

        self.imstat('master bias', master_bias)

        # finalise master bias and save
        master_bias.meta['combined'] = True
        master_bias = ccdp.gain_correct(master_bias, self.gain)
        master_bias.write(output_path + '/mbias.fits', overwrite=True)

        return master_bias

    def compute_master_dark(
        self,
        output_path,
        master_bias=None,
    ):

        # Load dark
        dark_images = self.load_dark()

        # compute master dark and print statistics
        master_dark = ccdp.combine(
            dark_images,
            method='average',
            sigma_clip=True,
            sigma_clip_low_thresh=5.,
            sigma_clip_high_thresh=5.,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=self.mem_limit
        )

        self.imstat('master dark', master_dark)

        # finalise master flat and save
        master_dark.meta['combined'] = True

        master_dark = ccdp.gain_correct(master_dark, self.gain)

        if master_bias is not None:
            master_dark = ccdp.subtract_bias(master_dark, master_bias)
            self.imstat('master dark corrected from bias', master_dark)

        master_dark.write(
            output_path + '/mdark' + self.im_filter + '.fits',
            overwrite=True,
        )

        return master_dark

    def compute_master_flat(
        self,
        output_path,
        master_bias=None,
        master_dark=None,
    ):

        # Load flat
        flat_images = self.load_flat()

        # compute master flat and print statistics
        master_flat = ccdp.combine(
            flat_images,
            method='average',
            sigma_clip=True,
            sigma_clip_low_thresh=5.,
            sigma_clip_high_thresh=5.,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=self.mem_limit
        )

        self.imstat('master flat', master_flat)

        # finalise master flat and save
        master_flat.meta['combined'] = True

        master_flat = ccdp.gain_correct(master_flat, self.gain)

        if master_bias is not None:
            master_flat = ccdp.subtract_bias(master_flat, master_bias)
            self.imstat('master flat corrected from bias', master_flat)

        if master_dark is not None:
            master_flat = ccdp.subtract_dark(master_flat, master_dark)
            self.imstat('master flat corrected from dark + bias', master_flat)

        master_flat.write(
            output_path + '/mflat_' + self.im_filter + '.fits',
            overwrite=True,
            )

        return master_flat

    def compute_light_images(
        self,
        output_path,
        master_bias=None,
        master_dark=None,
        master_flat=None,
    ):

        path = os.path.join(output_path, 'corrected/')
        os.makedirs(path, exist_ok=True)

        # Load lights
        names, light_images = self.load_light(summary=True)

        lights = []
        for fname in names:
            ccd = ccdp.CCDData.read(fname, unit='adu')
            ccd = ccdp.ccd_process(
                ccd,
                error=True,
                master_bias=master_bias,
                dark_frame=master_dark,
                master_flat=master_flat,
                gain=self.gain,
                readnoise=self.ron
                )
            lights.append(ccd)

            ccd.write(
                path + os.path.split(fname)[-1],
                overwrite=True
            )

        # Find new location of corrected lights
        path_corr = ccdp.ImageFileCollection(path)
        light_corr = path_corr.files_filtered(
            imagetyp='Light frame',
            include_path=True)

        # Print statistics of light images
        self.imstat(light_corr, lights)
