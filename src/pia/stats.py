# -*- coding: utf-8 -*-

import warnings
from astropy.utils.exceptions import AstropyWarning
import numpy as np
from astropy import units as u
import ccdproc as ccdp
from astropy.nddata import CCDData


class Stats:

    warnings.simplefilter('ignore', category=AstropyWarning)

    def load_bias(self, summary: bool = False):

        bias_images = []
        bias_fnames = self.bias.files_filtered(
            imagetyp='Bias Frame',
            include_path=True)

        print("\n", "%2d bias frames found" % len(bias_fnames))

        bias_summ = self.bias.summary[self.bias.summary['imagetyp'] == 'Bias Frame']

        print(bias_summ)
        print(
            "\n",
            bias_summ[
                'date-obs',
                'file',
                'object',
                'exposure'
                ],
            "\n"
            )

        # load bias frames and print statistics
        for fname in bias_fnames:
            ccd = ccdp.CCDData.read(fname, unit='adu')
            bias_images.append(ccd)

        if summary:
            self.imstat(bias_fnames, bias_images)

        return bias_images

    def load_dark(self, summary: bool = False):

        # Identify dark frames
        type_dark = "Dark frame"

        dark_images = []
        dark_fnames = self.dark.files_filtered(
            imagetyp=type_dark,
            include_path=True)

        print("\n", "%2d dark frames found" % len(dark_fnames))

        dark_summ = self.dark.summary[self.dark.summary['imagetyp'] == type_dark]

        print(
            "\n",
            dark_summ['date-obs', 'file', 'object', 'filter', 'exposure'],
            "\n"
        )

        # load flat frames and print statistics
        for fname in dark_fnames:
            ccd = ccdp.CCDData.read(fname, unit='adu')
            dark_images.append(ccd)

        if summary:
            self.imstat(dark_fnames, dark_images)

        return dark_images

    def load_flat(self, summary: bool = False):

        # Identify flat-field frames for a selected filter
        type_flat = "Flat field"

        flat_images = []
        flat_fnames = self.flat.files_filtered(
            imagetyp=type_flat,
            include_path=True)

        print("\n", "%2d flat frames found" % len(flat_fnames))

        flat_summ = self.flat.summary[self.flat.summary['imagetyp'] == type_flat]
        print(flat_summ)

        print(
            "\n",
            flat_summ['date-obs', 'file', 'object', 'filter', 'exposure'],
            "\n"
        )

        # load flat frames and print statistics
        for fname in flat_fnames:
            ccd = ccdp.CCDData.read(fname, unit='adu')
            flat_images.append(ccd)

        if summary:
            self.imstat(flat_fnames, flat_images)

        return flat_images

    def load_light(self, summary: bool = False):

        # Identify light images
        type_light = "Light frame"

        sci_images = []
        sci_fnames = self.light.files_filtered(
            imagetyp=type_light,
            include_path=True
        )
        print("\n", "%2d science frames found" % len(sci_fnames))

        sci_summ = self.light.summary[self.light.summary['imagetyp'] == type_light]

        print(
            "\n",
            sci_summ['date-obs', 'file', 'object', 'filter', 'exposure'],
            "\n"
            )
        for fname in sci_fnames:
            ccd = ccdp.CCDData.read(fname, unit='adu')
            sci_images.append(ccd)

        if summary:
            self.imstat(sci_fnames, sci_images)

        return sci_fnames, sci_images

    def compute_gain(self, ron: float = None):

        # Load bias images
        bias = self.load_bias()
        if self.flat is not None:
            # Load flat images
            flat = self.load_flat()

            b1_b2 = CCDData.subtract(bias[0], bias[1])
            std_b1_b2 = np.mean([np.std(data) for data in b1_b2])

            f1_f2 = CCDData.subtract(flat[0], flat[1])
            std_f1_f2 = np.mean([np.std(data) for data in f1_f2])

            G = ((np.mean(flat[0]) + np.mean(flat[1])) - \
                 (np.mean(bias[0]) - np.mean(bias[1]))) / (std_f1_f2**2 - std_b1_b2**2)

        elif ron is None and self.flat is None:
            raise ValueError(
                "Flat images or read noise have to be provided to compute gain"
            )

        elif ron is not None and self.flat is None:
            G = ron * np.sqrt(2) / std_b1_b2

        return G * u.electron / u.adu

    def compute_ron(self, gain: float):

        # Load bias images
        im_bias = self.load_bias()

        b1_b2 = CCDData.subtract(im_bias[0], im_bias[1])
        std_b1_b2 = np.mean([np.std(data) for data in b1_b2])

        ron = gain * std_b1_b2 / np.sqrt(2)

        return ron * u.electron
