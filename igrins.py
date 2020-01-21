%pylab
import astropy.units as u
import matplotlib.pylab as plt
import numpy as np 
import os 
import pandas as pd

from AFS_AFLS import afs
from astropy.io import fits
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import progressbar 

import matplotlib 
font = {'size'   : 14}
matplotlib.rc('font', **font)

class igrins_spec():
    def __init__(self, spec_file='kTau/SDCH_20180102_0039.spec.fits', 
                       wave_file=None, 
                       sn_file=None):
        self.spec_file = spec_file
        spec_hdu = fits.open(spec_file)
        self.hdr = spec_hdu[0].header
        self.spec = spec_hdu[0].data
        if len(spec_hdu) >= 2:
            self.wave = spec_hdu[1].data
        else:
            if wave_file == None:
                self.wave_file = spec_file.replace('spec', 'wave')
            else:
                self.wave_file = wave_file
            wave_hdu = fits.open(self.wave_file)
            self.wave = wave_hdu[0].data
        if np.nanmedian(self.wave) > 1000:
            self.wave /= 1000.
        if sn_file == None:
            self.sn_file = spec_file.replace('spec', 'sn')
        else: 
            self.sn_file = sn_file
        try:
            sn_hdu = fits.open(self.sn_file)
            self.sn = sn_hdu[0].data
        except:
            self.sn = np.ones_like(self.wave)
        self.zenith_angle = np.arccos(1./self.hdr['AMSTART'])*u.rad.to(u.deg)
        self.nord, self.npx = self.spec.data.shape
        self.band = self.hdr['BAND'].strip()
        self.resolution = np.zeros([self.nord, self.npx-1])
        for jj in np.arange(0, self.nord):
            self.resolution[jj, :] = self.wave[jj, 0::-1] / np.diff(self.wave[jj,:]) / 3.3
        self.res_avg = np.average(self.resolution, axis=0)*0.8
        self.res_coeffs = np.polyfit(np.arange(0, self.npx-1), self.res_avg, 2)
        
    def plot_spec(self, orders=None):
        if orders == None:
            orders = np.arange(0, self.nord)
        plt.figure()
        for jj in orders:
            plt.plot(self.wave[jj, :], self.spec[jj, :])
            plt.xlabel('Wavelength ($\mu$m)')
            plt.ylabel('Counts')
            plt.tight_layout();

    def flatten_spec(self, trim_a=200, trim_b=200):
        flat = np.zeros([self.nord, self.npx - trim_a - trim_b])
        blaze = np.zeros_like(flat)
        wave = np.zeros_like(flat)
        bar = progressbar.ProgressBar(max_value=self.nord)
        intens = np.zeros_like(flat)
        ord_num = np.zeros_like(flat)
        for jj in np.arange(0, self.nord):
            bar.update(jj)
            df = pd.DataFrame(np.c_[self.wave[jj, trim_a:(self.npx-trim_b)], 
                self.spec[jj, trim_a:(self.npx-trim_b)]], columns=('wv', 'intens'))
            df = df.fillna(0)
            order = df.copy(deep=True)
            res = afs(order, a=4, q=0.9)
            intens[jj, :] = res['intens']
            flat[jj, :] = res['flat']
            blaze[jj, :] = res['blaze']
            wave[jj, :] = res['wv']
            ord_num[jj, :] = np.zeros_like(res['intens']) + jj
        df= pd.DataFrame(np.c_[wave.reshape(wave.shape[0]*wave.shape[1]), 
                               intens.reshape(wave.shape[0]*wave.shape[1]),
                               flat.reshape(wave.shape[0]*wave.shape[1]),
                               blaze.reshape(wave.shape[0]*wave.shape[1]),
                               ord_num.reshape(wave.shape[0]*wave.shape[1])], 
                                                        columns=('WAVE', 'INTENS', 'FLAT', 'BLAZE', 'ORDER'))
        df = df.sort_values('WAVE').fillna(1)
        self.flat = df

    def  plot_flat(self):
        plt.figure()
        plt.plot(self.flat['WAVE'], self.flat['FLAT'])
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel("Flattened Spectrum")
        plt.tight_layout();
    

s = igrins_spec('SDCH_20180122_0080.spec.fits')
s.plot_spec()
s.flatten_spec()
s.plot_flat()
   