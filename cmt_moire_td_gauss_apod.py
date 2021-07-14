# -*- coding: utf-8 -*-
"""
@author: t.maybour
"""

import time
import datetime

from scipy.integrate import solve_ivp
import numpy as np

import si_helper as si
import cv2 as cv
import cplots as cp

def main():

################################################
################################################
################################################

    cm = cm_class()

################################################
################################################
################################################

    project = 'gratings'
    sub_project = 'cmt_moire_td_gauss_apod'
    data_folder = 'data'

################################################
################################################
################################################

    ###############
    cm.plots = cp.cplots()
    cm.save_dir = 'moire_output/'
    cm.plots.clean_dir(cm.save_dir)
    ###############

    cm.c = np.float64(299792458) # speed of light
    cm.epsilon0 = np.float(8.85418782e-12)
    cm.mu0 = np.float(1.25663706e-6)
    cm.nu = np.sqrt(cm.mu0 / cm.epsilon0)

################################################
#################### INPUTS ####################
################################################

    cm.l0 = 1550e-9 # carrier wavelength (1/m)
    cm.pfbw = -1
    cm.intensity_cm2 = 1 # Intensity (W/cm^2)
    cm.xdamping = -1
    cm.xbuffer = -1
    cm.glen = 15e-3
    cm.extinction0 = 1e-9/4 # imaginary refractive index
    cm.dn = 1e-3
    cm.cond = 1e4
    cm.ascale = 16
    cm.ds = 1e-3
    cm.core = 10e-6

################################################

    fps = 25
    vid_len = 10
    cm.tv_nopo = fps * vid_len

################################################

    cm.k0 = np.float64((2 * np.pi) / cm.l0)
    cm.w0 = np.float64(cm.c * cm.k0)
    cm.f0 = np.float64(cm.w0 / (2 * np.pi))
    cm.n0 = cm.sellmeier_fused_silica(cm.l0) # real refractive index
    cm.vp = cm.c / cm.n0
    cm.beta0 = cm.n0 * cm.k0
    cm.absorb = cm.extinction0*cm.beta0/cm.n0

    cm.dB = cm.l0 / (2 * cm.n0) # grating period (m)
    cm.kappa0 = (cm.dn * cm.k0) / 2 # grating strength (1/m)

    # cm.int2amp = np.sqrt((2*cm.intensity*1e4)/(cm.c*cm.n0*cm.epsilon0))
    # cm.ampsqr2int = cm.c*cm.n0*cm.epsilon0/2
    # cm.ampsqr2int_cm2 = cm.c*cm.n0*cm.epsilon0/(2*1e4)
    # cm.int2power = (1e-5) * cm.ampsqr2int

    cm.ampsqr2int = (cm.c*cm.n0*cm.epsilon0/2)
    cm.ampsqr2int_cm2 = cm.c*cm.n0*cm.epsilon0/(2*1e4)
    cm.amp = np.sqrt((2*cm.intensity_cm2*1e4)/(cm.c*cm.n0*cm.epsilon0))
    cm.intensity = cm.ampsqr2int * cm.amp**2
    cm.intensity_cm2 = cm.ampsqr2int_cm2 * cm.amp**2
    cm.power = (cm.core**2) * cm.intensity

    print('------------------------')

    print('power :' + si.si_input_text(cm.power) + 'W')
    print('intensity :' + si.si_input_text(cm.intensity) + 'W/m^2')
    print('intensity :' + si.si_input_text(cm.intensity_cm2) + 'W/cm^2')

    # print(si.si_input_text(cm.int2power) + 'W')
    # print(si.si_input_text(cm.ampsqr2int) + 'W/m^2')
    # print(si.si_input_text(cm.ampsqr2int_cm2) + 'W/cm^2')

    if cm.pfbw < 0:
        cm.pfbw = 4*cm.n0*cm.dn*cm.c/(cm.l0*(4*cm.n0**2 - cm.dn))

    print('f0: ' + si.si_freq_text(cm.f0))
    print('n0: ' + str(cm.n0))
    print('absorb: '+str(cm.absorb*1e-2) + ' 1/cm')
    print('pfbw: ' + si.si_freq_text(cm.pfbw))

################################################
############### POSITION SETUP #################
################################################

    cm.width_div = 100

################################################
################ CREATE PULSE ##################
################################################

    cm.sigma = (np.pi * cm.pfbw) / (cm.c * np.sqrt(2*np.log(cm.width_div)))
    cm.pulse_tlen = (2 / (cm.c*cm.sigma)) * np.sqrt(2*np.log(cm.width_div))
    cm.pulse_xlen = cm.pulse_tlen * cm.vp

    print('pulse length: ' + si.si_length_text(cm.pulse_xlen))
    print('pulse time: ' + si.si_time_text(cm.pulse_tlen))

    ######################
    ######################
    ######################

    if cm.xdamping < 0:
        cm.xdamping = cm.pulse_xlen / 2

    if cm.xbuffer < 0:
        cm.xbuffer = cm.pulse_xlen

    if cm.glen < 0:
        cm.glen = cm.pulse_xlen

    cm.xlen = cm.glen + 2*cm.xbuffer + 2*cm.xdamping
    if cm.xlen <= 0:
        cm.xlen = cm.pulse_xlen

    cm.dx = cm.pulse_xlen / 1000
    cm.xv_nopo = int(cm.xlen/cm.dx) + 1
    cm.xv = np.linspace(-cm.xlen/2,cm.xlen/2,cm.xv_nopo,dtype=np.float64)
    cm.dx = np.abs(cm.xv[1]-cm.xv[0])

    cm.x0 = -(cm.glen + cm.xbuffer) / 2
    cm.xv_src = (np.abs(cm.xv-cm.x0)).argmin()

    cm.condv = np.zeros(cm.xv_nopo, dtype=np.float64)
    for ii in range(0, cm.xv_nopo):
        if (cm.xv[ii] >= cm.glen/2 + cm.xbuffer):
            cm.condv[ii] = cm.cond*(0.5 + 0.5*np.cos((np.pi/cm.xdamping)*(cm.xv[ii]-(cm.glen/2+cm.xbuffer+cm.xdamping))))
        elif (cm.xv[ii] <= -cm.glen/2 - cm.xbuffer):
            cm.condv[ii] = cm.cond*(0.5 + 0.5*np.cos((np.pi/cm.xdamping)*(-cm.xv[ii]-(cm.glen/2+cm.xbuffer+cm.xdamping))))

    cm.kappa = np.zeros(cm.xv_nopo, dtype=np.float64)
    cm.loss = np.zeros(cm.xv_nopo, dtype=np.float64)
    cm.shift = np.zeros(cm.xv_nopo, dtype=np.float64)
    cm.apod = np.zeros(cm.xv_nopo, dtype=np.float64)

    for ii in range(0, cm.xv_nopo):
        if np.abs(cm.xv[ii]) <= cm.glen/2:
            cm.loss[ii] = cm.absorb
            cm.kappa[ii] = cm.kappa0*np.cos(2*np.pi*cm.xv[ii]/cm.ds)
            cm.apod[ii] = np.exp(-cm.ascale*cm.xv[ii]**2/cm.glen**2)

################################################
################ SETUP TIMING ##################
################################################

    cm.tamp_scale = 1
    cm.tamp = 1

    cm.t0 = 1.5 * (cm.pulse_tlen / 2)
    cm.tlen = cm.t0 + np.abs(2*cm.x0) / cm.vp

    cm.tv = np.linspace(0,cm.tlen,cm.tv_nopo,np.float64)
    cm.dt = cm.dx / (2*cm.vp)

    cm.tpulse = np.zeros(cm.tv_nopo, dtype=np.float64)
    for ii in range(0, cm.tv_nopo):
        cm.tpulse[ii] = cm.tpulse_func(cm.tv[ii])

################################################
################ OUTPUT GRAPHS #################
################################################

    cm.plots.default_args()

    cm.plots.axes_label_size = 35
    cm.plots.tick_label_size = 35
    cm.plots.major_tick_length = 8
    cm.plots.major_tick_width = 3
    cm.plots.linewidth = 6

    ###################################

    cm.plots.x_scale = si.si_scale(np.max(cm.xv))
    cm.plots.x_label = 'Length (' + si.si_prefix(np.max(cm.xv)) + 'm)'

    cm.plots.line_plot(cm.xv, cm.condv, filename='condv', save_dir=cm.save_dir)
    cm.plots.line_plot(cm.xv, cm.kappa, filename='kappa', save_dir=cm.save_dir)
    cm.plots.line_plot(cm.xv, cm.apod, filename='apod', save_dir=cm.save_dir)

    ###################################

    cm.plots.x_scale = si.si_scale(np.max(cm.tv))
    cm.plots.x_label = 'Time (' + si.si_prefix(np.max(cm.tv)) + 's)'

    cm.plots.vline = True
    cm.plots.vline_values = [cm.t0-cm.pulse_tlen/2,cm.t0+cm.pulse_tlen/2]
    cm.plots.vline_lw = [2,1]
    cm.plots.vline_color = ['g','g']
    cm.plots.vline_style = ['--','--']
    cm.plots.use_title = True

    cm.plots.line_plot(cm.tv, cm.tpulse, filename='tpulse', save_dir=cm.save_dir)

################################################
################################################

    cm.dudz = np.zeros(cm.xv_nopo,dtype=np.complex128)
    cm.dvdz = np.zeros(cm.xv_nopo,dtype=np.complex128)

    cm.modes = np.zeros((2,cm.tv_nopo,cm.xv_nopo),dtype=np.complex128)
    cm.dfdt = np.zeros((2,cm.xv_nopo),dtype=np.complex128)

################################################
################################################

    cm.source = np.zeros(cm.xv_nopo, dtype=np.float64)
    cm.u0 = np.zeros(cm.xv_nopo,dtype=np.complex128)
    cm.v0 = np.zeros(cm.xv_nopo,dtype=np.complex128)
    cm.modes0 = np.concatenate((cm.u0,cm.v0), axis=0)

################################################
################################################

    sol = solve_ivp(cm.dfdt_func,[0,cm.t0],cm.modes0,t_eval=[0,cm.t0],rtol=1e-6,max_step=cm.dt,method='RK45')
    cm.max_ampsqr = np.real(np.abs(sol.y[cm.xv_src,1])**2)

    cm.tamp_scale = 1 / np.sqrt(cm.max_ampsqr)
    cm.tamp = cm.amp

################################################
################ RUN SIMULATION ################
################################################

    cm.source = np.zeros(cm.xv_nopo, dtype=np.float64)
    cm.u0 = np.zeros(cm.xv_nopo,dtype=np.complex128)
    cm.v0 = np.zeros(cm.xv_nopo,dtype=np.complex128)
    cm.modes0 = np.concatenate((cm.u0,cm.v0), axis=0)

    cm.total_timing = time.time()
    cm.timing = time.time()

    sol = solve_ivp(cm.dfdt_func,[cm.tv[0],cm.tv[-1]],cm.modes0,t_eval=cm.tv,rtol=1e-6,max_step=cm.dt,method='RK45')

    cm.current_time = time.time() - cm.total_timing
    print('sim time: ' + str(datetime.timedelta(seconds=cm.current_time)))

    for ii in range(0, cm.tv_nopo):
        cm.modes[0,ii,:] = sol.y[:cm.xv_nopo,ii]
        cm.modes[1,ii,:] = sol.y[cm.xv_nopo:,ii]

################################################
################ OUTPUT GRAPHS #################
################################################

    cm.plots.default_args()

    cm.plots.axes_label_size = 35
    cm.plots.tick_label_size = 35
    cm.plots.major_tick_length = 8
    cm.plots.major_tick_width = 3
    cm.plots.linewidth = 6

    cm.plots.x_scale = si.si_scale(np.max(cm.xv))
    cm.plots.x_label = 'Length (' + si.si_prefix(np.max(cm.xv)) + 'm)'

    intensity_field_fwd = cm.ampsqr2int_cm2*np.abs(cm.modes[0,:,:])**2
    intensity_field_bwd = cm.ampsqr2int_cm2*np.abs(cm.modes[1,:,:])**2

    int_max = np.max(intensity_field_fwd)

    cm.plots.y_scale = si.si_scale(int_max)
    cm.plots.y_label = 'Intensity (' + si.si_prefix(np.max(int_max)) + 'W/cm^2)'

    cm.plots.y_lims = True
    cm.plots.y_lim_values = [0, 1.2*cm.plots.y_scale*np.max(int_max)]

    cm.plots.vline = True
    cm.plots.vline_values = [-cm.x0-cm.xbuffer/2,-cm.x0,-cm.x0+cm.xbuffer/2,cm.x0-cm.xbuffer/2,cm.x0,cm.x0+cm.xbuffer/2]
    cm.plots.vline_lw = [1,3,1,1,3,1]
    cm.plots.vline_color = ['g','r','g','g','r','g']
    cm.plots.vline_style = ['--','--','--','--','--','--']
    cm.plots.use_title = True

    ################################################

    cm.total_timing = time.time()
    cm.timing = time.time()

    cm.plots.title = si.si_time_text(cm.tv[0])
    img = cm.plots.fig2cv(cm.plots.line_hoz_split_plot(cm.xv, intensity_field_fwd[0,:], cm.xv, intensity_field_bwd[0,:]))

    cm.vid_name = 'moire_output/cmt_moire_td_gauss_apod.mp4'
    cm.vid_fourcc = cv.VideoWriter_fourcc(*'mp4v')
    cm.vid_out = cv.VideoWriter(cm.vid_name, cm.vid_fourcc, fps, (img.shape[1],img.shape[0]), True)
    cm.vid_out.write(img)

    for ii in range(0,len(sol.t)):
        cm.plots.title = si.si_time_text(cm.tv[ii])
        img = cm.plots.fig2cv(cm.plots.line_hoz_split_plot(cm.xv, intensity_field_fwd[ii,:], cm.xv, intensity_field_bwd[ii,:]))
        cm.vid_out.write(img)
    cm.vid_out.release()

    cm.current_time = time.time() - cm.total_timing
    print('vid time: ' + str(datetime.timedelta(seconds=cm.current_time)))

#################################################################################
#################################################################################
#################################################################################

class cm_class():

    def sellmeier_fused_silica(self, l0):
        l0 = 1e6 * l0
        n_sqr = 1 + ((0.6961663*l0**2) / (l0**2 - 0.0684043**2)) + ((0.4079426*l0**2) / (l0**2 - 0.1162414**2)) + ((0.8974794*l0**2) / (l0**2 - 9.896161**2))
        return np.sqrt(n_sqr)

    def sellmeier_lithium_niobate(self, l0):
        l0 = 1e6 * l0
        n_sqr = 1 + ((2.6734*l0**2) / (l0**2 - 0.01764)) + ((1.2290*l0**2) / (l0**2 - 0.05914)) + ((12.614*l0**2) / (l0**2 - 474.60))
        return np.sqrt(n_sqr)

    def tpulse_func(self, t):
        return self.tamp * self.tamp_scale * np.exp(-0.5*(self.c**2)*(self.sigma**2)*(t-self.t0)**2)

    def xpulse_func(self, t):
        z = t * self.vp
        return self.xpulse_amp * np.exp(-0.5*(self.sigma**2)*(z-self.x0_input)**2)

    def dfdt_func(self,t,modes):

        self.u0[:] = modes[:self.xv_nopo]  # mode u - fwd
        self.v0[:] = modes[self.xv_nopo:]  # mode v - bwd

        self.dudz[:] = np.gradient(self.u0[:], self.xv, edge_order=2)
        self.dvdz[:] = np.gradient(self.v0[:], self.xv, edge_order=2)

        self.source[self.xv_src] = self.tpulse_func(t)

        self.dfdt[0,:] = self.vp*(self.source - self.dudz - (self.loss + self.condv)*self.u0 - self.kappa*self.apod*self.v0)
        self.dfdt[1,:] = self.vp*(self.dvdz - (self.loss + self.condv)*self.v0 + self.kappa*self.apod*self.u0)

        return np.concatenate((self.dfdt[0,:],self.dfdt[1,:]), axis=0)

#################################################################################
#################################################################################
#################################################################################

if __name__ == '__main__':
    main()
