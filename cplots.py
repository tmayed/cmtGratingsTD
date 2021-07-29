import os
import glob
import cv2 as cv
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class cplots():

    def __init__(self):
        self.default_args()

    def default_args(self):
        #################################
        self.filename = ''
        self.save_dir = ''
        #################################
        self.width = 20
        self.height = 16
        self.axes_label_size = 35
        self.tick_label_size = 35
        self.axes_label_padding = 35
        self.margins = False
        #################################
        self.x_label = ''
        self.y_label = ''
        self.z_label = ''
        self.x_scale = 1
        self.y_scale = 1
        self.z_scale = 1
        #################################
        self.linewidth = 6
        self.linestyle = '-'
        self.major_tick_length = 5
        self.major_tick_width = 3
        self.tick_label_padding = 15
        #################################
        self.xlims = np.nan
        self.ylims = np.nan
        #################################
        self.title = False
        self.title_size = 35
        #################################
        self.vline = False
        self.vline_values = []
        self.vline_lw = []
        self.vline_color = []
        self.vline_style = []
        self.vline_size = 4
        #################################
        self.hline = False
        self.hline_values = []
        self.hline_lw = []
        self.hline_color = []
        self.hline_style = []
        self.hline_size = 4
        #################################

    def set_kwargs(self, kwargs):
        self.filename = kwargs.get('filename', self.filename)
        self.save_dir = kwargs.get('save_dir', self.save_dir)

    def axis_setup(self):
        plt.rc('axes', labelsize=self.axes_label_size)
        plt.rc('xtick', labelsize=self.tick_label_size)
        plt.rc('ytick', labelsize=self.tick_label_size)

    def set_save_dir(self):
        if (len(self.save_dir) > 0) :
            if (self.save_dir[-1] != '/'):
                self.save_dir = self.save_dir + '/'
        else:
            self.save_dir = 'graphs/'
        self.create_dir(self.save_dir)

    def create_dir(self, dir_in):
        if (dir_in[-1] != '/'):
            dir_in = dir_in + '/'
        if not os.path.exists(dir_in):
            os.makedirs(dir_in)

    def clean_dir(self, dir_in):

        if (dir_in[-1] != '/'):
            dir_in = dir_in + '/'

        self.create_dir(dir_in)

        to_clear = dir_in + '*'
        files = glob.glob(to_clear)
        for f in files:
            os.remove(f)

    def set_xlims(self):
        if isinstance(self.xlims, list):
            if len(self.xlims) == 2:
                plt.xlim([self.xlims[0],self.xlims[-1]])

    def set_ylims(self):
        if isinstance(self.ylims, list):
            if len(self.ylims) == 2:
                plt.ylim([self.ylims[0],self.ylims[-1]])

    def set_vlines(self):
        if self.vline:
            ylims = plt.gca().axes.get_ylim()
            for ii in range(0, len(self.vline_values)):
                plt.vlines(x=self.x_scale*self.vline_values[ii],ymin=ylims[0],ymax=ylims[-1],lw=self.vline_lw[ii],color=self.vline_color[ii],linestyle=self.vline_style[ii])

    def set_hlines(self):
        if self.hline:
            xlims = plt.gca().axes.get_xlim()
            for ii in range(0, len(self.hline_values)):
                plt.hlines(y=self.y_scale*self.hline_values[ii],xmin=xlims[0],xmax=xlims[-1],lw=self.hline_lw[ii],color=self.hline_color[ii],linestyle=self.hline_style[ii])

    ####################################
    ####################################
    ####################################

    def fig2cv(self,fig):
        fig.canvas.draw()
        img = cv.cvtColor(np.array(fig.canvas.renderer.buffer_rgba()), cv.COLOR_RGBA2BGR)
        plt.close()
        return img

    ####################################
    ####################################
    ####################################

    def line_plot(self, x, data, *args, **kwargs):

        self.set_kwargs(kwargs)
        self.set_save_dir()
        self.axis_setup()

        y = np.array(data)

        plt.figure(figsize=(self.width, self.height))
        plt.ticklabel_format(useOffset=False)

        self.set_xlims()
        self.set_ylims()

        if self.margins == False:
            plt.margins(x=0)
        plt.xlabel(self.x_label, labelpad=self.axes_label_padding)
        plt.ylabel(self.y_label, labelpad=self.axes_label_padding)

        plt.plot(self.x_scale*x,self.y_scale*y,linewidth=self.linewidth)

        self.set_vlines()
        self.set_hlines()

        if self.title != False:
            plt.title(self.title, fontsize=self.title_size)

        plt.tight_layout()
        plt.savefig(self.save_dir + self.filename + ".png")
        plt.close()

    ####################################
    ####################################
    ####################################

    def line_hoz_split_plot_save(self, x1, y1, x2, y2, *args, **kwargs):
        fig = self.line_hoz_split_plot(x1, y1, x2, y2, args, kwargs)
        self.set_save_dir()
        plt.savefig(self.save_dir + self.filename + ".png")
        plt.close()

    def line_hoz_split_plot(self, x1, y1, x2, y2, *args, **kwargs):

        self.set_kwargs(kwargs)
        self.axis_setup()

        ####################

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (self.width, self.height))
        fig.gca().clear()

        self.set_xlims()
        self.set_ylims()

        if self.margins == False:
            ax1.margins(x=0)
            ax2.margins(x=0)

        if self.use_title:
            fig.suptitle(self.title, fontsize=self.title_size)

        ax1.set_ylabel(self.y_label, labelpad=self.axes_label_padding)
        ax1.ticklabel_format(useOffset=False)
        ax1.plot(self.x_scale*x1,self.y_scale*y1,linewidth=self.linewidth,linestyle=self.linestyle)

        ax2.set_xlabel(self.x_label, labelpad=self.axes_label_padding)
        ax2.set_ylabel(self.y_label, labelpad=self.axes_label_padding)
        ax2.ticklabel_format(useOffset=False)
        ax2.plot(self.x_scale*x2,self.y_scale*y2,linewidth=self.linewidth,linestyle=self.linestyle)

        if isinstance(self.ylims, list):
            if len(self.ylims) == 2:
                ax1.set_ylim(self.ylims[0],self.ylims[1])
                ax2.set_ylim(self.ylims[0],self.ylims[1])

        if self.vline:
            for ii in range(0, len(self.vline_values)):
                ax1.vlines(x=self.x_scale*self.vline_values[ii],ymin=self.y_lim_values[0],ymax=self.y_lim_values[1],lw=self.vline_lw[ii],color=self.vline_color[ii],linestyle=self.vline_style[ii])
                ax2.vlines(x=self.x_scale*self.vline_values[ii],ymin=self.y_lim_values[0],ymax=self.y_lim_values[1],lw=self.vline_lw[ii],color=self.vline_color[ii],linestyle=self.vline_style[ii])

        if self.hline:
            xlims1 = ax1.get_xlim()
            xlims2 = ax2.get_xlim()
            for ii in range(0, len(self.hline_values)):
                ax1.hlines(y=self.y_scale*self.hline_values[ii],xmin=xlims1[0],xmax=xlims1[-1],lw=self.hline_lw[ii],color=self.hline_color[ii],linestyle=self.hline_style[ii])
                ax2.hlines(y=self.y_scale*self.hline_values[ii],xmin=xlims2[0],xmax=xlims2[-1],lw=self.hline_lw[ii],color=self.hline_color[ii],linestyle=self.hline_style[ii])

        return fig
