# -*- coding: utf-8 -*-
"""
@author: t.maybour
"""

import si_prefix as si # pip install si-prefix

def si_exp(input):
    split = si.split(input)
    return split[1]

def si_scale(input):
    split = si.split(input)
    return 10**(-split[1])

def si_prefix(input):
    split = si.split(input)
    return si.prefix(split[1]).replace(' ', '')

def si_input_text(input):
    split = si.split(input)
    return '{:.3f}'.format(input * 10**(-split[1])) + ' ' + si.prefix(split[1])

def si_freq_text(input):
    return si_input_text(input) + 'Hz'

def si_length_text(input):
    return si_input_text(input) + 'm'

def si_time_text(input):
    return si_input_text(input) + 's'


##################################

def si_input_text2(input):
    split = si.split(input)
    return '{:.0f}'.format(input * 10**(-split[1])) + si.prefix(split[1])

def si_length_text2(input):
    return si_input_text2(input) + 'm'
