# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:52:17 2019

@author: Andrew Wentzel
"""
import setuptools
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(name = 'cv_functions',
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize('cv_functions.pyx'),
      include_dirs = [numpy.get_include()]
      )