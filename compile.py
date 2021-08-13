from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("predictions",  ["predictions.py"]),
]

setup(
    name = 'E-Monitoring Predictions',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)