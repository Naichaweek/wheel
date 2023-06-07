from distutils.core import setup
from Cython.Build import cythonize

#--------------------------------------------------------#
#         将python文件拖入setup.py同目录下
#  按下 python setup.py build_ext  --inplace，即可生成C文件
#--------------------------------------------------------#

setup(ext_modules = cythonize("evaluation.py"))
