from mpi4py import MPI
import ctypes
import os
import sys

# Sets up ctypes library

_libdir = os.path.join(os.path.dirname(__file__), 'cmp_c_library/cmake-build-release')
_libname = 'libcmp_c_library'

# try:
print(os.path.join(_libdir, "{}.so".format(_libname)))
_lib = ctypes.CDLL(os.path.join(_libdir, "{}.so".format(_libname)))
# except OSError:
#     try:
#         _lib = ctypes.CDLL(os.path.join(_libdir, "lib{}.dylib".format(_libname)))
#     except OSError:
#         sys.exit('Missing lib{0}.so or lib{0}.dylib'.format(_libname))


# Sets up MPI utilities

if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    MPI_Comm = ctypes.c_int
else:
    MPI_Comm = ctypes.c_void_p

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master = rank == 0
world_size = comm.Get_size()


# Modified from https://dbader.org/blog/python-ctypes-tutorial-part-2
def wrap_function(funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = _lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func
