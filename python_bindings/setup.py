"""
setup.py — custom CMake-based build for the _foundationpose_cpp pybind11 extension.

The CMakeLists.txt already sets LIBRARY_OUTPUT_DIRECTORY to
  <source_root>/foundationpose_cpp/
so the .so lands in the Python package directory in-tree.
This makes `pip install -e .` work instantly after the first build.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Marker extension — sources are handled by CMake, not distutils."""

    def __init__(self, name: str, source_dir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class CMakeBuild(build_ext):
    """Run CMake + make to build the pybind11 extension."""

    def build_extension(self, ext: CMakeExtension) -> None:
        # Use a persistent build dir inside the source tree so that
        # compile_commands.json and incremental build artifacts survive across
        # `pip install -e .` invocations (setuptools' default build_temp is a
        # /tmp directory that is deleted after each run).
        build_temp = Path(ext.source_dir) / "build"
        build_temp.mkdir(parents=True, exist_ok=True)

        # Resolve where pybind11 lives so CMake can find it
        try:
            import pybind11
            pybind11_dir = pybind11.get_cmake_dir()
        except ImportError:
            pybind11_dir = None

        cmake_args = ["-DCMAKE_BUILD_TYPE=Release",
                      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
        if pybind11_dir:
            cmake_args.append(f"-Dpybind11_DIR={pybind11_dir}")

        # Explicitly target the running Python so pybind11 builds for the
        # correct version (important when inside a conda env).
        cmake_args.append(f"-DPYTHON_EXECUTABLE={sys.executable}")

        # When building inside a conda env, cmake picks up the conda
        # cross-compiler (x86_64-conda-linux-gnu-c++) which lives inside a
        # private sysroot and cannot link against the real system libs
        # (GLIBC_PRIVATE conflicts).  Force use of the native system compiler
        # and linker so it can reach both system headers/libs (glog, OpenCV, etc.)
        # and the conda Python library that pybind11 will locate automatically.
        for cxx in ["/usr/bin/g++", "/usr/bin/c++"]:
            if os.path.isfile(cxx):
                cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")
                break
        for cc in ["/usr/bin/gcc", "/usr/bin/cc"]:
            if os.path.isfile(cc):
                cmake_args.append(f"-DCMAKE_C_COMPILER={cc}")
                break
        # conda modifies PATH so its own `ld` may be found before /usr/bin/ld;
        # force the system linker to avoid sysroot conflicts.
        for ld in ["/usr/bin/ld"]:
            if os.path.isfile(ld):
                cmake_args.append(f"-DCMAKE_LINKER={ld}")
                break

        # Explicitly point at the system OpenCV cmake config; CMake won't find
        # it via CMAKE_PREFIX_PATH when the conda toolchain is active.
        system_opencv_dirs = [
            "/usr/lib/x86_64-linux-gnu/cmake/opencv4",
            "/usr/lib/aarch64-linux-gnu/cmake/opencv4",
            "/usr/share/opencv4",
        ]
        if "OpenCV_DIR" not in os.environ:
            for d in system_opencv_dirs:
                if os.path.isfile(os.path.join(d, "OpenCVConfig.cmake")):
                    cmake_args.append(f"-DOpenCV_DIR={d}")
                    break
        else:
            cmake_args.append(f"-DOpenCV_DIR={os.environ['OpenCV_DIR']}")

        build_args = ["--build", ".", "--", f"-j{os.cpu_count() or 4}"]

        # When running inside a conda env, conda injects CFLAGS/LDFLAGS with
        # rpath/sysroot settings that mix the conda and system libc, causing
        # GLIBC_PRIVATE link errors.  Strip those variables and remove the
        # conda env's bin dir from PATH so the system linker (ld) is found
        # instead of conda's, which points at a private sysroot.
        env = os.environ.copy()
        for var in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "CC", "CXX", "LD", "AS"):
            env.pop(var, None)
        conda_prefix = env.get("CONDA_PREFIX", "")
        if conda_prefix:
            path_dirs = env.get("PATH", "").split(":")
            path_dirs = [d for d in path_dirs if not d.startswith(conda_prefix)]
            env["PATH"] = ":".join(path_dirs)

        subprocess.run(
            ["cmake", ext.source_dir, *cmake_args],
            cwd=build_temp,
            check=True,
            env=env,
        )
        subprocess.run(
            ["cmake", *build_args],
            cwd=build_temp,
            check=True,
            env=env,
        )

        # CMakeLists.txt writes the .so directly into foundationpose_cpp/ in the
        # source tree (via LIBRARY_OUTPUT_DIRECTORY).  Setuptools' `develop` /
        # `build_ext --inplace` commands also expect the .so at the path returned
        # by get_ext_fullpath(); copy it there so both flows work.
        so_name = Path(self.get_ext_filename(ext.name)).name  # e.g. _foundationpose_cpp.cpython-310-*.so
        src_so = Path(ext.source_dir) / "foundationpose_cpp" / so_name
        dst_so = Path(self.get_ext_fullpath(ext.name))
        dst_so.parent.mkdir(parents=True, exist_ok=True)
        if src_so.exists() and src_so.resolve() != dst_so.resolve():
            shutil.copy2(src_so, dst_so)


setup(
    name="foundationpose-cpp",
    version="0.1.0",
    packages=["foundationpose_cpp"],
    package_data={"foundationpose_cpp": ["*.so", "*.pyd"]},
    ext_modules=[CMakeExtension("foundationpose_cpp._foundationpose_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
