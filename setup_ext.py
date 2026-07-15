"""
setup_ext.py — Build the C++ chess engine + MCTS pybind11 extension.

Usage:
    python setup_ext.py build_ext --inplace

Produces:
    engine/chess_ext*.pyd   (Windows)
    engine/chess_ext*.so    (Linux/macOS)

Requirements:
    pip install pybind11
    Windows: Visual Studio Build Tools, "Desktop development with C++"
             (cl.exe need NOT be on PATH — setuptools locates MSVC via vswhere)
    Linux:   g++ >= 9

Note: the C++ sources (chess_core/mcts_core/bindings) are part of the repo and
the extension builds normally. A prebuilt chess_ext*.pyd keeps the C++ backend
working without a compiler; a rebuild is only required after editing the .cpp.
"""

from __future__ import annotations

import os
import sys
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


# ── pybind11 include path ─────────────────────────────────────────────────

def pybind11_include() -> str:
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        raise RuntimeError(
            "pybind11 not found.  Run:  pip install pybind11"
        )


# ── Compiler flags ────────────────────────────────────────────────────────

class BuildExt(_build_ext):
    """Custom build_ext that sets platform-appropriate compile flags."""

    def build_extensions(self) -> None:
        # Fail early with an actionable message if no C++ toolchain is found,
        # instead of a cryptic compiler/link error mid-build.
        try:
            cc = self.compiler.compiler_type
            if cc == "msvc":
                # Triggers setuptools' MSVC auto-detection (vswhere).
                self.compiler.initialize()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "No usable C++ compiler found.\n"
                "  Windows : install 'Visual Studio Build Tools' with the\n"
                "            'Desktop development with C++' workload, then\n"
                "            re-run:  python setup_ext.py build_ext --inplace\n"
                "  Linux   : install g++ >= 9   (apt install build-essential)\n"
                "Note: the prebuilt chess_ext*.pyd (if present) keeps working\n"
                "without a compiler; a rebuild is only needed after C++ edits.\n"
                f"Underlying error: {exc!r}"
            ) from exc

        ct = self.compiler.compiler_type

        # AddressSanitizer instrumentation (audit S1): CHESS_AI_ASAN=1
        # builds a diagnostic .pyd that traps use-after-free / heap
        # overflow in the extension's own allocations (tree nodes, boards,
        # move lists) at the faulting instruction instead of corrupting the
        # heap and crashing minutes later in an unrelated malloc. Debug-only:
        # ~2-5x slower, asserts active, needs clang_rt.asan_dynamic-x86_64
        # on PATH (ships with MSVC; audit/scripts/stress_s1.py sets it up).
        asan = os.environ.get("CHESS_AI_ASAN", "") == "1"

        if ct == "msvc":
            # MSVC (Windows)
            for ext in self.extensions:
                if asan:
                    ext.extra_compile_args = [
                        "/Od",        # No optimisation — precise ASan stacks
                        "/Zi",        # Debug info (symbolised reports)
                        "/std:c++17",
                        "/arch:AVX2",
                        "/EHsc",
                        "/bigobj",
                        "/fsanitize=address",
                        # NDEBUG deliberately NOT defined: asserts active.
                    ]
                    ext.extra_link_args = ["/DEBUG", "/INCREMENTAL:NO"]
                else:
                    ext.extra_compile_args = [
                        "/O2",        # Optimise
                        "/std:c++17",
                        "/arch:AVX2", # Enables POPCNT + fast bit ops on Ryzen 7 8700F
                        "/EHsc",
                        "/DNDEBUG",
                        "/bigobj",    # Large object files (magic tables)
                    ]
        else:
            # GCC / Clang
            for ext in self.extensions:
                if asan:
                    ext.extra_compile_args = [
                        "-O1", "-g", "-std=c++17", "-march=native",
                        "-fno-omit-frame-pointer", "-fsanitize=address",
                    ]
                    ext.extra_link_args = ["-fsanitize=address"]
                else:
                    ext.extra_compile_args = [
                        "-O3",
                        "-std=c++17",
                        "-march=native",  # Use all available CPU features (AVX2, POPCNT)
                        "-fvisibility=hidden",
                        "-DNDEBUG",
                    ]
                if sys.platform == "darwin":
                    ext.extra_compile_args += ["-stdlib=libc++"]
                    ext.extra_link_args = (ext.extra_link_args or []) + ["-stdlib=libc++"]

        super().build_extensions()


# ── Extension definition ─────────────────────────────────────────────────

engine_dir = Path(__file__).parent / "engine"

chess_ext = Extension(
    name="chess_ext",
    sources=[
        str(engine_dir / "chess_core.cpp"),
        str(engine_dir / "mcts_core.cpp"),
        str(engine_dir / "bindings.cpp"),
    ],
    include_dirs=[
        str(engine_dir),
        pybind11_include(),
    ],
    language="c++",
)


# ── Setup ─────────────────────────────────────────────────────────────────

setup(
    name="chess_ext",
    version="1.0.0",
    ext_modules=[chess_ext],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)

