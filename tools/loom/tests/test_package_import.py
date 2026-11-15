"""The package defers its viewer and animation-driver modules, so `python -m loom.viewer` and
`python -m loom.anim` -- the native viewer's live channels -- run without runpy's double-import
RuntimeWarning on the user's console."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-W", "error::RuntimeWarning"] + args, cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def test_package_import_leaves_viewer_and_anim_unloaded():
    p = _run(["-c", "import sys, loom; print('loom.viewer' in sys.modules, 'loom.anim' in sys.modules)"])
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["False", "False"]


def test_lazy_names_resolve_to_their_modules():
    p = _run(["-c", "from loom import ViewerModel, serve_live, load_build; "
                    "print(ViewerModel.__module__, serve_live.__module__, load_build.__module__)"])
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["loom.viewer", "loom.anim", "loom.viewer"]


def test_dash_m_launches_without_the_runpy_warning():
    for mod in ("loom.viewer", "loom.anim"):
        p = _run(["-m", mod, "--help"])
        assert p.returncode == 0, (mod, p.stderr)
        assert "RuntimeWarning" not in p.stderr, (mod, p.stderr)
