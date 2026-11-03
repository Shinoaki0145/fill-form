import ast
from contextlib import chdir
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fill_form.config import Settings

NOTEBOOK = Path(__file__).resolve().parents[1] / "Run_Fill_Form_Colab.ipynb"


class LaunchTests(unittest.TestCase):
    def test_notebook_compiles_and_config_needs_no_keys_or_login(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<cell>", "exec")
                self.assertEqual(cell["outputs"], [])
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if "SERVER_ENV = " in "".join(c["source"]))
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            scope = dict(Path=Path, os=os, sys=sys, subprocess=subprocess,
                         PROJECT=Path(folder), FAKE=False, SHARE=True)
            with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
                exec(compile(source, "<config>", "exec"), scope)
            self.assertEqual(scope["SERVER_ENV"]["SEGMENT_SILENCE_SECONDS"], "0.7")

    def test_clone_and_reuse_standalone_and_nested_repositories(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "REPO_DIR = " in "".join(c["source"]))
        for layout in ("", "fill-form"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                work = root / "notebook"
                work.mkdir()
                checkout = root / "repo"
                tree = ast.parse(source)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REPO_DIR":
                        node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(checkout))], keywords=[])
                ast.fix_missing_locations(tree)
                def clone(command, check):
                    self.assertEqual(command[:2], ["git", "clone"])
                    project = Path(command[-1]) / layout
                    (project / "fill_form").mkdir(parents=True)
                    (project / "app.py").touch()
                    (project / "fill_form/backend.py").touch()
                with chdir(work), patch("subprocess.run", side_effect=clone) as run:
                    scope = {}
                    exec(compile(tree, "<bootstrap>", "exec"), scope)
                    self.assertEqual(scope["PROJECT"], checkout / layout)
                    run.assert_called_once()
                    exec(compile(tree, "<bootstrap>", "exec"), {})
                    run.assert_called_once()

    def test_launch_without_auth(self):
        import app
        ui = Mock()
        with patch.object(app.Settings, "load", return_value=Settings(fake=True, share=True)), \
             patch.object(app, "Controller"), patch.object(app, "build_ui", return_value=ui), \
             patch.object(app.atexit, "register"):
            app.main()
        self.assertIsNone(ui.queue.return_value.launch.call_args.kwargs["auth"])

    def test_invalid_settings(self):
        for key, value in [("WEB_PORT", "0"), ("SEGMENT_SILENCE_SECONDS", "nan")]:
            with self.subTest(key=key), patch.dict(os.environ, {"ASR_ENV_PATH": "/nonexistent", key: value}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.load()
