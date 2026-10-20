import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "functions" / "io_cortexmac.py"


def load_io_cortexmac():
    # io_cortexmac imports sibling modules (critic_damp, ...) from functions/
    if str(MODULE_PATH.parent) not in sys.path:
        sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("io_cortexmac", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module


class IoCortexMacWriterTests(unittest.TestCase):
    def setUp(self):
        self.io = load_io_cortexmac()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_trc_file_reads_back_with_the_same_data(self):
        saved = self.dir / "walk1_copy.trc"
        h, df = self.io.read_trc(
            ROOT / "data" / "walk1.trc", fname2=str(saved), show_msg=False
        )
        h2, df2 = self.io.read_trc(saved, show_msg=False)

        np.testing.assert_allclose(df2.to_numpy(), df.to_numpy(), atol=1e-6)
        data_line = saved.read_text().splitlines(keepends=True)[6]
        self.assertTrue(data_line.endswith("\t\n"))

    def test_forces_file_reads_back_with_the_same_data(self):
        h, df = self.io.read_forces(
            ROOT / "data" / "walk1.forces", mm2m=False, show_msg=False
        )
        saved = self.dir / "walk1_copy.forces"
        self.io.write_forces(str(saved), h, df.copy(), show_msg=False)
        h2, df2 = self.io.read_forces(saved, mm2m=False, show_msg=False)

        self.assertEqual(list(df2.columns), list(df.columns))
        np.testing.assert_allclose(df2.to_numpy(), df.to_numpy(), atol=1e-6)

    def test_mot_file_has_header_and_data(self):
        df = pd.DataFrame(
            {"R_ground_force_vy": [700.0, 710.5], "R_ground_force_px": [0.1, 0.12]},
            index=pd.Index([0.0, 0.01], name="time"),
        )
        saved = self.dir / "grf.mot"
        self.io.write_mot(str(saved), df, show_msg=False)

        lines = saved.read_text().splitlines()
        self.assertEqual(lines[1:4], ["datacolumns 3", "datarows 2", "range 0.0 0.01"])
        self.assertEqual(lines[4], "endheader")
        data = pd.read_csv(saved, sep="\t", skiprows=6, index_col=0)
        np.testing.assert_allclose(data.to_numpy(), df.to_numpy())
        self.assertEqual(list(data.columns), list(df.columns))


if __name__ == "__main__":
    unittest.main()
