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

    def grf_dataframe(self):
        return pd.DataFrame(
            {"R_ground_force_vy": [700.0, 710.5], "R_ground_force_px": [0.1, 0.12]},
            index=pd.Index([0.0, 0.01], name="time"),
        )

    def test_mot_file_has_opensim_header(self):
        saved = self.dir / "grf.mot"
        self.io.write_mot(str(saved), self.grf_dataframe(), show_msg=False)

        lines = saved.read_text().splitlines()
        self.assertEqual(
            lines[:6],
            [
                "grf.mot",
                "version=1",
                "nRows=2",
                "nColumns=3",
                "inDegrees=no",
                "endheader",
            ],
        )
        self.assertEqual(
            lines[6].split("\t"), ["time", "R_ground_force_vy", "R_ground_force_px"]
        )

    def test_mot_file_records_angles_in_degrees(self):
        saved = self.dir / "angles.mot"
        self.io.write_mot(
            str(saved), self.grf_dataframe(), in_degrees=True, show_msg=False
        )

        self.assertEqual(saved.read_text().splitlines()[4], "inDegrees=yes")

    def test_mot_file_reads_back_with_the_same_data(self):
        df = self.grf_dataframe()
        saved = self.dir / "grf.mot"
        self.io.write_mot(str(saved), df, show_msg=False)
        h, df2 = self.io.read_mot(saved, show_msg=False)

        self.assertEqual(
            h,
            {
                "name": "grf.mot",
                "version": 1,
                "nRows": 2,
                "nColumns": 3,
                "inDegrees": False,
            },
        )
        self.assertEqual(list(df2.columns), list(df.columns))
        self.assertEqual(df2.index.name, "time")
        np.testing.assert_allclose(df2.to_numpy(), df.to_numpy())
        np.testing.assert_allclose(df2.index.to_numpy(), df.index.to_numpy())

    def test_read_mot_reads_an_opensim_file(self):
        h, df = self.io.read_mot(ROOT / "data" / "vas_int_r.mot", show_msg=False)

        self.assertEqual(
            h,
            {
                "name": "UNKNOWN",
                "version": 1,
                "nRows": 100,
                "nColumns": 3,
                "inDegrees": False,
            },
        )
        self.assertEqual(list(df.columns), ["knee_angle_r", "vas_int_r"])
        self.assertEqual(df.shape, (100, 2))
        self.assertAlmostEqual(df.iloc[0, 0], -119.99999986)

    def test_read_mot_reads_a_legacy_simm_header(self):
        legacy = self.dir / "legacy.mot"
        legacy.write_text(
            "name legacy.mot\ndatacolumns 3\ndatarows 2\nrange 0 0.01\nendheader\n"
            "time\ta\tb\n0.00\t1.0\t2.0\n0.01\t3.0\t4.0\n"
        )
        h, df = self.io.read_mot(legacy, show_msg=False)

        self.assertEqual(
            h,
            {
                "name": "legacy.mot",
                "datacolumns": 3,
                "datarows": 2,
                "range": [0.0, 0.01],
            },
        )
        np.testing.assert_allclose(df.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()


class ForceMomentTests(unittest.TestCase):
    def test_moment_uses_point_minus_origin_cross_force(self):
        io = load_io_cortexmac()
        data = np.array([[5., 6., 7., 2., 3., 4., 8.]])
        actual = io.grf_moments(data, [1., 1., 1.], show_msg=False)
        np.testing.assert_allclose(actual, [[5., 6., 7., -4., 8., 4.]])


class DelsysReaderTests(unittest.TestCase):
    def test_current_pandas_reads_separate_rates_and_scales_emg_to_millivolts(self):
        source = ROOT / 'functions' / 'delsys_demo.py'
        demo_spec = importlib.util.spec_from_file_location('delsys_demo', source)
        demo = importlib.util.module_from_spec(demo_spec)
        demo_spec.loader.exec_module(demo)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.add'
            path.write_text(demo.delsys_demo_text())
            emg, imu = load_io_cortexmac().read_delsys(path, fname2='', emg=True, imu=True, resample=[0, 0], show_msg=False)
        self.assertEqual(emg.shape, (1000, 1))
        self.assertEqual(imu.shape, (100, 9))
        self.assertAlmostEqual(emg.index[1], 0.001, places=7)
        self.assertAlmostEqual(imu.index[1], 0.01, places=7)
        self.assertLessEqual(emg.iloc[:, 0].abs().max(), 1.001)
        self.assertGreater(emg.iloc[:, 0].abs().max(), 0.95)


class DelsysOffsetTests(unittest.TestCase):
    def test_asynchronous_native_start_times_remain_distinct(self):
        demo_spec = importlib.util.spec_from_file_location('delsys_offset_demo', ROOT / 'functions' / 'delsys_demo.py')
        demo = importlib.util.module_from_spec(demo_spec)
        demo_spec.loader.exec_module(demo)
        lines = demo.delsys_demo_text().splitlines()
        for i in range(22, 1022):
            fields = lines[i].split(',')
            fields[2] = str(float(fields[2]) + 150)
            lines[i] = ','.join(fields)
        for i in range(1024, 1124):
            fields = lines[i].split(',')
            fields[2] = str(float(fields[2]) + 300)
            lines[i] = ','.join(fields)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'offset.add'
            path.write_text('\n'.join(lines) + '\n')
            emg, imu = load_io_cortexmac().read_delsys(path, fname2='', emg=True, imu=True, resample=[0, 0], show_msg=False)
        self.assertAlmostEqual(emg.index[0], 1)
        self.assertAlmostEqual(imu.index[0], 2)
