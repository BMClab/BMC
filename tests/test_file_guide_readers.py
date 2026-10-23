"""Regressions for independently reviewed motion-file tutorial readers."""

import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import c3d
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    if str(ROOT / "functions") not in sys.path:
        sys.path.insert(0, str(ROOT / "functions"))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module


class FileGuideReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c3d_readers = [
            load_module("review_c3d", "functions/c3d_data.py"),
            load_module("review_open_c3d", "notebooks/open_c3d_file.py"),
            load_module("review_read_c3d", "notebooks/read_c3d.py"),
        ]
        cls.trc_readers = [
            (load_module("review_trc", "functions/read_trc.py"), "fmt"),
            (load_module("review_trc_notebook", "notebooks/read_trc.py"), "fmt"),
            (load_module("review_io", "functions/io_cortexmac.py"), "df_multi"),
            (
                load_module(
                    "review_mac_notebook", "notebooks/read_mac_ascii_files.py"
                ),
                "df_multi",
            ),
        ]
        cls.mac_readers = [
            module for module, option in cls.trc_readers if option == "df_multi"
        ]

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)

    def make_c3d(self, labels, analog_labels=("EMG",), analog_count=1):
        writer = c3d.Writer(point_rate=100, analog_rate=200 if analog_count else 0)
        writer.set_point_labels(labels)
        if analog_labels is not None:
            writer.set_analog_labels(analog_labels)
        points = np.tile([0.0, 2.0, 3.0, 0.0, 1.0], (2, 1))
        analog = (
            np.arange(analog_count * 2, dtype=float).reshape(analog_count, 2)
            if analog_count
            else np.empty((0, 0))
        )
        frames = np.empty((2, 2), dtype=object)
        for index in range(2):
            frames[index, 0] = points
            frames[index, 1] = analog
        writer.add_frames(frames)
        path = self.path / "test.c3d"
        with path.open("wb") as stream:
            writer.write(stream)
        return path

    def test_c3d_point_label_count_does_not_determine_point_count(self):
        for labels in (["A"], ["A", "B", "Unused"]):
            path = self.make_c3d(labels)
            for module in self.c3d_readers:
                with self.subTest(labels=labels, module=module.__name__):
                    _, points, _ = module.load_c3d(path)
                    self.assertEqual(points.shape, (2, 6))
                    self.assertEqual(points.columns[0], ("A", "X"))
                    self.assertEqual(points.iloc[0, 0], 0)  # valid zero, not missing
                    self.assertTrue(points.columns.is_unique)

    def test_c3d_analog_labels_match_used_channels_and_may_be_absent(self):
        for labels, count in [(None, 0), (None, 1), (("A",), 2), (("A", "Unused"), 1)]:
            path = self.make_c3d(["A", "B"], labels, count)
            for module in self.c3d_readers:
                with self.subTest(labels=labels, count=count, module=module.__name__):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, _, analog = module.load_c3d(path)
                    self.assertEqual(analog.shape[1], count)
                    self.assertEqual(len(analog), 4 if count else 0)
                    self.assertTrue(analog.columns.is_unique)

    def test_c3d_generated_suffix_does_not_collide_with_existing_label(self):
        path = self.make_c3d(["A", "A_2"], ("A", "A", "A_2"), 3)
        for module in self.c3d_readers:
            with self.subTest(module=module.__name__):
                _, _, analog = module.load_c3d(path)
                self.assertTrue(analog.columns.is_unique)
                self.assertIn("A_2", analog.columns)

    def make_trc(self):
        path = self.path / "partial.trc"
        rows = [
            ["PathFileType", "4", "(X/Y/Z)", "partial.trc"],
            [
                "DataRate",
                "CameraRate",
                "NumFrames",
                "NumMarkers",
                "Units",
                "OrigDataRate",
                "OrigDataStartFrame",
                "OrigNumFrames",
            ],
            [100, 100, 2, 3, "mm", 100, 1, 2],
            ["Frame#", "Time", "A", "", "", "B", "", "", "Empty", "", ""],
            ["", "", "X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3"],
            [],
            [1, 0, 1, "", 3, 4, 5, 6, "", "", ""],
            [2, 0.01, 2, "", 4, 5, 6, 7, "", "", ""],
        ]
        path.write_text("\n".join("\t".join(map(str, row)) for row in rows) + "\n")
        return path

    def test_dropna_keeps_partial_marker_triplet_and_drops_whole_empty_marker(self):
        path = self.make_trc()
        for module, option in self.trc_readers:
            for multi in (False, True):
                kwargs = {
                    option: ("multi" if multi else "uni") if option == "fmt" else multi
                }
                with self.subTest(module=module.__name__, multi=multi):
                    header, data = module.read_trc(
                        path, dropna=True, show_msg=False, **kwargs
                    )
                    self.assertEqual(header["nmarkers"], 2)
                    self.assertEqual(header["markers"], ["A", "B"])
                    values = data.to_numpy() if multi else data.iloc[:, 2:].to_numpy()
                    self.assertEqual(values.shape, (2, 6))
                    self.assertTrue(np.isnan(values[:, 1]).all())
                    np.testing.assert_array_equal(values[:, 3:], [[4, 5, 6], [5, 6, 7]])

    def test_dropna_xarray_keeps_partial_marker_xyz(self):
        for module, option in self.trc_readers:
            if option != "fmt":
                continue
            with self.subTest(module=module.__name__):
                data = module.read_trc(
                    self.make_trc(), dropna=True, fmt="xarray", show_msg=False
                )
                self.assertEqual(data.shape, (2, 2, 3))
                self.assertTrue(np.isnan(data.sel(marker="A", component="Y")).all())

    def make_forces(self):
        path = self.path / "plates.forces"
        labels = [
            f"{axis}{plate}"
            for plate in (1, 10)
            for axis in ("FX", "FY", "FZ", "X", "Y", "Z", "MZ")
        ]
        path.write_text(
            "[Force Data]\nNumberOfForcePlates=2\nSampleRate=100\nNumberOfSamples=2\n#Sample\t"
            + "\t".join(labels)
            + "\n1\t"
            + "\t".join(map(str, range(14)))
            + "\n2\t"
            + "\t".join(map(str, range(14)))
            + "\n"
        )
        return path

    def test_force_selection_matches_complete_plate_number_and_scales_integer_columns(
        self,
    ):
        for module in self.mac_readers:
            with self.subTest(module=module.__name__):
                header, data = module.read_forces(
                    self.make_forces(), forcepla=[1], show_msg=False
                )
                self.assertEqual(header["NumberOfForcePlates"], 1)
                self.assertEqual(data.shape, (2, 7))
                self.assertEqual(
                    data.columns.tolist(),
                    [f"{axis}1" for axis in ("FX", "FY", "FZ", "X", "Y", "Z", "MZ")],
                )
                self.assertAlmostEqual(data.iloc[0]["X1"], 0.003)
                self.assertAlmostEqual(data.iloc[0]["MZ1"], 0.006)
                with self.assertRaises(ValueError):
                    module.read_forces(
                        self.make_forces(), forcepla=[99], show_msg=False
                    )

    def test_calibration_uses_full_plate_number(self):
        path = self.path / "plate10.cal"
        path.write_text(
            "10\n1 10 10\n"
            + "\n".join(" ".join(map(str, row)) for row in np.eye(6))
            + "\n0 0 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        )
        for module in self.mac_readers:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.read_cal(path, show_msg=False)["fp"], [10])

    def trc_table(self, times):
        return pd.DataFrame(
            {
                "Frame#": np.arange(1, len(times) + 1),
                "Time": times,
                "Ax": 2 * times,
                "Ay": 3 * times,
                "Az": 4 * times,
            }
        )

    def force_table(self, times):
        return pd.DataFrame(
            {f"{axis}1": 5 * times for axis in ("FX", "FY", "FZ", "X", "Y", "Z", "MZ")},
            index=pd.Index(times, name="Time"),
        )

    def read_v3d(self, path):
        return pd.read_csv(path, sep="\t", skiprows=5, header=None)

    def test_visual3d_equal_rate_tables_align_by_time_not_dataframe_index(self):
        times = np.array([1.0, 1.01, 1.02])
        for module in self.mac_readers:
            with self.subTest(module=module.__name__):
                path = self.path / "aligned.txt"
                module.write_v3dtxt(
                    path, self.trc_table(times), self.force_table(times), show_msg=False
                )
                data = self.read_v3d(path)
                self.assertEqual(data.shape, (3, 12))
                np.testing.assert_allclose(data.iloc[:, 1], times)
                np.testing.assert_allclose(data.iloc[:, 2], 2 * times)
                np.testing.assert_allclose(data.iloc[:, 5], 5 * times)
                self.assertFalse(data.isna().any().any())

    def test_visual3d_different_rates_interpolate_on_measured_common_times(self):
        trc_times = np.array([1.0, 1.01, 1.02])
        force_times = 1 + np.arange(6) / 200
        expected = 1 + np.arange(5) / 200
        for module in self.mac_readers:
            with self.subTest(module=module.__name__):
                path = self.path / "resampled.txt"
                module.write_v3dtxt(
                    path,
                    self.trc_table(trc_times),
                    self.force_table(force_times),
                    show_msg=False,
                )
                data = self.read_v3d(path)
                np.testing.assert_allclose(data.iloc[:, 1], expected)
                np.testing.assert_allclose(data.iloc[:, 2], 2 * expected)
                np.testing.assert_allclose(data.iloc[:, 5], 5 * expected)

    def test_visual3d_rejects_offset_or_different_duration(self):
        times = np.arange(4) / 100
        for module in self.mac_readers:
            for force_times in (times + 0.1, np.arange(10) / 100):
                with (
                    self.subTest(module=module.__name__, times=force_times),
                    self.assertRaisesRegex(ValueError, "time|interval"),
                ):
                    module.write_v3dtxt(
                        self.path / "bad.txt",
                        self.trc_table(times),
                        self.force_table(force_times),
                        show_msg=False,
                    )

    def test_visual3d_trc_filename_is_converted_to_meters(self):
        for module in self.mac_readers:
            with self.subTest(module=module.__name__):
                path = self.path / "markers-only.txt"
                module.write_v3dtxt(path, str(self.make_trc()), "", show_msg=False)
                self.assertAlmostEqual(self.read_v3d(path).iloc[0, 2], 0.001)


if __name__ == "__main__":
    unittest.main()
