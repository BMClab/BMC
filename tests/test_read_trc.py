import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "functions" / "read_trc.py"
TRC_PATH = ROOT / "data" / "walk.trc"


def load_read_trc():
    spec = importlib.util.spec_from_file_location("read_trc", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.read_trc


class ReadTrcTests(unittest.TestCase):
    def setUp(self):
        self.read_trc = load_read_trc()

    def test_multi_format_labels_columns_with_marker_names(self):
        h, df = self.read_trc(TRC_PATH, fmt="multi", show_msg=False)

        expected = [m.replace(".", "_") for m in h["markers"] for _ in range(3)]
        self.assertEqual(list(df.columns.get_level_values("Marker")), expected)
        self.assertEqual(df.columns.get_level_values("Marker")[0], "R_ASIS")
        self.assertEqual(
            list(df.columns.get_level_values("Coordinate")[:3]), ["X", "Y", "Z"]
        )
        self.assertEqual(df.index.name, "Time")

    def test_multi_and_uni_formats_hold_the_same_data(self):
        _, df_uni = self.read_trc(TRC_PATH, fmt="uni", show_msg=False)
        _, df_multi = self.read_trc(TRC_PATH, fmt="multi", show_msg=False)

        np.testing.assert_array_equal(
            df_multi.to_numpy(), df_uni.iloc[:, 2:].to_numpy()
        )
        np.testing.assert_array_equal(
            df_multi.index.to_numpy(), df_uni["Time"].to_numpy()
        )

    def test_saved_file_reads_back_with_the_same_data(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "walk_copy.trc"
            h, df = self.read_trc(
                TRC_PATH, fname2=str(saved), fmt="uni", show_msg=False
            )
            h2, df2 = self.read_trc(saved, fmt="uni", show_msg=False)

        self.assertEqual(h2["markers"], h["markers"])
        np.testing.assert_allclose(df2.to_numpy(), df.to_numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
