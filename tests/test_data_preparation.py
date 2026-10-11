import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.data_preparation import DataPreparation


class DataPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = Path(__file__).parents[1]
        cls.config = yaml.safe_load((project / "src/config.yaml").read_text())

    def test_extract_lease_info(self):
        self.assertEqual(DataPreparation._extract_lease_info("61 years 4 months"), 736)
        self.assertEqual(DataPreparation._extract_lease_info("72"), 864)
        self.assertIsNone(DataPreparation._extract_lease_info(None))

    def test_convert_storey_range(self):
        self.assertEqual(DataPreparation._convert_storey_range("10 TO 12"), 11.0)

    def test_missing_columns_raise_clear_error(self):
        prep = DataPreparation(self.config)
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prep.validate_columns(pd.DataFrame({"resale_price": [500000]}))


if __name__ == "__main__":
    unittest.main()
