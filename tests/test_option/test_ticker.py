import json
import shutil
import unittest
from os.path import exists
from pathlib import Path

from pytse_client import option

class TestOption(unittest.TestCase):
    """
    Test cases for the `option` function in pytse_client.download.

    The `option` function downloads historical data for ticker symbols specified
    in `pytse_client/data/symbols_option.json`. It supports:
    - Generating DataFrames with columns: date, open, high, low, close, adjClose,
      yesterday, value, volume, count (and jdate if include_jdate=True).
    - Saving data as CSV or Excel files with `output_format` ('csv', 'excel', or None).
    - Returning only DataFrames without saving with `only_dataframe=True`.
    - Adjusting prices for capital increase/dividends with `adjust=True`.
    - Adding Persian (Jalali) date column with `include_jdate=True`.
    - Parallel downloading using ThreadPoolExecutor (max 10 workers).
    - Warning if download is incomplete: "Warning, download did not complete, re-run the code".

    Tests verify:
    - Correct DataFrame generation and column presence.
    - Proper CSV and Excel file saving.
    - Behavior with `only_dataframe=True`.
    - Error handling for missing JSON file or invalid output_format.
    """
    
    def setUp(self) -> None:
        """
        Set up test environment by creating a temporary JSON file and output directory.
        """
        self.output_dir = "test_option_output"
        self.json_path = "pytse_client/data/symbols_option.json"
        
        # Create temporary symbols_option.json
        self.test_symbols = [
            {"ضهرم2011": {"index": "57164323162144239", "code": "IRO9AHRM2851", "name": "اختیارخ اهرم-15000-1404/02/31"}},
            {"ضهرم2012": {"index": "32000546295435383", "code": "IRO9AHRM2861", "name": "اختیارخ اهرم-16000-1404/02/31"}}
        ]
        Path("pytse_client/data").mkdir(parents=True, exist_ok=True)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_symbols, f, ensure_ascii=False, indent=2)
        
        return super().setUp()

    def tearDown(self) -> None:
        """
        Clean up by removing temporary JSON file and output directory.
        """
        if exists(self.json_path):
            Path(self.json_path).unlink()
        if exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        return super().tearDown()

    def test_download_dataframe(self):
        """
        Test that `option` returns a valid DataFrame with correct columns.
        """
        result = option(
            output_format=None,
            only_dataframe=True,
            include_jdate=True
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result, "Result dictionary is empty")
        
        # Check for at least one symbol
        symbol = "ضهرم2011"
        self.assertIn(symbol, result, f"Symbol {symbol} not in result")
        
        df = result[symbol]
        self.assertFalse(df.empty, "DataFrame is empty")
        
        # Expected columns
        expected_columns = [
            "date", "open", "high", "low", "close", "adjClose",
            "yesterday", "value", "volume", "count", "jdate"
        ]
        missing_columns = [
            col for col in expected_columns if col not in df.columns
        ]
        self.assertEqual(
            len(missing_columns), 0, f"Missing columns: {missing_columns}"
        )
        
        # Check date column is datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(df["date"]),
            "Date column is not datetime"
        )

    def test_save_csv(self):
        """
        Test that `option` saves data as CSV files correctly.
        """
        result = option(
            output_format="csv",
            base_path=self.output_dir,
            include_jdate=True,
            adjust=True
        )
        self.assertTrue(result, "Result dictionary is empty")
        
        # Check CSV files exist
        for symbol in ["ضهرم2011", "ضهرم2012"]:
            csv_path = Path(self.output_dir) / f"{symbol}-ت.csv"
            self.assertTrue(
                exists(csv_path),
                f"CSV file not created for {symbol}"
            )
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            self.assertFalse(df.empty, f"CSV file for {symbol} is empty")
            expected_columns = [
                "date", "open", "high", "low", "close", "adjClose",
                "yesterday", "value", "volume", "count", "jdate"
            ]
            missing_columns = [
                col for col in expected_columns if col not in df.columns
            ]
            self.assertEqual(
                len(missing_columns), 0, f"Missing columns in CSV: {missing_columns}"
            )

    def test_save_excel(self):
        """
        Test that `option` saves data as Excel files correctly.
        Requires openpyxl to be installed.
        """
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed, skipping Excel test")
        
        result = option(
            output_format="excel",
            base_path=self.output_dir,
            include_jdate=True
        )
        self.assertTrue(result, "Result dictionary is empty")
        
        # Check Excel files exist
        for symbol in ["ضهرم2011", "ضهرم2012"]:
            excel_path = Path(self.output_dir) / f"{symbol}.xlsx"
            self.assertTrue(
                exists(excel_path),
                f"Excel file not created for {symbol}"
            )
            
            # Verify Excel content
            df = pd.read_excel(excel_path)
            self.assertFalse(df.empty, f"Excel file for {symbol} is empty")
            expected_columns = [
                "date", "open", "high", "low", "close", "adjClose",
                "yesterday", "value", "volume", "count", "jdate"
            ]
            missing_columns = [
                col for col in expected_columns if col not in df.columns
            ]
            self.assertEqual(
                len(missing_columns), 0, f"Missing columns in Excel: {missing_columns}"
            )

    def test_only_dataframe(self):
        """
        Test that `option` with only_dataframe=True does not save files.
        """
        result = option(
            output_format="csv",  # Should be ignored due to only_dataframe=True
            only_dataframe=True,
            base_path=self.output_dir
        )
        self.assertTrue(result, "Result dictionary is empty")
        
        # Check no files are saved
        self.assertFalse(
            exists(self.output_dir),
            "Output directory was created despite only_dataframe=True"
        )
        
        # Verify DataFrame content
        symbol = "ضهرم2012"
        self.assertIn(symbol, result, f"Symbol {symbol} not in result")
        df = result[symbol]
        self.assertFalse(df.empty, "DataFrame is empty")
        expected_columns = [
            "date", "open", "high", "low", "close", "adjClose",
            "yesterday", "value", "volume", "count"
        ]
        missing_columns = [
            col for col in expected_columns if col not in df.columns
        ]
        self.assertEqual(
            len(missing_columns), 0, f"Missing columns: {missing_columns}"
        )

    def test_missing_json_file(self):
        """
        Test that `option` raises FileNotFoundError when symbols_option.json is missing.
        """
        # Remove temporary JSON file
        Path(self.json_path).unlink()
        
        with self.assertRaises(FileNotFoundError):
            option(output_format=None, only_dataframe=True)

    def test_invalid_output_format(self):
        """
        Test that `option` raises ValueError for invalid output_format.
        """
        with self.assertRaises(ValueError):
            option(output_format="invalid", only_dataframe=False)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOption)
    unittest.TextTestRunner(verbosity=3).run(suite)