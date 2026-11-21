"""P0 baselines for the existing research helpers; no downloads or file writes."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.data.download import merge_prices, write_parquet_batches
from src.data.universe_process import build_membership


class ResearchBaselineTests(unittest.TestCase):
    def test_download_cli_runs_all_in_order_or_only_requested_step(self):
        from src.data import download

        steps = []
        with patch.object(download, "download_calendar", lambda: steps.append("calendar")), \
             patch("src.data.universe_process.main", lambda: steps.append("membership")), \
             patch.object(download, "download_prices", lambda: steps.append("prices")), \
             patch.object(download, "download_industry", lambda: steps.append("industry")):
            download.main([])
            self.assertEqual(steps, ["calendar", "membership", "prices", "industry"])
            for name in steps.copy():
                steps.clear()
                download.main([name])
                self.assertEqual(steps, [name])

    def test_interrupted_batch_write_preserves_existing_parquet(self):
        frame = pd.DataFrame({"score": [1.0]})

        def interrupted():
            yield frame.assign(score=2.0)
            raise ValueError("batch failed")

        with TemporaryDirectory() as temp:
            path = Path(temp) / "scores.parquet"
            write_parquet_batches([frame], path)
            with self.assertRaisesRegex(ValueError, "batch failed"):
                write_parquet_batches(interrupted(), path)
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_frequency_names_and_latest_run_keep_trading_periods_separate(self):
        from src.config import rebalance_frequency
        from src.scripts.run_all import latest_run, new_run

        for days, suffix in ((1, "daily"), (5, "weekly"), (10, "10d"), (20, "20d")):
            self.assertEqual(rebalance_frequency(days)[0], suffix)
            self.assertTrue(new_run(days).name.startswith(f"{suffix}_"))
        with TemporaryDirectory() as temp, patch("src.scripts.run_all.RUNS", Path(temp)):
            runs = [
                ("daily_20260101_100000_000000", 1),
                ("weekly_20260102_100000_000000", 5),
                ("20260103_100000_000000", 5),
            ]
            for name, days in runs:
                path = Path(temp) / name
                path.mkdir()
                (path / "config.json").write_text(json.dumps({"rebalance_days": days}))
                (path / "manifest.json").write_text("{}")
            self.assertEqual(latest_run("manifest.json", rebalance_days=1).name, runs[0][0])
            self.assertEqual(latest_run("manifest.json", rebalance_days=5).name, runs[2][0])
            with self.assertRaises(FileNotFoundError):
                latest_run("manifest.json", rebalance_days=10)

    def test_run_all_prepares_missing_inputs_once_in_dependency_order(self):
        from src.data.research_panel import prepare_inputs

        dates = pd.bdate_range("2013-09-02", periods=3)
        config = {"segments": {"test": ["2013-09-02", "2013-09-04"]}}
        with TemporaryDirectory() as temp:
            paths = {name: Path(temp) / f"{name}.parquet" for name in ("calendar", "membership", "prices", "industries")}
            steps = []

            def calendar():
                steps.append("calendar")
                pd.DataFrame({"cal_date": dates.strftime("%Y%m%d")}).to_parquet(paths["calendar"])

            def members():
                steps.append("membership")
                pd.DataFrame(dict(trade_date=dates, ticker="A")).to_parquet(paths["membership"])

            def prices():
                steps.append("prices")
                pd.DataFrame(dict(trade_date=dates, ticker="A", adj_factor=1.)).to_parquet(paths["prices"])

            def industry():
                steps.append("industry")
                paths["industries"].touch()

            with patch("src.data.research_panel.CSI500_INPUTS", paths), patch(
                "src.data.download.download_calendar", side_effect=calendar
            ), patch("src.data.universe_process.main", side_effect=members), patch(
                "src.data.download.download_prices", side_effect=prices
            ), patch(
                "src.data.download.download_industry", side_effect=industry
            ):
                prepare_inputs(config, need_industry=False)
                self.assertEqual(steps, ["calendar", "membership", "prices"])
                pd.testing.assert_index_equal(prepare_inputs(config), dates, check_names=False)
                pd.testing.assert_index_equal(prepare_inputs(config), dates, check_names=False)
            self.assertEqual(steps, ["calendar", "membership", "prices", "industry"])

    def test_early_membership_uses_the_last_known_snapshot(self):
        snapshots = pd.DataFrame(dict(
            con_code=["A", "B"], trade_date=["20130830", "20130903"], weight=[100., 100.],
            index_code="000905.SH",
        ))
        calendar = pd.DataFrame(dict(cal_date=["20130902", "20130903", "20130904"]))
        result = build_membership(snapshots, calendar)
        self.assertEqual(result.ticker.to_list(), ["A", "B", "B"])
        self.assertTrue(result.source_snapshot_date.le(result.trade_date).all())
        with self.assertRaises(ValueError):
            build_membership(snapshots.iloc[1:], calendar)

    def test_price_download_matches_adjustments_by_stock_and_date(self):
        daily = pd.DataFrame(dict(ts_code=["A", "A"], trade_date=["20140102", "20140103"], close=[10., 11.]))
        factors = pd.DataFrame(dict(ts_code=["A", "A"], trade_date=["20140103", "20140102"], adj_factor=[2., 1.]))
        result = merge_prices(daily, factors)
        self.assertEqual(result.adj_factor.to_list(), [1., 2.])
        with self.assertRaises(ValueError):
            merge_prices(daily, factors.iloc[:1])

    def test_industry_download_only_fills_confirmed_gaps(self):
        from src.data.download import supplement_history
        from src.factors.neutralization import align_industry_history

        keys = pd.DataFrame(dict(ticker="A", trade_date=pd.date_range("2023-01-01", periods=9)))
        primary = pd.DataFrame([
            ("old.SI", "20230101", "20230102"), ("new.SI", "20230107", None),
            ("old.SI", "20230108", "20230109"),
        ], columns=["l1_code", "in_date", "out_date"]).assign(ts_code="A", l1_name=lambda df: df.l1_code)
        secondary = pd.DataFrame([
            ("old", "2023-01-02", "2023-01-06"), ("new", "2023-01-05", "2023-01-06"),
            ("new", "1900-01-01", None), ("unrelated_index", "2023-01-01", None),
        ], columns=["index_code", "in_date", "out_date"])
        result, audit = supplement_history(keys, primary, lambda ticker: secondary)
        actual = align_industry_history(keys, result)
        self.assertEqual(actual.fillna("-").tolist(),
                         ["old.SI", "old.SI", "old.SI", "old.SI", "-", "-", "new.SI", "-", "new.SI"])
        self.assertEqual(audit.status.value_counts().to_dict(), {
            "已补一级行业": 3, "次源行业冲突": 1, "主源行业冲突": 1, "无历史一级行业记录": 1})
        self.assertEqual(len(result), len(primary) + 1)

if __name__ == "__main__":
    unittest.main()
