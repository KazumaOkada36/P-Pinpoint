import unittest

from pinpoint_mcp.data.loader import load_raw
from pinpoint_mcp.engine.ranker import rank_locations
from pinpoint_mcp.server import (
    analyze_county,
    compare_locations,
    get_industry_trends,
    list_supported_regions,
    recommend_locations,
)


class LoaderTests(unittest.TestCase):
    def test_load_raw_handles_non_utf8_files(self):
        raw = load_raw()
        self.assertIn("35001", raw)
        self.assertEqual(raw["35001"]["state_abbr"], "NM")


class RecommendationTests(unittest.TestCase):
    def test_rank_locations_returns_results(self):
        result = rank_locations("cafe", "California", ["foot traffic"], 3)
        self.assertEqual(result["business_type"], "cafe")
        self.assertEqual(result["region_label"], "California")
        self.assertEqual(len(result["results"]), 3)
        self.assertGreater(result["results"][0]["score"], 0)

    def test_recommend_locations_normalizes_priorities_and_top_n(self):
        result = recommend_locations("tech startup", "Bay Area", ["growth", "growth", "unknown"], "999")
        self.assertEqual(result["priorities"], ["growth potential"])
        self.assertLessEqual(len(result["results"]), 50)


class CountyAnalysisTests(unittest.TestCase):
    def test_analyze_county_returns_industry_descriptions(self):
        result = analyze_county("Santa Clara", "CA", "tech", ["growth potential"])
        self.assertEqual(result["county"], "Santa Clara")
        first_industry = next(iter(result["industries"].values()))
        self.assertIn("description", first_industry)

    def test_analyze_county_reports_ambiguous_prefix(self):
        result = analyze_county("Santa", "CA", "tech")
        self.assertIn("ambiguous", result["error"].lower())
        self.assertTrue(result["suggestions"])


class TrendAndCompareTests(unittest.TestCase):
    def test_get_industry_trends_returns_county_series(self):
        result = get_industry_trends("information", "CA", "Santa Clara")
        self.assertEqual(result["linecode"], 45)
        self.assertEqual(len(result["years"]), len(result["values"]))
        self.assertEqual(result["geoname"], "Santa Clara, CA")

    def test_compare_locations_uses_cleaned_priorities(self):
        result = compare_locations(
            [{"county": "Santa Clara", "state": "CA"}, {"county": "Los Angeles", "state": "CA"}],
            "cafe",
            ["foot traffic", "foot traffic", "unknown"],
        )
        self.assertEqual(result["priorities"], ["foot traffic"])
        self.assertEqual(len(result["comparison"]), 2)
        self.assertIn("score", result["comparison"][0])

    def test_compare_locations_surfaces_ambiguity(self):
        result = compare_locations([{"county": "Santa", "state": "CA"}], "cafe")
        self.assertIn("ambiguous", result["comparison"][0]["error"].lower())


class ListingTests(unittest.TestCase):
    def test_supported_regions_contains_expected_keys(self):
        result = list_supported_regions()
        self.assertIn("states", result)
        self.assertIn("business_types", result)
        self.assertIn("priority_options", result)
        self.assertGreater(result["total_counties_in_dataset"], 0)


if __name__ == "__main__":
    unittest.main()
