from __future__ import annotations

from nl_interface.service import _extract_cli_args


def test_extract_cli_args_parses_metric_area():
    result = _extract_cli_args("3BHK with area of 3000 sqm")

    assert result["total_area"] == 3000.0
    assert result["area_unit"] == "sq.m"


def test_extract_cli_args_converts_square_feet_to_sqm():
    result = _extract_cli_args("Need a 2BHK of 1200 sqft")

    assert round(result["total_area"], 3) == 111.484
    assert result["area_unit"] == "sq.m"



def test_extract_cli_args_parses_plot_size_with_star_separator():
    result = _extract_cli_args("design a 4bhk flat on 100m * 150m plot")

    assert result["boundary_size"] == [100.0, 150.0]
    assert result["boundary_role"] == "site"
