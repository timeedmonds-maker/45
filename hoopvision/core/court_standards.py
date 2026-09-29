"""Court-standard geometry for HoopVision Solved Engine 1.2.

The detector still observes the same 33 semantic basketball-court landmarks.
This module changes only the canonical metric coordinates those landmarks map
to.  NBA remains byte-for-byte compatible with the pinned nbacv configuration;
FIBA uses the current full-size FIBA court dimensions/markings. NCAA_MENS
uses the current NCAA men's full-court markings.

FIBA source: Official Basketball Rules 2024, Article 2 (effective 1 Oct 2024):
28.00 x 15.00 m court; 4.90 m restricted-area width; free-throw line 5.80 m
from the endline; 6.75 m three-point arc; parallel three-point line 0.90 m
from the sideline; basket centre 1.575 m from the endline; centre-circle
radius 1.80 m; no-charge semi-circle radius 1.30 m; throw-in mark 8.325 m
from the nearest endline.

NCAA source: official 2025-26 NCAA Men's and Women's Basketball Court diagram:
94 x 50 ft; men's three-point arc 22 ft 1 3/4 in; 12 ft lane; 19 ft lane
length; 15 ft free-throw distance from the backboard; 6 ft centre-circle
radius; men's restricted-area radius 4 ft; basket centre 63 in from endline;
recommended 28 ft sideline mark. NCAA's men's rules page still links this
court diagram for the current rules resources.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CourtStandard:
    name: str
    court_width_cm: float
    court_length_cm: float
    three_point_arc_radius_cm: float
    straight_section_three_point_line_cm: float
    paint_width_cm: float
    paint_length_cm: float
    center_circle_radius_cm: float
    restricted_area_radius_cm: float
    rim_diameter_cm: float
    sideline_to_three_point_line_cm: float
    baseline_to_rim_center_cm: float
    baseline_to_throw_line_cm: float
    baseline_to_throw_line_length_cm: float

    @property
    def vertices(self) -> list[tuple[float, float]]:
        paint_start_cm = (self.court_width_cm - self.paint_width_cm) / 2.0
        return [
            (0.0, 0.0),
            (0.0, self.sideline_to_three_point_line_cm),
            (0.0, paint_start_cm),
            (0.0, paint_start_cm + self.paint_width_cm),
            (0.0, self.court_width_cm - self.sideline_to_three_point_line_cm),
            (0.0, self.court_width_cm),
            (self.baseline_to_rim_center_cm, self.court_width_cm / 2.0),
            (self.straight_section_three_point_line_cm, self.sideline_to_three_point_line_cm),
            (self.straight_section_three_point_line_cm, self.court_width_cm - self.sideline_to_three_point_line_cm),
            (self.paint_length_cm, paint_start_cm),
            (self.paint_length_cm, paint_start_cm + self.paint_width_cm / 2.0),
            (self.paint_length_cm, paint_start_cm + self.paint_width_cm),
            (self.baseline_to_throw_line_cm, 0.0),
            (self.baseline_to_rim_center_cm + self.three_point_arc_radius_cm, self.court_width_cm / 2.0),
            (self.baseline_to_throw_line_cm, self.court_width_cm),
            (self.court_length_cm / 2.0, 0.0),
            (self.court_length_cm / 2.0, self.court_width_cm / 2.0),
            (self.court_length_cm / 2.0, self.court_width_cm),
            (self.court_length_cm - self.baseline_to_throw_line_cm, 0.0),
            (self.court_length_cm - self.baseline_to_rim_center_cm - self.three_point_arc_radius_cm, self.court_width_cm / 2.0),
            (self.court_length_cm - self.baseline_to_throw_line_cm, self.court_width_cm),
            (self.court_length_cm - self.paint_length_cm, paint_start_cm),
            (self.court_length_cm - self.paint_length_cm, paint_start_cm + self.paint_width_cm / 2.0),
            (self.court_length_cm - self.paint_length_cm, paint_start_cm + self.paint_width_cm),
            (self.court_length_cm - self.straight_section_three_point_line_cm, self.sideline_to_three_point_line_cm),
            (self.court_length_cm - self.straight_section_three_point_line_cm, self.court_width_cm - self.sideline_to_three_point_line_cm),
            (self.court_length_cm - self.baseline_to_rim_center_cm, self.court_width_cm / 2.0),
            (self.court_length_cm, 0.0),
            (self.court_length_cm, self.sideline_to_three_point_line_cm),
            (self.court_length_cm, paint_start_cm),
            (self.court_length_cm, paint_start_cm + self.paint_width_cm),
            (self.court_length_cm, self.court_width_cm - self.sideline_to_three_point_line_cm),
            (self.court_length_cm, self.court_width_cm),
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "court_standard": self.name,
            "court_length_cm": self.court_length_cm,
            "court_width_cm": self.court_width_cm,
            "three_point_arc_radius_cm": self.three_point_arc_radius_cm,
            "straight_section_three_point_line_cm": self.straight_section_three_point_line_cm,
            "paint_width_cm": self.paint_width_cm,
            "paint_length_cm": self.paint_length_cm,
            "center_circle_radius_cm": self.center_circle_radius_cm,
            "restricted_area_radius_cm": self.restricted_area_radius_cm,
            "rim_diameter_cm": self.rim_diameter_cm,
            "sideline_to_three_point_line_cm": self.sideline_to_three_point_line_cm,
            "baseline_to_rim_center_cm": self.baseline_to_rim_center_cm,
            "baseline_to_throw_line_cm": self.baseline_to_throw_line_cm,
            "baseline_to_throw_line_length_cm": self.baseline_to_throw_line_length_cm,
            "landmark_count": len(self.vertices),
        }


NBA = CourtStandard(
    name="NBA",
    court_width_cm=1524.0,
    court_length_cm=2865.0,
    three_point_arc_radius_cm=724.0,
    straight_section_three_point_line_cm=424.0,
    paint_width_cm=488.0,
    paint_length_cm=579.0,
    center_circle_radius_cm=183.0,
    restricted_area_radius_cm=122.0,
    rim_diameter_cm=46.0,
    sideline_to_three_point_line_cm=91.0,
    baseline_to_rim_center_cm=160.0,
    baseline_to_throw_line_cm=835.0,
    baseline_to_throw_line_length_cm=50.0,
)

_FIBA_HALF_WIDTH = 750.0
_FIBA_CORNER_Y = _FIBA_HALF_WIDTH - 90.0
_FIBA_ARC_X_OFFSET = math.sqrt(675.0**2 - _FIBA_CORNER_Y**2)

FIBA = CourtStandard(
    name="FIBA",
    court_width_cm=1500.0,
    court_length_cm=2800.0,
    three_point_arc_radius_cm=675.0,
    straight_section_three_point_line_cm=157.5 + _FIBA_ARC_X_OFFSET,
    paint_width_cm=490.0,
    paint_length_cm=580.0,
    center_circle_radius_cm=180.0,
    restricted_area_radius_cm=130.0,
    rim_diameter_cm=45.0,
    sideline_to_three_point_line_cm=90.0,
    baseline_to_rim_center_cm=157.5,
    baseline_to_throw_line_cm=832.5,
    baseline_to_throw_line_length_cm=15.0,
)

NCAA_MENS = CourtStandard(
    name="NCAA_MENS",
    court_width_cm=1524.0,
    court_length_cm=2865.12,
    three_point_arc_radius_cm=675.005,
    straight_section_three_point_line_cm=301.625,
    paint_width_cm=365.76,
    paint_length_cm=579.12,
    center_circle_radius_cm=182.88,
    restricted_area_radius_cm=121.92,
    rim_diameter_cm=45.72,
    sideline_to_three_point_line_cm=101.9175,
    baseline_to_rim_center_cm=160.02,
    baseline_to_throw_line_cm=853.44,
    baseline_to_throw_line_length_cm=5.08,
)

COURT_STANDARDS = {"NBA": NBA, "FIBA": FIBA, "NCAA_MENS": NCAA_MENS}


def get_court_standard(name: str) -> CourtStandard:
    key = str(name).strip().upper()
    try:
        return COURT_STANDARDS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported court standard {name!r}; expected one of {sorted(COURT_STANDARDS)}") from exc


def configure_nbacv_court(court_module: Any, name: str) -> dict[str, Any]:
    """Bind the pinned nbacv homography code to an explicit court standard.

    The keypoint detector and homography implementation are unchanged.  Only
    the canonical court-space coordinates for the same 33 semantic landmarks
    are replaced before calibration starts.
    """
    spec = get_court_standard(name)
    court_module.CONFIG = spec
    court_module.VERTICES_CM = np.asarray(spec.vertices, dtype=np.float64)
    if court_module.VERTICES_CM.shape != (33, 2):
        raise RuntimeError(f"{spec.name} court must expose exactly 33 landmark vertices")
    payload = spec.manifest()
    payload.update({
        "adapter": "hoopvision.core.court_standards.configure_nbacv_court",
        "nbacv_landmark_semantics_unchanged": True,
        "homography_algorithm_unchanged": True,
    })
    return payload
