"""
Region definitions and boundary data for AI SCOUT.

Provides geographic boundaries for administrative regions that NGOs can be assigned to.
Currently includes Austria states as MVP, expandable to other countries.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Region:
    """A geographic region with bounding box and metadata."""
    id: str
    name: str
    country: str
    country_code: str
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    cities: List[Dict[str, Any]]  # List of major cities with lat/lon


# Austria states (Bundesländer) - MVP dataset
AUSTRIA_STATES = [
    Region(
        id="AT-1",
        name="Burgenland",
        country="Austria",
        country_code="AT",
        bbox=(16.2, 47.0, 17.2, 47.9),
        cities=[
            {"name": "Eisenstadt", "lat": 47.845, "lon": 16.523},
            {"name": "Oberwart", "lat": 47.287, "lon": 16.200},
            {"name": "Neusiedl am See", "lat": 47.949, "lon": 16.843},
        ]
    ),
    Region(
        id="AT-2",
        name="Kärnten",
        country="Austria",
        country_code="AT",
        bbox=(12.6, 46.3, 14.9, 47.1),
        cities=[
            {"name": "Klagenfurt", "lat": 46.624, "lon": 14.308},
            {"name": "Villach", "lat": 46.610, "lon": 13.849},
            {"name": "Wolfsberg", "lat": 46.842, "lon": 14.842},
        ]
    ),
    Region(
        id="AT-3",
        name="Niederösterreich",
        country="Austria",
        country_code="AT",
        bbox=(14.3, 47.4, 17.1, 49.0),
        cities=[
            {"name": "St. Pölten", "lat": 48.204, "lon": 15.625},
            {"name": "Wiener Neustadt", "lat": 47.815, "lon": 16.245},
            {"name": "Krems", "lat": 48.409, "lon": 15.612},
            {"name": "Amstetten", "lat": 48.122, "lon": 14.874},
            {"name": "Mödling", "lat": 48.085, "lon": 16.283},
        ]
    ),
    Region(
        id="AT-4",
        name="Oberösterreich",
        country="Austria",
        country_code="AT",
        bbox=(12.1, 47.5, 14.9, 48.8),
        cities=[
            {"name": "Linz", "lat": 48.306, "lon": 14.286},
            {"name": "Wels", "lat": 48.157, "lon": 14.024},
            {"name": "Steyr", "lat": 48.039, "lon": 14.419},
            {"name": "Leonding", "lat": 48.279, "lon": 14.253},
        ]
    ),
    Region(
        id="AT-5",
        name="Salzburg",
        country="Austria",
        country_code="AT",
        bbox=(12.4, 46.8, 14.2, 48.0),
        cities=[
            {"name": "Salzburg", "lat": 47.809, "lon": 13.055},
            {"name": "Hallein", "lat": 47.682, "lon": 13.100},
            {"name": "Saalfelden", "lat": 47.427, "lon": 12.848},
        ]
    ),
    Region(
        id="AT-6",
        name="Steiermark",
        country="Austria",
        country_code="AT",
        bbox=(13.0, 46.3, 16.2, 47.8),
        cities=[
            {"name": "Graz", "lat": 47.070, "lon": 15.439},
            {"name": "Leoben", "lat": 47.382, "lon": 15.097},
            {"name": "Kapfenberg", "lat": 47.444, "lon": 15.293},
            {"name": "Bruck an der Mur", "lat": 47.410, "lon": 15.270},
        ]
    ),
    Region(
        id="AT-7",
        name="Tirol",
        country="Austria",
        country_code="AT",
        bbox=(10.1, 46.6, 12.8, 47.8),
        cities=[
            {"name": "Innsbruck", "lat": 47.269, "lon": 11.404},
            {"name": "Kufstein", "lat": 47.583, "lon": 12.172},
            {"name": "Telfs", "lat": 47.307, "lon": 11.072},
            {"name": "Hall in Tirol", "lat": 47.283, "lon": 11.507},
        ]
    ),
    Region(
        id="AT-8",
        name="Vorarlberg",
        country="Austria",
        country_code="AT",
        bbox=(9.5, 46.8, 10.2, 47.6),
        cities=[
            {"name": "Bregenz", "lat": 47.503, "lon": 9.747},
            {"name": "Dornbirn", "lat": 47.412, "lon": 9.744},
            {"name": "Feldkirch", "lat": 47.238, "lon": 9.597},
        ]
    ),
    Region(
        id="AT-9",
        name="Wien",
        country="Austria",
        country_code="AT",
        bbox=(16.1, 48.1, 16.6, 48.3),
        cities=[
            {"name": "Wien", "lat": 48.208, "lon": 16.374},
        ]
    ),
]


# Build lookup dictionaries
_REGIONS_BY_ID: Dict[str, Region] = {r.id: r for r in AUSTRIA_STATES}
_REGIONS_BY_NAME: Dict[str, Region] = {r.name.lower(): r for r in AUSTRIA_STATES}


def get_region(region_id: Optional[str]) -> Optional[Region]:
    """Get a region by its ID (e.g., 'AT-3') or name (e.g., 'Niederösterreich')."""
    if not region_id:
        return None
    # Try ID first, then name (case-insensitive)
    region = _REGIONS_BY_ID.get(region_id)
    if not region:
        region = _REGIONS_BY_NAME.get(region_id.lower())
    return region


def get_all_regions() -> List[Region]:
    """Get all available regions."""
    return list(AUSTRIA_STATES)


def get_regions_by_country(country_code: str) -> List[Region]:
    """Get all regions for a specific country code (e.g., 'AT')."""
    return [r for r in AUSTRIA_STATES if r.country_code == country_code.upper()]


def is_point_in_region(lon: float, lat: float, region_id: str) -> bool:
    """Check if a point falls within a region's bounding box."""
    region = get_region(region_id)
    if not region:
        return False
    min_lon, min_lat, max_lon, max_lat = region.bbox
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def is_bbox_in_region(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, region_id: str
) -> bool:
    """Check if a bounding box is entirely contained within a region."""
    region = get_region(region_id)
    if not region:
        return False
    r_min_lon, r_min_lat, r_max_lon, r_max_lat = region.bbox
    return (
        r_min_lon <= min_lon
        and r_max_lon >= max_lon
        and r_min_lat <= min_lat
        and r_max_lat >= max_lat
    )


def validate_geometry_in_region(geometry: Dict[str, Any], region_id: str) -> Tuple[bool, str]:
    """
    Validate that a GeoJSON geometry falls within the user's assigned region.
    
    Returns: (is_valid, error_message)
    """
    region = get_region(region_id)
    if not region:
        return False, f"Unknown region: {region_id}"
    
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    
    if geom_type == "Point":
        lon, lat = coords
        if is_point_in_region(lon, lat, region_id):
            return True, ""
        return False, f"Point ({lat:.4f}°N, {lon:.4f}°E) is outside your assigned region: {region.name}"
    
    elif geom_type == "Polygon":
        # Get all coordinates from the polygon
        poly_coords = coords[0]  # First ring (outer boundary)
        lons = [c[0] for c in poly_coords]
        lats = [c[1] for c in poly_coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        if is_bbox_in_region(min_lon, min_lat, max_lon, max_lat, region_id):
            return True, ""
        return False, (
            f"Selected area extends outside your assigned region: {region.name}. "
            f"Please draw your Area of Interest within {region.name}, Austria."
        )
    
    return False, f"Unsupported geometry type: {geom_type}"
