import os
import simplekml
import logging
from sentinelhub import BBox

logger = logging.getLogger(__name__)

def generate_kml(job_id: str, bbox: BBox, output_dir: str = "results") -> str:
    """
    Generates a KML file for the given job and bounding box.
    """
    try:
        kml = simplekml.Kml()
        kml_filename = f"Mission_{job_id}.kml"
        output_path = os.path.join(output_dir, kml_filename)

        # Create a polygon for the BBox
        # BBox is (minx, miny, maxx, maxy)
        # KML coords are (lon, lat)
        minx, miny, maxx, maxy = bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1]

        # Define the polygon coordinates (closing the loop)
        coords = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny)
        ]

        pol = kml.newpolygon(name=f"Job {job_id} Area", outerboundaryis=coords)
        pol.style.linestyle.color = simplekml.Color.red
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = simplekml.Color.changealphaint(50, simplekml.Color.green) # Semi-transparent green

        # Add a point for the center
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        pnt = kml.newpoint(name="Mission Center", coords=[(center_lon, center_lat)])
        pnt.description = f"Job ID: {job_id}\nCenter: {center_lat:.5f}, {center_lon:.5f}"

        kml.save(output_path)
        logger.info(f"KML generated at {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate KML for job {job_id}: {e}")
        return None
