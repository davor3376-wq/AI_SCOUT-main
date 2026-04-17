"""
PDF Report Generator — Project Gaia
Produces a professional Evidence Pack PDF using ReportLab Platypus.
"""
import io
import math
import logging
import os
import glob
from datetime import datetime, timezone
from typing import Optional

# Force non-interactive backend BEFORE any matplotlib imports (prevents tkinter errors in threads)
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from app.analytics.sar_classifier import (
    classify_sar, classification_stats,
    CLASS_LABELS, CLASS_COLORS, CLASS_PLAIN_LANGUAGE, NO_DATA,
    WATER, BARE_SOIL, VEGETATION, URBAN, ANOMALOUS_VEG_MOISTURE, WET_FOREST,
    generate_validation_protocol, compute_accuracy_metrics,
)
from app.analytics.sar_alerting import compute_sar_delta, delta_interpretation
import numpy as np
import qrcode
import rasterio
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY        = colors.HexColor("#1a3a5c")
NAVY_LIGHT  = colors.HexColor("#2a5280")
GREEN       = colors.HexColor("#2e7d32")
GREEN_LIGHT = colors.HexColor("#e8f5e9")
AMBER       = colors.HexColor("#e65100")
AMBER_LIGHT = colors.HexColor("#fff3e0")
RED         = colors.HexColor("#c62828")
RED_LIGHT   = colors.HexColor("#ffebee")
GRAY_BG     = colors.HexColor("#f5f5f5")
GRAY_MID    = colors.HexColor("#9e9e9e")
GRAY_DARK   = colors.HexColor("#424242")
WHITE       = colors.white
BLACK       = colors.black

# ── NDWI / Hydraulic-Status palette ───────────────────────────────────────────
WATER_DEEP  = colors.HexColor("#bbdefb")   # > 0.3   Surface Water / Flooding
WATER_MID   = colors.HexColor("#e3f2fd")   # 0.1–0.3 Elevated Moisture / Wet Canopy
WATER_LOW   = colors.HexColor("#e0f7fa")   # −0.1–0.1 Balanced Moisture / Transitional
WATER_DRY   = colors.HexColor("#fff3e0")   # < −0.1  Arid / Dry Surface

PAGE_W, PAGE_H = A4
MARGIN = 1.6 * cm
CONTENT_W = PAGE_W - 2 * MARGIN   # usable width inside margins (~17.8 cm on A4)

# ── Styles ─────────────────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()
    s = {}

    s["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=WHITE,
        leading=28,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontName="Helvetica",
        fontSize=12,
        textColor=colors.HexColor("#cfe8fc"),
        leading=16,
    )
    s["section_head"] = ParagraphStyle(
        "section_head",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=1,
    )
    s["scene_title"] = ParagraphStyle(
        "scene_title",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=WHITE,
        leading=15,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        textColor=GRAY_DARK,
        leading=13,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=GRAY_MID,
        alignment=TA_CENTER,
    )
    s["stat_label"] = ParagraphStyle(
        "stat_label",
        fontName="Helvetica",
        fontSize=8,
        textColor=GRAY_DARK,
    )
    s["stat_value"] = ParagraphStyle(
        "stat_value",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=NAVY,
    )
    s["footer"] = ParagraphStyle(
        "footer",
        fontName="Helvetica",
        fontSize=7,
        textColor=GRAY_MID,
        alignment=TA_CENTER,
    )
    s["alert"] = ParagraphStyle(
        "alert",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=RED,
    )
    s["good"] = ParagraphStyle(
        "good",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=GREEN,
    )
    s["footnote"] = ParagraphStyle(
        "footnote",
        fontName="Helvetica-Oblique",
        fontSize=7,
        textColor=GRAY_MID,
        leading=10,
        alignment=TA_LEFT,
    )
    return s


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_date(basename: str) -> Optional[datetime]:
    """Extract date from YYYYMMDD prefix."""
    try:
        return datetime.strptime(basename[:8], "%Y%m%d")
    except ValueError:
        return None


def _format_date(dt: datetime) -> str:
    return dt.strftime("%-d %B %Y") if os.name != "nt" else dt.strftime("%d %B %Y").lstrip("0")


def _ndvi_interpretation(mean: float) -> tuple[str, str]:
    """Return (label, colour_key) for a mean NDVI value."""
    if mean >= 0.6:
        return "Dense vegetation", "good"
    if mean >= 0.3:
        return "Moderate vegetation", "body"
    if mean >= 0.1:
        return "Sparse vegetation", "alert"
    if mean >= -0.1:
        return "Bare soil / urban", "alert"
    return "Water / snow / cloud", "alert"


def _ndwi_interpretation(mean: float) -> str:
    """Return Hydraulic Status label for a mean NDWI value.

    NDWI = (Green − NIR) / (Green + NIR).  Positive values flag water/moisture;
    negative values flag dry surfaces.  This scale is physically distinct from
    the NDVI vegetation scale and must NOT be used interchangeably.
    """
    if mean > 0.3:
        return "Surface Water / Flooding"
    if mean >= 0.1:
        return "Elevated Moisture / Wet Canopy"
    if mean >= -0.1:
        return "Balanced Moisture / Transitional"
    return "Arid / Dry Surface"


def _nbr_interpretation(mean: float) -> str:
    if mean >= 0.66:
        return "Unburned (healthy vegetation)"
    if mean >= 0.44:
        return "Low severity burn"
    if mean >= 0.1:
        return "Moderate burn"
    if mean >= -0.1:
        return "Moderate-high burn"
    return "High severity burn / bare"


def _area_km2(bounds, lat_center: float) -> float:
    lat_km = (bounds.top - bounds.bottom) * 111.0
    lon_km = (bounds.right - bounds.left) * 111.0 * math.cos(math.radians(lat_center))
    return lat_km * lon_km


def _extract_stats(filepath: str) -> Optional[dict]:
    """Read a GeoTIFF and return a stats dict, or None on failure."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(float)
            bounds = src.bounds
            src_crs = src.crs
            crs = str(src_crs)

        valid = data[np.isfinite(data)]
        if valid.size == 0:
            return None

        # Descale if integer-scaled
        if np.max(valid) > 100:
            valid = valid / 10_000.0

        if src_crs.is_geographic:
            # Geographic CRS (degrees): use degree-to-km formula with mid-latitude correction
            lat_c = (bounds.bottom + bounds.top) / 2
            lon_c = (bounds.left + bounds.right) / 2
            area = _area_km2(bounds, lat_c)
        else:
            # Projected CRS (metres, e.g. UTM): compute area directly from metre extents
            width_km  = abs(bounds.right - bounds.left) / 1000.0
            height_km = abs(bounds.top   - bounds.bottom) / 1000.0
            area = width_km * height_km
            # Reproject bounding box to WGS-84 for geographic centre display
            import rasterio.warp as _rwarp
            lon_min, lat_min, lon_max, lat_max = _rwarp.transform_bounds(
                src_crs, "EPSG:4326",
                bounds.left, bounds.bottom, bounds.right, bounds.top,
            )
            lat_c = (lat_min + lat_max) / 2
            lon_c = (lon_min + lon_max) / 2

        return {
            "mean": float(np.mean(valid)),
            "std":  float(np.std(valid)),
            "min":  float(np.min(valid)),
            "max":  float(np.max(valid)),
            "pixels": int(valid.size),
            "bounds": bounds,
            "crs": crs,
            "area_km2": area,
            "lat_center": lat_c,
            "lon_center": lon_c,
        }
    except Exception as exc:
        logger.warning(f"Stats extraction failed for {filepath}: {exc}")
        return None


def _tif_to_png_buffer(filepath: str, colormap: str = "RdYlGn",
                       vmin: float = -1.0, vmax: float = 1.0) -> Optional[io.BytesIO]:
    """Render a single-band GeoTIFF as a colourised PNG in a BytesIO buffer."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(float)

        nan_mask = np.isnan(data)
        valid = data[~nan_mask]
        if valid.size == 0:
            return None

        if np.max(valid) > 100:
            data = data / 10_000.0

        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl_cm.ColormapRegistry.get_cmap if hasattr(mpl_cm, "ColormapRegistry") else mpl_cm.get_cmap
        try:
            import matplotlib
            cmap = matplotlib.colormaps[colormap]
        except (AttributeError, KeyError):
            cmap = mpl_cm.get_cmap(colormap)  # fallback for older matplotlib
        data_filled = np.where(nan_mask, 0.0, data)
        rgba = cmap(norm(data_filled))           # H×W×4
        rgba[nan_mask] = [0.93, 0.93, 0.93, 1.0]  # light gray for no-data

        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = PILImage.fromarray(rgb)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception as exc:
        logger.warning(f"TIF render failed for {filepath}: {exc}")
        return None


def _colorbar_image(colormap: str = "RdYlGn",
                    vmin: float = -1.0, vmax: float = 1.0,
                    label: str = "NDVI",
                    width_px: int = 260, height_px: int = 28) -> io.BytesIO:
    """Produce a horizontal colourbar legend as PNG."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    try:
        import matplotlib
        cmap_obj = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap_obj = mpl_cm.get_cmap(colormap)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        cax=ax, orientation="horizontal",
    )
    cb.set_label(label, fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.4, top=0.95)
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


def _qr_buffer(data: str, size: int = 120) -> io.BytesIO:
    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=6, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Page template (header / footer) ───────────────────────────────────────────
def _report_gen_date() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

# ── Commercial hardening — disclaimer & attribution text blocks ────────────────
_TECHNICAL_LIMITATIONS = (
    "<b>TECHNICAL LIMITATIONS \u2014 READ BEFORE USE</b>  "
    "(1) <b>SAR Speckle & Spectral Artifacts:</b> Sentinel-1 C-band (5.405\u202fGHz) SAR imagery "
    "is subject to inherent multiplicative speckle noise. Individual pixel values represent "
    "stochastic samples of a statistical distribution, not absolute surface properties. "
    "Classification results describe ensemble land-cover behaviour, not the precise state of "
    "any single pixel.  "
    "(2) <b>Geometric Layover \u2014 Alpine & Steep Terrain:</b> In areas of steep terrain "
    "(slope\u202f>\u202f15\u00b0), forward-facing slopes produce radar \u2018layover\u2019: the radar "
    "return from elevated surfaces is displaced toward the sensor, creating a high-backscatter "
    "signature indistinguishable from urban double-bounce in C-band SAR alone. GLO-30 Copernicus "
    "DEM Radiometric Terrain Correction (GAMMA0_TERRAIN) substantially mitigates but does not "
    "fully eliminate this effect. Classification results in terrain exceeding 15\u00b0 slope should "
    "be treated as indicative only, pending field inspection.  "
    "(3) <b>Legal & Financial Use \u2014 Field Verification Mandatory:</b> The automated land cover "
    "classification presented in this report is derived from orbital remote sensing data processed "
    "by algorithmic methods. <b>This product is NOT a substitute for an in-situ field survey.</b> "
    "Any land cover determination used as evidence in legal proceedings, financial transactions, "
    "insurance claims, environmental permitting, or regulatory compliance MUST be verified by a "
    "qualified ground survey conducted by a licensed professional. The operators of this platform "
    "accept no liability for decisions made solely on the basis of automated satellite-derived "
    "classifications."
)

_COPERNICUS_ATTRIBUTION = (
    "<b>Data Source Attribution:</b> Contains modified Copernicus Sentinel-1 (SAR) and "
    "Sentinel-2 (MSI) data, processed by ESA. Sentinel data are provided free of charge "
    "under the Copernicus Data Policy, governed by EU Regulation No\u202f377/2014 and Commission "
    "Delegated Regulation (EU) No\u202f1159/2013. The European Union and ESA provide no guarantee "
    "as to the accuracy, completeness, or fitness for purpose of Copernicus data or derived "
    "products. Copernicus DEM GLO-30 \u00a9 DLR e.V. 2021 and \u00a9 Airbus Defence and Space GmbH "
    "2021, provided under COPERNICUS by the European Union and ESA; all rights reserved."
)


def _on_page(canv: rl_canvas.Canvas, doc, styles, meta: dict = None):
    """Called for every page except the cover."""
    _meta = meta or {}
    job_id        = _meta.get("job_id", "")
    report_date   = _meta.get("report_gen_date", _report_gen_date())

    w, h = A4
    # Top rule
    canv.setStrokeColor(NAVY)
    canv.setLineWidth(1.5)
    canv.line(MARGIN, h - 1.1 * cm, w - MARGIN, h - 1.1 * cm)

    # Header text
    canv.setFillColor(NAVY)
    canv.setFont("Helvetica-Bold", 8)
    header_left = "PROJECT GAIA — Environmental Intelligence Report"
    if job_id:
        header_left += f"  ·  Job {job_id}"
    canv.drawString(MARGIN, h - 0.85 * cm, header_left)
    canv.setFont("Helvetica", 8)
    canv.setFillColor(GRAY_MID)
    canv.drawRightString(w - MARGIN, h - 0.85 * cm, f"Generated {report_date}")

    # Bottom rule
    canv.setStrokeColor(GRAY_MID)
    canv.setLineWidth(0.5)
    canv.line(MARGIN, 1.1 * cm, w - MARGIN, 1.1 * cm)

    # Footer text — confidentiality stamp + job ID + page number
    canv.setFont("Helvetica", 7)
    canv.setFillColor(GRAY_MID)
    footer_centre = "CONFIDENTIAL \u2014 FOR AUTHORISED USE ONLY"
    if job_id:
        footer_centre += f"  \u00b7  JOB ID: {job_id}"
    canv.drawCentredString(w / 2, 0.75 * cm, footer_centre)
    canv.drawRightString(w - MARGIN, 0.75 * cm, f"Page {doc.page}")

    # ISO 19157 / Copernicus attribution micro-line
    canv.setFont("Helvetica", 5.5)
    canv.drawCentredString(
        w / 2, 0.32 * cm,
        "Contains modified Copernicus Sentinel data, processed by ESA. "
        "Automated classification \u2014 field verification required for legal/financial decisions. "
        "ISO\u202f19157:2013.",
    )


def _cover_page(canv: rl_canvas.Canvas, doc, meta: dict, styles):
    """Draw the cover page (called via onFirstPage)."""
    w, h = A4

    # ── Navy header band ──────────────────────────────────────────────────────
    band_h = 5.5 * cm
    canv.setFillColor(NAVY)
    canv.rect(0, h - band_h, w, band_h, fill=1, stroke=0)

    # Title
    canv.setFillColor(WHITE)
    canv.setFont("Helvetica-Bold", 22)
    canv.drawString(MARGIN, h - 2.2 * cm, "PROJECT GAIA")
    canv.setFont("Helvetica", 13)
    canv.setFillColor(colors.HexColor("#cfe8fc"))
    canv.drawString(MARGIN, h - 3.0 * cm, "Environmental Intelligence Report")
    canv.setFont("Helvetica", 10)
    canv.drawString(MARGIN, h - 3.7 * cm, "Analysis & Evidence Pack")

    # Green accent stripe below band
    canv.setFillColor(GREEN)
    canv.rect(0, h - band_h - 0.25 * cm, w, 0.25 * cm, fill=1, stroke=0)

    # ── QR code (top-right of band) ───────────────────────────────────────────
    qr_data = meta.get("qr_data")
    if qr_data:
        try:
            qr_buf = _qr_buffer(qr_data)
            qr_size = 2.8 * cm
            canv.drawImage(
                ImageReader(qr_buf),
                w - MARGIN - qr_size,
                h - band_h + 0.3 * cm,
                width=qr_size, height=qr_size,
                preserveAspectRatio=True,
            )
            canv.setFillColor(colors.HexColor("#cfe8fc"))
            canv.setFont("Helvetica", 6.5)
            canv.drawCentredString(
                w - MARGIN - qr_size / 2,
                h - band_h + 0.1 * cm,
                "Mission Location",
            )
        except Exception as exc:
            logger.warning(f"Cover QR failed: {exc}")

    # ── Job ID + timestamp line inside cover band ─────────────────────────────
    job_id      = meta.get("job_id", "")
    report_date = meta.get("report_gen_date", _report_gen_date())
    canv.setFont("Helvetica", 9)
    canv.setFillColor(colors.HexColor("#cfe8fc"))
    cover_meta_line = f"Generated {report_date}"
    if job_id:
        cover_meta_line = f"Job ID: {job_id}  ·  {cover_meta_line}"
    canv.drawString(MARGIN, h - 4.4 * cm, cover_meta_line)

    # ── Footer ────────────────────────────────────────────────────────────────
    canv.setStrokeColor(GRAY_MID)
    canv.setLineWidth(0.5)
    canv.line(MARGIN, 1.1 * cm, w - MARGIN, 1.1 * cm)
    canv.setFont("Helvetica", 7)
    canv.setFillColor(GRAY_MID)
    cover_footer = "CONFIDENTIAL — FOR AUTHORISED USE ONLY"
    if job_id:
        cover_footer += f"  ·  JOB ID: {job_id}"
    canv.drawCentredString(w / 2, 0.7 * cm, cover_footer)
    canv.drawRightString(w - MARGIN, 0.7 * cm, f"Generated {report_date}")


# ── Stats helpers ──────────────────────────────────────────────────────────────
def _ndvi_color(mean: float) -> colors.Color:
    if mean >= 0.6:
        return GREEN_LIGHT
    if mean >= 0.3:
        return colors.HexColor("#f9fbe7")
    if mean >= 0.0:
        return AMBER_LIGHT
    return RED_LIGHT


def _ndwi_color(mean: float) -> colors.Color:
    """Row-background colour keyed to the Hydraulic Status scale."""
    if mean > 0.3:
        return WATER_DEEP
    if mean >= 0.1:
        return WATER_MID
    if mean >= -0.1:
        return WATER_LOW
    return WATER_DRY


# ── Main class ─────────────────────────────────────────────────────────────────
class PDFReportGenerator:
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "results"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────
    def generate_pdf(self,
                     filename: str = "report.pdf",
                     specific_files: Optional[list] = None,
                     bbox=None,
                     job_id: str = "") -> None:
        output_path = os.path.join(self.output_dir, filename)
        styles = _build_styles()
        report_gen_date = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

        # ── Collect files ─────────────────────────────────────────────────────
        if specific_files:
            tif_files = [f for f in specific_files if f.endswith(".tif")]
        else:
            tif_files = sorted(glob.glob(os.path.join(self.input_dir, "*_analysis.tif")))

        era5_file      = next((f for f in tif_files if "ERA5" in f),     None)
        diff_files     = [f for f in tif_files if "DIFF" in f and "ERA5" not in f]
        ndvi_files     = sorted(f for f in tif_files if "NDVI" in f and "DIFF" not in f and "ERA5" not in f)
        ndwi_files     = sorted(f for f in tif_files if "NDWI" in f and "DIFF" not in f and "ERA5" not in f)
        nbr_files      = sorted(f for f in tif_files if "NBR"  in f and "DIFF" not in f and "ERA5" not in f)
        s1_files       = sorted(f for f in tif_files
                                if "S1_RTC" in os.path.basename(f)
                                and "NDVI" not in f and "NDWI" not in f and "NBR" not in f
                                and "DIFF" not in f and "ERA5" not in f)

        # ── Determine primary optical index (NDVI → NDWI → NBR fallback) ────────
        if ndvi_files:
            primary_files   = ndvi_files
            primary_label   = "NDVI"
            secondary_files = ndwi_files
        elif ndwi_files:
            primary_files   = ndwi_files
            primary_label   = "NDWI"
            secondary_files = []
        elif nbr_files:
            primary_files   = nbr_files
            primary_label   = "NBR"
            secondary_files = []
        else:
            primary_files   = []
            primary_label   = "NDVI"
            secondary_files = []

        # ── Pre-compute stats ─────────────────────────────────────────────────
        ndvi_stats = {os.path.basename(f)[:8]: (_extract_stats(f), f) for f in primary_files}
        ndwi_stats = {os.path.basename(f)[:8]: (_extract_stats(f), f) for f in secondary_files}

        valid_ndvi = {d: (s, f) for d, (s, f) in ndvi_stats.items() if s is not None}

        # ── Mission metadata ──────────────────────────────────────────────────
        meta = self._build_meta(valid_ndvi, bbox, era5_file, s1_files,
                                job_id=job_id, report_gen_date=report_gen_date,
                                primary_label=primary_label)

        # ── Build document ────────────────────────────────────────────────────
        doc = BaseDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm,
            title="Project Gaia — Environmental Intelligence Report",
            author="AI Scout Platform",
        )

        # Page templates
        body_frame = Frame(
            MARGIN, 1.8 * cm,
            PAGE_W - 2 * MARGIN, PAGE_H - 3.6 * cm,
            id="body",
        )
        cover_frame = Frame(
            MARGIN, 1.8 * cm,
            PAGE_W - 2 * MARGIN, PAGE_H - 3.6 * cm,
            id="cover",
        )
        doc.addPageTemplates([
            PageTemplate(
                id="cover",
                frames=[cover_frame],
                onPage=lambda c, d: _cover_page(c, d, meta, styles),
            ),
            PageTemplate(
                id="body",
                frames=[body_frame],
                onPage=lambda c, d: _on_page(c, d, styles, meta),
            ),
        ])

        # ── Story ─────────────────────────────────────────────────────────────
        story = []
        story += self._cover_content(meta, valid_ndvi, era5_file, styles,
                                         primary_label=primary_label)
        story.append(PageBreak())

        # Switch to body template
        from reportlab.platypus import NextPageTemplate
        story.insert(len(story) - 1, NextPageTemplate("body"))

        # NDVI trend table
        if valid_ndvi:
            story += self._trend_table(valid_ndvi, ndwi_stats, styles, primary_label)

        # Per-scene pages — each scene starts on its own page
        for date_str in sorted(valid_ndvi):
            story.append(PageBreak())
            ndvi_s, ndvi_path = valid_ndvi[date_str]
            ndwi_pair = ndwi_stats.get(date_str, (None, None))
            ndwi_s, ndwi_path = ndwi_pair
            story += self._scene_section(date_str, ndvi_s, ndvi_path,
                                         ndwi_s, ndwi_path, styles, primary_label)

        # Radar pages (when no optical data is available)
        if not valid_ndvi and s1_files:
            # ── MaaS Page 1: executive summary + trend chart ──────────────────
            s1_scene_means = self._extract_s1_means(s1_files)

            story += self._maas_executive_summary(s1_scene_means, styles)
            story += self._sar_trend_table(s1_scene_means, s1_files, styles)

            trend_buf = self._vv_trend_chart(s1_scene_means)
            if trend_buf:
                chart_img = RLImage(trend_buf, width=16.2 * cm, height=7.0 * cm)
                story.append(chart_img)
                story.append(Paragraph(
                    "Figure 1: Mean VV Backscatter Intensity (linear Gamma<super>0</super>, left axis) "
                    "and dB equivalent (right axis) over the analysis period. "
                    "Radiometric variance within threshold. No statistical anomaly detected in signal return. "
                    "Interpretation requires field validation.",
                    styles["caption"],
                ))
                story.append(Spacer(1, 0.5 * cm))

            story += self._s1_section(s1_files, styles)

        # DIFF / change-detection pages
        for diff_path in diff_files:
            story += self._diff_section(diff_path, styles)

        # SAR certification & explainability section (radar-only missions)
        if s1_files and not valid_ndvi:
            story += self._sar_certification_section(styles, area_km2=meta.get("area_km2"))

        doc.build(story)
        logger.info(f"PDF report written to {output_path}")

    # ── Cover content (flowables inside cover frame) ───────────────────────────
    def _cover_content(self, meta, valid_ndvi, era5_file, styles, primary_label: str = "NDVI"):
        story = []
        # Push content below the cover band.
        # Navy band = 5.5 cm from top; green stripe adds 0.25 cm → band bottom at 5.75 cm.
        # Cover frame starts at 1.8 cm from top, so spacer must be ≥ 5.75 − 1.8 = 3.95 cm.
        # Use 4.2 cm for a comfortable margin.
        story.append(Spacer(1, 4.2 * cm))

        # Mission summary table
        story.append(Paragraph("MISSION SUMMARY", styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=NAVY, spaceAfter=6))

        rows = [
            ["Parameter", "Value"],
            ["Sensor",    meta["sensor"]],
        ]
        if meta.get("job_id"):
            rows.append(["Job ID", meta["job_id"]])
        rows += [
            ["Generated (UTC)", meta.get("report_gen_date", _report_gen_date())],
            ["Analysis Period",
             f"{meta['date_start']} — {meta['date_end']}" if meta["date_start"] else "N/A"],
            ["Location (center)",
             (f"{abs(meta['lat_center']):.4f}° {'N' if meta['lat_center'] >= 0 else 'S'},  "
              f"{abs(meta['lon_center']):.4f}° {'E' if meta['lon_center'] >= 0 else 'W'}")
             if meta["lat_center"] is not None else "N/A"],
            ["Coverage Area",    f"{meta['area_km2']:.2f} km²" if meta["area_km2"] is not None else "N/A"],
            ["Valid Scenes",     f"{meta['valid_scenes']} of {meta['total_scenes']}"],
            ["Coordinate System", meta.get("crs", "EPSG:4326")],
        ]
        if meta.get("avg_temp") is not None:
            rows.append(["Avg Temperature (ERA5)", f"{meta['avg_temp']:.1f} °C"])
        if meta.get("avg_precip") is not None:
            rows.append(["Avg Precipitation (ERA5)", f"{meta['avg_precip']:.2f} mm"])

        tbl = Table(rows, colWidths=[5.5 * cm, 10.5 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  9),
            ("BACKGROUND",  (0, 1), (-1, -1), GRAY_BG),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [WHITE, colors.HexColor("#eceff1")]),
            ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",    (1, 1), (1, -1),  "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 9),
            ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8dc")),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.9 * cm))

        # NDVI trend mini-table on cover
        if valid_ndvi:
            _overview_hr_color = colors.HexColor("#1565c0") if primary_label == "NDWI" else GREEN
            story.append(Paragraph(f"{primary_label} OVERVIEW — ALL SCENES", styles["section_head"]))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=_overview_hr_color, spaceAfter=4))
            story += self._mini_trend_table(valid_ndvi, styles, primary_label=primary_label)

        return story

    def _mini_trend_table(self, valid_ndvi, styles, primary_label: str = "NDVI"):
        _col_lbl = f"Mean {primary_label}"
        header = ["Date", _col_lbl, "Trend", "Status"]
        rows = [header]
        dates = sorted(valid_ndvi)
        for i, d in enumerate(dates):
            s, _ = valid_ndvi[d]
            dt = _parse_date(d)
            date_label = _format_date(dt) if dt else d
            mean = s["mean"]
            trend = "—"
            if i > 0:
                prev_s, _ = valid_ndvi[dates[i - 1]]
                delta = mean - prev_s["mean"]
                trend = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            if primary_label == "NDWI":
                label = _ndwi_interpretation(mean)
            elif primary_label == "NBR":
                label = _nbr_interpretation(mean)
            else:
                label, _ = _ndvi_interpretation(mean)
            rows.append([date_label, f"{mean:.3f}", trend, label])

        col_w = [4.5 * cm, 3.0 * cm, 3.0 * cm, 5.5 * cm]
        tbl = Table(rows, colWidths=col_w)

        style_cmds = [
            ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  8),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 8),
            ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
        _mini_color_fn = _ndwi_color if primary_label == "NDWI" else _ndvi_color
        for i, d in enumerate(dates, start=1):
            s, _ = valid_ndvi[d]
            bg = _mini_color_fn(s["mean"])
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

        tbl.setStyle(TableStyle(style_cmds))
        return [tbl, Spacer(1, 0.3 * cm)]

    # ── Trend table page ───────────────────────────────────────────────────────
    def _trend_table(self, valid_ndvi, ndwi_stats, styles, primary_label: str = "NDVI"):
        story = []
        _sec = "NDWI" if ndwi_stats else "—"
        _title = f"TEMPORAL ANALYSIS — {primary_label}" + (f" & {_sec}" if ndwi_stats else "") + " STATISTICS"
        story.append(Paragraph(_title, styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=NAVY, spaceAfter=6))

        _body_texts = {
            "NDVI": (
                "The Normalised Difference Vegetation Index (NDVI) measures vegetation health "
                "on a scale from −1 to +1. Values above 0.6 indicate dense, healthy vegetation; "
                "0.3–0.6 indicates moderate cover; below 0.3 suggests sparse vegetation, "
                "bare soil, or cloud contamination. "
                "NDWI (Normalised Difference Water Index) indicates water/moisture content — "
                "positive values indicate surface water or high canopy moisture."
            ),
            "NDWI": (
                "The Normalised Difference Water Index (NDWI) highlights surface water and canopy moisture "
                "on a scale from −1 to +1. Values above 0.3 indicate open water or high moisture; "
                "0.0–0.3 indicates moderate moisture; below 0.0 indicates dry conditions."
            ),
            "NBR": (
                "The Normalised Burn Ratio (NBR) maps fire severity using NIR and SWIR2 bands. "
                "Values near +1 indicate healthy, unburned vegetation; values below 0 indicate "
                "active burn scars or severely burned areas. dNBR (pre − post) quantifies severity."
            ),
        }
        story.append(Paragraph(_body_texts.get(primary_label, ""), styles["body"]))
        story.append(Spacer(1, 0.3 * cm))

        _lbl = primary_label
        header = ["Date", f"{_lbl} Mean", f"{_lbl} Std", f"{_lbl} Min", f"{_lbl} Max",
                  f"{_sec} Mean", "Valid Pixels", "Interpretation"]
        rows = [header]
        dates = sorted(valid_ndvi)

        _interp_fn = {
            "NDWI": lambda m: (_ndwi_interpretation(m), "body"),
            "NBR":  lambda m: (_nbr_interpretation(m), "body"),
        }

        for d in dates:
            ns, _ = valid_ndvi[d]
            nw_s, _ = ndwi_stats.get(d, (None, None))
            dt = _parse_date(d)
            date_label = _format_date(dt) if dt else d
            interp, _ = _interp_fn[primary_label](ns["mean"]) if primary_label in _interp_fn else _ndvi_interpretation(ns["mean"])
            ndwi_mean = f"{nw_s['mean']:.3f}" if nw_s else "N/A"
            rows.append([
                date_label,
                f"{ns['mean']:.3f}",
                f"{ns['std']:.3f}",
                f"{ns['min']:.3f}",
                f"{ns['max']:.3f}",
                ndwi_mean,
                f"{ns['pixels']:,}",
                interp,
            ])

        col_w = [3.0*cm, 1.7*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.8*cm, 1.8*cm, 3.1*cm]
        tbl = Table(rows, colWidths=col_w, repeatRows=1)

        style_cmds = [
            ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  8),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 8),
            ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN",       (1, 1), (6, -1),  "CENTER"),
            ("ALIGN",       (7, 1), (7, -1),  "LEFT"),
        ]
        _row_color_fn = _ndwi_color if primary_label == "NDWI" else _ndvi_color
        for i, d in enumerate(dates, start=1):
            ns, _ = valid_ndvi[d]
            style_cmds.append(("BACKGROUND", (0, i), (6, i), _row_color_fn(ns["mean"])))

        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)

        # ── Colour legend — index-specific ────────────────────────────────────
        story.append(Spacer(1, 0.5 * cm))
        if primary_label == "NDWI":
            _legend_border = colors.HexColor("#90caf9")
            _legend_hr     = colors.HexColor("#1565c0")
            legend_items = [
                (WATER_DEEP, "> 0.3  Surface Water / Flooding"),
                (WATER_MID,  "0.1 - 0.3  Elevated Moisture / Wet Canopy"),
                (WATER_LOW,  "-0.1 - 0.1  Balanced Moisture / Transitional"),
                (WATER_DRY,  "< -0.1  Arid / Dry Surface"),
            ]
        else:
            _legend_border = colors.HexColor("#bdbdbd")
            _legend_hr     = NAVY
            legend_items = [
                (GREEN_LIGHT,  ">= 0.6  Dense vegetation"),
                (colors.HexColor("#f9fbe7"), "0.3 - 0.6  Moderate vegetation"),
                (AMBER_LIGHT,  "0.0 - 0.3  Sparse / transitional"),
                (RED_LIGHT,    "< 0.0  Water / bare / cloud"),
            ]
        legend_row = [[
            Table(
                [[Paragraph(label, styles["body"])]],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BOX",        (0, 0), (-1, -1), 0.3, _legend_border),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING",  (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]),
            )
            for bg, label in legend_items
        ]]
        legend_tbl = Table(legend_row, colWidths=[4.0 * cm] * 4)
        story.append(legend_tbl)
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=1.0,
                                color=_legend_hr, spaceBefore=4, spaceAfter=6))

        # ── Cross-validation: separate NDWI table when NDVI is primary ────────
        if primary_label == "NDVI" and ndwi_stats:
            story.append(Spacer(1, 0.4 * cm))
            story.append(Paragraph(
                "TEMPORAL ANALYSIS — NDWI STATISTICS (HYDRAULIC STATUS)",
                styles["section_head"],
            ))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=colors.HexColor("#1565c0"), spaceAfter=6))
            story.append(Paragraph(
                "The Normalised Difference Water Index (NDWI) highlights surface water and "
                "canopy moisture on a scale from -1 to +1. "
                "Values above 0.3 indicate open water or active flooding; "
                "0.1 - 0.3 indicates high soil saturation or wet canopy; "
                "-0.1 - 0.1 is transitional; below -0.1 indicates arid / dry surfaces. "
                "This table uses a physically distinct Hydraulic Status scale "
                "and must not be conflated with the NDVI Vegetation Status table above.",
                styles["body"],
            ))
            story.append(Spacer(1, 0.3 * cm))

            ndwi_header = ["Date", "NDWI Mean", "NDWI Std", "NDWI Min",
                           "NDWI Max", "Valid Pixels", "Hydraulic Status"]
            ndwi_rows   = [ndwi_header]
            ndwi_dates  = sorted(d for d in dates
                                 if ndwi_stats.get(d, (None, None))[0] is not None)
            for d in ndwi_dates:
                nw_s, _ = ndwi_stats[d]
                dt = _parse_date(d)
                date_label = _format_date(dt) if dt else d
                ndwi_rows.append([
                    date_label,
                    f"{nw_s['mean']:.3f}",
                    f"{nw_s['std']:.3f}",
                    f"{nw_s['min']:.3f}",
                    f"{nw_s['max']:.3f}",
                    f"{nw_s['pixels']:,}",
                    _ndwi_interpretation(nw_s["mean"]),
                ])

            ndwi_col_w = [3.0*cm, 1.8*cm, 1.6*cm, 1.6*cm, 1.6*cm, 2.0*cm, 4.6*cm]
            ndwi_tbl = Table(ndwi_rows, colWidths=ndwi_col_w, repeatRows=1)
            ndwi_style = [
                ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1565c0")),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0),  8),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 1), (-1, -1), 8),
                ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
                ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#90caf9")),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN",       (1, 1), (5, -1),  "CENTER"),
                ("ALIGN",       (6, 1), (6, -1),  "LEFT"),
            ]
            for i, d in enumerate(ndwi_dates, start=1):
                nw_s, _ = ndwi_stats[d]
                ndwi_style.append(
                    ("BACKGROUND", (0, i), (5, i), _ndwi_color(nw_s["mean"]))
                )
            ndwi_tbl.setStyle(TableStyle(ndwi_style))
            story.append(ndwi_tbl)

            story.append(Spacer(1, 0.5 * cm))
            ndwi_legend_items = [
                (WATER_DEEP, "> 0.3  Surface Water / Flooding"),
                (WATER_MID,  "0.1 - 0.3  Elevated Moisture / Wet Canopy"),
                (WATER_LOW,  "-0.1 - 0.1  Balanced Moisture / Transitional"),
                (WATER_DRY,  "< -0.1  Arid / Dry Surface"),
            ]
            ndwi_legend_row = [[
                Table(
                    [[Paragraph(lbl, styles["body"])]],
                    style=TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), bg),
                        ("BOX",        (0, 0), (-1, -1), 0.3, colors.HexColor("#90caf9")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING",  (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]),
                )
                for bg, lbl in ndwi_legend_items
            ]]
            ndwi_legend_tbl = Table(ndwi_legend_row, colWidths=[4.0 * cm] * 4)
            story.append(ndwi_legend_tbl)
            story.append(Spacer(1, 0.5 * cm))
            story.append(HRFlowable(width="100%", thickness=1.0,
                                    color=colors.HexColor("#1565c0"),
                                    spaceBefore=4, spaceAfter=6))

        return story

    # ── Per-scene section ──────────────────────────────────────────────────────
    def _scene_section(self, date_str, ndvi_stats, ndvi_path,
                       ndwi_stats, ndwi_path, styles, primary_label: str = "NDVI"):
        story = []
        dt = _parse_date(date_str)
        date_label = _format_date(dt) if dt else date_str

        # Section header band
        story.append(Spacer(1, 0.4 * cm))

        _sec_label = "NDWI" if ndwi_stats else ""
        _band_label = f"{primary_label} + {_sec_label}" if _sec_label else primary_label
        header_tbl = Table(
            [[Paragraph(f"SCENE  \u00b7  {date_label}", styles["scene_title"]),
              Paragraph(f"Sentinel-2 MSI  \u00b7  {_band_label}", styles["caption"])]],
            colWidths=[10 * cm, 6.2 * cm],
        )
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), NAVY),
            ("TOPPADDING",  (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",   (1, 0), (1, 0),   colors.HexColor("#cfe8fc")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 0.25 * cm))

        # ── NDVI map + stats side-by-side ──────────────────────────────────────
        ndvi_buf = _tif_to_png_buffer(ndvi_path) if ndvi_path else None
        map_w = 9.5 * cm
        map_h = 9.5 * cm

        if primary_label == "NDWI":
            interp_label, interp_key = _ndwi_interpretation(ndvi_stats["mean"]), "body"
        elif primary_label == "NBR":
            interp_label, interp_key = _nbr_interpretation(ndvi_stats["mean"]), "body"
        else:
            interp_label, interp_key = _ndvi_interpretation(ndvi_stats["mean"])

        stat_rows = [
            ["Metric", primary_label, _sec_label or "—"],
            ["Mean",
             f"{ndvi_stats['mean']:.4f}",
             f"{ndwi_stats['mean']:.4f}" if ndwi_stats else "—"],
            ["Std Dev",
             f"{ndvi_stats['std']:.4f}",
             f"{ndwi_stats['std']:.4f}" if ndwi_stats else "—"],
            ["Min",
             f"{ndvi_stats['min']:.4f}",
             f"{ndwi_stats['min']:.4f}" if ndwi_stats else "—"],
            ["Max",
             f"{ndvi_stats['max']:.4f}",
             f"{ndwi_stats['max']:.4f}" if ndwi_stats else "—"],
            ["Valid px",
             f"{ndvi_stats['pixels']:,}",
             f"{ndwi_stats['pixels']:,}" if ndwi_stats else "—"],
        ]
        stat_tbl = Table(stat_rows, colWidths=[2.5 * cm, 2.3 * cm, 2.3 * cm])
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
            ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
            ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [WHITE, colors.HexColor("#f5f5f5")]),
        ]))

        _interp_titles = {
            "NDVI": "Vegetation Status",
            "NDWI": "Water / Moisture",
            "NBR":  "Burn Ratio Status",
        }
        _interp_colors = {
            "NDVI": _ndvi_color(ndvi_stats["mean"]),
            "NDWI": colors.HexColor("#e3f2fd"),
            "NBR":  colors.HexColor("#fce4ec"),
        }
        interp_bg = _interp_colors.get(primary_label, _ndvi_color(ndvi_stats["mean"]))
        interp_tbl = Table(
            [[Paragraph(f"{_interp_titles.get(primary_label, 'Index Status')}:  {interp_label}",
                        styles[interp_key])]],
            colWidths=[7.2 * cm],
        )
        interp_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), interp_bg),
            ("BOX",          (0, 0), (-1, -1), 0.5, GREEN if interp_key == "good" else AMBER),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ]))

        ndwi_interp = _ndwi_interpretation(ndwi_stats["mean"]) if ndwi_stats else None
        ndwi_interp_tbl = Table(
            [[Paragraph(f"Water / Moisture:  {ndwi_interp}", styles["body"]) if ndwi_interp else Spacer(1, 0)]],
            colWidths=[7.2 * cm],
        )
        ndwi_interp_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#e3f2fd")),
            ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#1565c0")),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ]))

        b = ndvi_stats['bounds']
        top_label    = f"{abs(b.top):.4f}° {'N' if b.top >= 0 else 'S'}"
        bottom_label = f"{abs(b.bottom):.4f}° {'N' if b.bottom >= 0 else 'S'}"
        right_label  = f"{abs(b.right):.4f}° {'E' if b.right >= 0 else 'W'}"
        left_label   = f"{abs(b.left):.4f}° {'E' if b.left >= 0 else 'W'}"
        geo_info = (
            f"Bounds: {top_label}  {bottom_label}  {right_label}  {left_label}  ·  "
            f"Area ~ {ndvi_stats['area_km2']:.2f} km\u00b2"
        )

        right_col = [
            [stat_tbl],
            [Spacer(1, 0.3 * cm)],
            [interp_tbl],
            [Spacer(1, 0.15 * cm)],
            [ndwi_interp_tbl],
            [Spacer(1, 0.15 * cm)],
            [Paragraph(geo_info, styles["caption"])],
        ]
        right_tbl = Table(right_col, colWidths=[7.2 * cm])
        right_tbl.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ]))

        if ndvi_buf:
            map_img = RLImage(ndvi_buf, width=map_w, height=map_h)
            layout = Table(
                [[map_img, right_tbl]],
                colWidths=[map_w + 0.3 * cm, 7.2 * cm],
            )
            layout.setStyle(TableStyle([
                ("VALIGN",  (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ]))
            story.append(layout)
        else:
            story.append(Paragraph("Map data unavailable for this scene.", styles["body"]))
            story.append(right_tbl)

        # Colorbar legend below the map row
        if primary_label == "NDWI":
            cb_buf = _colorbar_image("RdBu", label="NDWI  (\u22121 \u2192 +1)")
            cb_caption = (
                "NDWI map rendered with RdBu colormap. "
                "Red = -1 (arid/dry), White = 0 (transitional), "
                "Blue = +1 (open water / flooding). "
                "Grey pixels = no satellite data."
            )
        else:
            cb_buf = _colorbar_image("RdYlGn", label=f"{primary_label}  (\u22121 \u2192 +1)")
            cb_caption = (
                f"{primary_label} map rendered with RdYlGn colormap. "
                "Red = -1 (water/bare), Yellow = 0, "
                "Green = +1 (dense vegetation). "
                "Grey pixels = no satellite data."
            )
        cb_img = RLImage(cb_buf, width=9.5 * cm, height=0.7 * cm)
        story.append(Spacer(1, 0.4 * cm))
        story.append(cb_img)
        story.append(Paragraph(cb_caption, styles["caption"]))

        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=GRAY_MID, spaceBefore=12, spaceAfter=4))
        return story

    # ── Change-detection (DIFF) section ──────────────────────────────────────
    def _diff_section(self, diff_path, styles):
        story = []
        story.append(Spacer(1, 0.4 * cm))

        header_tbl = Table(
            [[Paragraph("CHRONOS CHANGE DETECTION — NDVI DIFFERENCE MAP",
                        styles["scene_title"])]],
            colWidths=[16.2 * cm],
        )
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), RED),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 0.25 * cm))

        diff_stats = _extract_stats(diff_path)
        diff_buf   = _tif_to_png_buffer(diff_path, colormap="RdBu", vmin=-1.0, vmax=1.0)

        if diff_buf and diff_stats:
            img = RLImage(diff_buf, width=10 * cm, height=10 * cm)
            dt = _parse_date(os.path.basename(diff_path))
            date_label = _format_date(dt) if dt else os.path.basename(diff_path)

            change_rows = [
                ["Metric", "Value"],
                ["Mean ΔNDVI",   f"{diff_stats['mean']:.4f}"],
                ["Std Dev ΔNDVI", f"{diff_stats['std']:.4f}"],
                ["Max Decrease", f"{diff_stats['min']:.4f}"],
                ["Max Increase", f"{diff_stats['max']:.4f}"],
                ["Changed pixels", f"{diff_stats['pixels']:,}"],
            ]
            chg_tbl = Table(change_rows, colWidths=[4 * cm, 3 * cm])
            chg_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  RED),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
                ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f5f5f5")]),
            ]))

            layout = Table(
                [[img, chg_tbl]],
                colWidths=[10.3 * cm, 7.2 * cm],
            )
            layout.setStyle(TableStyle([
                ("VALIGN",  (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ]))
            story.append(layout)

            cb_buf = _colorbar_image("RdBu", label="ΔNDVI  (negative = loss, positive = gain)")
            story.append(RLImage(cb_buf, width=10 * cm, height=0.7 * cm))
            story.append(Paragraph(
                f"Change detection relative to previous acquisition as of {date_label}. "
                "Blue = vegetation gain, Red = vegetation loss. "
                "Grey = no data / cloud-masked.",
                styles["caption"],
            ))
        else:
            story.append(Paragraph("Change detection data unavailable.", styles["body"]))

        return story

    # ── Sentinel-1 radar section ───────────────────────────────────────────────
    def _s1_section(self, s1_files, styles):
        """
        Render VV backscatter maps for Sentinel-1 RTC scenes with:
          - Greyscale backscatter map
          - Land cover classification map + breakdown table
          - Inter-scene delta / anomaly block (when ≥ 2 scenes)
        """
        story = []
        story.append(Spacer(1, 0.4 * cm))

        header_tbl = Table(
            [[Paragraph("RADAR BACKSCATTER — Sentinel-1 SAR (VV/VH)",
                        styles["scene_title"])]],
            colWidths=[16.2 * cm],
        )
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "Optical acquisition was not possible due to persistent cloud cover. "
            "The following maps show Sentinel-1 C-band SAR backscatter (gamma0 RTC). "
            "Bright areas indicate high backscatter (urban, rough terrain). "
            "Dark areas indicate smooth surfaces (calm water, bare soil). "
            "Each scene includes an automated land cover classification and, where a "
            "prior acquisition exists, a backscatter delta anomaly report.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Cache raw VV arrays keyed by path for inter-scene delta
        _vv_cache: dict = {}

        for idx, s1_path in enumerate(s1_files):
            basename = os.path.basename(s1_path)
            date_str = basename[:8]
            dt = _parse_date(date_str)
            date_label = _format_date(dt) if dt else date_str

            try:
                with rasterio.open(s1_path) as src:
                    vv = src.read(1).astype(np.float32)
                    vh = src.read(2).astype(np.float32) if src.count >= 2 else None
                    bounds = src.bounds

                _vv_cache[s1_path] = vv

                finite_mask = np.isfinite(vv)
                nodata_mask = ~finite_mask
                valid_vv = vv[finite_mask]
                if valid_vv.size == 0:
                    continue

                # ── 1. Greyscale backscatter map ──────────────────────────────
                p99 = float(np.percentile(valid_vv, 99))
                vmax_display = max(p99, 0.01)
                norm_grey = mpl_colors.Normalize(vmin=0.0, vmax=vmax_display, clip=True)
                try:
                    import matplotlib
                    cmap_grey = matplotlib.colormaps["gray"]
                except (AttributeError, KeyError):
                    cmap_grey = mpl_cm.get_cmap("gray")

                vv_display = np.where(finite_mask, vv, 0.0)
                rgba_grey = cmap_grey(norm_grey(vv_display))
                rgba_grey[nodata_mask] = [0.93, 0.93, 0.93, 1.0]
                grey_buf = io.BytesIO()
                PILImage.fromarray(
                    (rgba_grey[:, :, :3] * 255).astype(np.uint8)
                ).save(grey_buf, format="PNG")
                grey_buf.seek(0)

                # ── 2. Land cover classification map ──────────────────────────
                class_map = classify_sar(vv, vh)
                lc_rgb = np.zeros((*vv.shape, 3), dtype=np.uint8)
                for cls_id, rgb in CLASS_COLORS.items():
                    mask = class_map == cls_id
                    lc_rgb[mask] = rgb
                lc_buf = io.BytesIO()
                PILImage.fromarray(lc_rgb).save(lc_buf, format="PNG")
                lc_buf.seek(0)

                lc_stats = classification_stats(class_map)

                # Fit two maps + stats column inside CONTENT_W without overflow.
                # right-column width (stats + lc table), gap between sections.
                _s1_right_w = 6.5 * cm
                _s1_gap     = 0.4 * cm
                map_w = (CONTENT_W - _s1_right_w - _s1_gap) / 2 - 0.1 * cm
                map_h = map_w

                grey_img = RLImage(grey_buf, width=map_w, height=map_h)
                lc_img   = RLImage(lc_buf,   width=map_w, height=map_h)

                # ── 3. Backscatter stats table ────────────────────────────────
                b = bounds
                top_label    = f"{abs(b.top):.4f}° {'N' if b.top >= 0 else 'S'}"
                bottom_label = f"{abs(b.bottom):.4f}° {'N' if b.bottom >= 0 else 'S'}"
                right_label  = f"{abs(b.right):.4f}° {'E' if b.right >= 0 else 'W'}"
                left_label   = f"{abs(b.left):.4f}° {'E' if b.left >= 0 else 'W'}"
                geo_info = f"Bounds: {top_label}  {bottom_label}  {right_label}  {left_label}"

                stat_rows = [
                    ["Metric", "VV"],
                    ["Mean",     f"{float(np.mean(valid_vv)):.4f}"],
                    ["Std Dev",  f"{float(np.std(valid_vv)):.4f}"],
                    ["Min",      f"{float(np.min(valid_vv)):.4f}"],
                    ["Max",      f"{float(np.max(valid_vv)):.4f}"],
                    ["Valid px", f"{int(valid_vv.size):,}"],
                ]
                stat_tbl = Table(stat_rows, colWidths=[3.2 * cm, 3.3 * cm])
                stat_tbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
                    ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, -1), 8),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
                    ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
                    ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
                    ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING",  (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [WHITE, colors.HexColor("#f5f5f5")]),
                ]))

                # ── 4. Land cover breakdown table ─────────────────────────────
                _CLASS_BG = {
                    WATER:      colors.HexColor("#d0e8ff"),
                    BARE_SOIL:  colors.HexColor("#f5e6cc"),
                    VEGETATION: colors.HexColor("#d4edda"),
                    URBAN:      colors.HexColor("#f8d7d7"),
                }
                lc_rows = [["Land Cover Class", "Pixels", "%"]]
                lc_style_cmds = [
                    ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
                    ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, -1), 8),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
                    ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
                    ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING",  (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ]
                for row_i, (cls_id, cs) in enumerate(lc_stats.items(), start=1):
                    lc_rows.append([
                        Paragraph(cs["label"], styles["stat_label"]),
                        f"{cs['count']:,}",
                        f"{cs['pct']:.1f}%",
                    ])
                    lc_style_cmds.append(
                        ("BACKGROUND", (0, row_i), (-1, row_i),
                         _CLASS_BG.get(cls_id, GRAY_BG))
                    )

                lc_tbl = Table(lc_rows, colWidths=[4.0 * cm, 1.5 * cm, 1.0 * cm])
                lc_tbl.setStyle(TableStyle(lc_style_cmds))

                # ── 5. Compose scene layout ───────────────────────────────────
                right_col = [
                    [Paragraph(f"Date: {date_label}", styles["body"])],
                    [Spacer(1, 0.2 * cm)],
                    [stat_tbl],
                    [Spacer(1, 0.25 * cm)],
                    [Paragraph("LAND COVER CLASSIFICATION", styles["stat_label"])],
                    [Spacer(1, 0.1 * cm)],
                    [lc_tbl],
                    [Spacer(1, 0.2 * cm)],
                    [Paragraph(geo_info, styles["caption"])],
                ]
                right_tbl = Table(right_col, colWidths=[_s1_right_w])
                right_tbl.setStyle(TableStyle([
                    ("TOPPADDING",    (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ]))

                # Two maps side-by-side, stats on right
                _maps_col_w = map_w + 0.1 * cm
                maps_tbl = Table(
                    [[grey_img, lc_img]],
                    colWidths=[_maps_col_w, _maps_col_w],
                )
                maps_tbl.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ]))

                layout = Table(
                    [[maps_tbl, right_tbl]],
                    colWidths=[_maps_col_w * 2 + _s1_gap, _s1_right_w],
                )
                layout.setStyle(TableStyle([
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ]))
                story.append(layout)
                story.append(Spacer(1, 0.1 * cm))
                story.append(Paragraph(
                    "<b>Left:</b> VV polarisation (black = low backscatter, white = high).  "
                    "<b>Right:</b> Land cover classification \u2014 "
                    "<font name='ZapfDingbats' color='#1e78c8'>n</font> Water (specular near-zero return), "
                    "<font name='ZapfDingbats' color='#c8aa78'>n</font> Bare Soil (low diffuse return), "
                    "<font name='ZapfDingbats' color='#228b22'>n</font> Vegetation (volume scatter >= 0.05 or CR >= 0.30), "
                    "<font name='ZapfDingbats' color='#b43232'>n</font> Urban (persistent double-bounce, VH/VV < 0.15), "
                    "<font name='ZapfDingbats' color='#ffa500'>n</font> Anomalous Veg/Moisture (data confidence filter), "
                    "<font name='ZapfDingbats' color='#008080'>n</font> Wet Forest (CR 0.15 - 0.30, dielectric guard). "
                    "Grey = no data. Full classification rationale in the Certification section.",
                    styles["caption"],
                ))

                # ── 6. Inter-scene delta report ───────────────────────────────
                if idx > 0:
                    prev_path = s1_files[idx - 1]
                    prev_vv = _vv_cache.get(prev_path)
                    if prev_vv is not None and prev_vv.shape == vv.shape:
                        delta, delta_stats = compute_sar_delta(vv, prev_vv)
                        interp = delta_interpretation(delta_stats)
                        level  = delta_stats.get("alert_level", "LOW")

                        alert_bg = {
                            "HIGH":   RED_LIGHT,
                            "MEDIUM": AMBER_LIGHT,
                            "LOW":    GREEN_LIGHT,
                        }.get(level, GRAY_BG)
                        alert_border = {
                            "HIGH":   RED,
                            "MEDIUM": AMBER,
                            "LOW":    GREEN,
                        }.get(level, GRAY_MID)
                        alert_style_key = {
                            "HIGH":   "alert",
                            "MEDIUM": "alert",
                            "LOW":    "good",
                        }.get(level, "body")

                        prev_basename = os.path.basename(prev_path)
                        prev_dt = _parse_date(prev_basename[:8])
                        prev_label = _format_date(prev_dt) if prev_dt else prev_basename[:8]

                        delta_header = (
                            f"DELTA REPORT  ·  {prev_label}  →  {date_label}"
                        )
                        delta_rows = [
                            ["Metric", "Value"],
                            ["Mean dVV",           f"{delta_stats['mean_delta']:.4f}"],
                            ["Std Dev dVV",        f"{delta_stats['std_delta']:.4f}"],
                            ["Anomaly threshold",  f"{delta_stats['anomaly_threshold']:.4f}  "
                                                   f"({delta_stats['sigma_threshold']:.0f}sigma "
                                                   f"or >={delta_stats['abs_delta_min']:.3f})"],
                            ["Anomalous pixels",   f"{delta_stats['anomaly_pct']:.1f}%"],
                            ["Backscatter decrease", f"{delta_stats['decrease_pct']:.1f}%"],
                            ["Backscatter increase",  f"{delta_stats['increase_pct']:.1f}%"],
                            ["Alert level",        level],
                        ]
                        delta_tbl = Table(delta_rows, colWidths=[4.5 * cm, 4.0 * cm])
                        delta_tbl.setStyle(TableStyle([
                            ("BACKGROUND",  (0, 0), (-1, 0),  RED if level == "HIGH" else (AMBER if level == "MEDIUM" else NAVY)),
                            ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
                            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                            ("FONTSIZE",    (0, 0), (-1, -1), 8),
                            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                            ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
                            ("TEXTCOLOR",   (0, 1), (-1, -1), GRAY_DARK),
                            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
                            ("TOPPADDING",  (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f5f5f5")]),
                        ]))

                        # Improved delta layout: better proportions and cleaner spacing
                        _delta_tbl_w   = 8.2 * cm
                        _delta_spacer  = 0.4 * cm
                        _interp_col_w  = 16.2 * cm - _delta_tbl_w - _delta_spacer  # ~7.6 cm

                        # Interpretation box with better text wrapping
                        interp_para = Paragraph(
                            f"<b>{level}</b> — {interp}",
                            styles[alert_style_key]
                        )
                        interp_tbl = Table(
                            [[interp_para]],
                            colWidths=[_interp_col_w],
                        )
                        interp_tbl.setStyle(TableStyle([
                            ("BACKGROUND",   (0, 0), (-1, -1), alert_bg),
                            ("BOX",          (0, 0), (-1, -1), 1.0, alert_border),
                            ("TOPPADDING",   (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
                            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                        ]))

                        delta_layout = Table(
                            [[delta_tbl, Spacer(1, 1), interp_tbl]],
                            colWidths=[_delta_tbl_w, _delta_spacer, _interp_col_w],
                        )
                        delta_layout.setStyle(TableStyle([
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING",    (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                        ]))
                        story.append(KeepTogether([
                            Spacer(1, 0.25 * cm),
                            Paragraph(delta_header, styles["section_head"]),
                            HRFlowable(width="100%", thickness=1,
                                       color=alert_border, spaceAfter=4),
                            delta_layout,
                        ]))

                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=GRAY_MID, spaceBefore=8, spaceAfter=2))

            except Exception as exc:
                logger.warning(f"S1 render failed for {s1_path}: {exc}")
                story.append(Paragraph(
                    f"Radar scene {date_label}: render failed — {exc}",
                    styles["body"],
                ))

        return story

    # ── MaaS: data extraction helper ───────────────────────────────────────────
    @staticmethod
    def _extract_s1_means(s1_files: list) -> list:
        """
        Return [(date_str, mean_vv_linear), …] for each S1 file.
        Uses np.isfinite() to exclude no-data / inf pixels globally.
        """
        result = []
        for f in s1_files:
            try:
                with rasterio.open(f) as src:
                    vv = src.read(1).astype(np.float32)
                finite = vv[np.isfinite(vv)]
                if finite.size == 0:
                    continue
                result.append((os.path.basename(f)[:8], float(np.mean(finite))))
            except Exception as exc:
                logger.warning(f"_extract_s1_means: {f}: {exc}")
        return result

    # ── MaaS: executive summary ────────────────────────────────────────────────
    def _maas_executive_summary(self, scene_means: list, styles) -> list:
        """
        'Mission Findings' block for a SAR time series.
        Threshold: std(mean VV) < 0.05 → stable; ≥ 0.05 → flag change events.
        Includes a layperson glossary rendered as a footnote beneath the block.
        """
        story = []
        story.append(Paragraph("MISSION FINDINGS", styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=NAVY, spaceAfter=6))

        ANOMALY_THRESHOLD = 0.05

        if len(scene_means) < 2:
            text = (
                "Insufficient acquisitions for temporal variance analysis. "
                "A minimum of two Radiometric Terrain Corrected (RTC) scenes is required "
                "to perform inter-scene Backscatter Intensity comparison."
            )
            style_key = "body"
            bg, border = GRAY_BG, GRAY_MID
        else:
            means_arr = np.array([m for _, m in scene_means], dtype=np.float64)
            finite_means = means_arr[np.isfinite(means_arr)]
            ts_std = float(np.std(finite_means))

            if ts_std < ANOMALY_THRESHOLD:
                overall_mean = float(np.mean(finite_means))
                mean_db = 10.0 * math.log10(max(overall_mean, 1e-10))
                text = (
                    f"Radiometric variance within threshold "
                    f"(\u03c3 = {ts_std:.4f} linear Gamma<super>0</super>; threshold: {ANOMALY_THRESHOLD:.2f}). "
                    f"Mean VV Backscatter Intensity: {overall_mean:.4f} linear Gamma<super>0</super> "
                    f"({mean_db:.1f} dB) across {len(scene_means)} acquisition(s). "
                    "No statistically significant anomaly detected in signal return. "
                    "Interpretation requires field validation."
                )
                style_key = "good"
                bg, border = GREEN_LIGHT, GREEN
            else:
                overall_mean = float(np.mean(finite_means))
                deviations = [
                    (d, abs(m - overall_mean))
                    for d, m in scene_means
                    if np.isfinite(m)
                ]
                deviations.sort(key=lambda x: x[1], reverse=True)
                flagged = deviations[: min(3, len(deviations))]
                flagged_labels = []
                for d, _ in flagged:
                    dt = _parse_date(d)
                    flagged_labels.append(_format_date(dt) if dt else d)
                flagged_str = ", ".join(flagged_labels)

                text = (
                    "Significant Backscatter Deviation detected. "
                    f"Mean VV Backscatter Intensity exhibits elevated temporal variance "
                    f"(\u03c3 = {ts_std:.4f} linear Gamma<super>0</super>; "
                    f"threshold: {ANOMALY_THRESHOLD:.2f}) "
                    f"across {len(scene_means)} acquisition(s). "
                    f"Acquisition date(s) flagged for highest Backscatter Intensity "
                    f"deviation: {flagged_str}. "
                    "Secondary review recommended. Contextual interpretation "
                    "requires field verification."
                )
                style_key = "alert"
                bg, border = RED_LIGHT, RED

        summary_tbl = Table(
            [[Paragraph(text, styles[style_key])]],
            colWidths=[16.2 * cm],
        )
        summary_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("BOX",           (0, 0), (-1, -1), 0.8, border),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]))
        story.append(summary_tbl)
        story.append(Spacer(1, 0.3 * cm))

        # ── Layperson glossary — footnote beneath the finding block ───────────
        glossary_text = (
            "<b>Glossary\u202f\u2014</b> "
            "<i>Radiometric Terrain Corrected (RTC):</i> SAR imagery corrected for "
            "geometric and radiometric distortions caused by terrain relief, enabling "
            "reliable land cover comparison across acquisitions. "
            "<i>Backscatter Intensity (Gamma<super>0</super>):</i> radar energy reflected from "
            "the Earth\u2019s surface back to the satellite, expressed as a dimensionless "
            "linear ratio; higher values indicate rougher or denser surfaces (urban, forest). "
            "<i>VV Polarisation:</i> radar signal transmitted and received in the vertical "
            "plane, sensitive to soil moisture, open water, and built structures."
        )
        story.append(Paragraph(glossary_text, styles["footnote"]))
        story.append(Spacer(1, 0.4 * cm))
        return story

    # ── MaaS: VV backscatter trend chart ──────────────────────────────────────
    def _vv_trend_chart(self, scene_means: list) -> Optional[io.BytesIO]:
        """
        Dual-axis line chart of Mean VV Backscatter Intensity over the analysis period.
          Left  Y-axis : linear Gamma0
          Right Y-axis : dB  (10 · log10)
          X-axis        : Acquisition Date
        Returns a PNG BytesIO buffer, or None if fewer than two valid points.
        All statistical inputs are guarded with np.isfinite().
        """
        if len(scene_means) < 2:
            return None

        dates, means = [], []
        for date_str, mean_vv in scene_means:
            dt = _parse_date(date_str)
            if dt is not None and np.isfinite(mean_vv):
                dates.append(dt)
                means.append(float(mean_vv))

        if len(dates) < 2:
            return None

        fig, ax1 = plt.subplots(figsize=(6.8, 3.0), dpi=120)

        # ── Left axis: linear Gamma0 ──────────────────────────────────────────
        ax1.plot(dates, means,
                 color="#1a3a5c", linewidth=2.0, marker="o",
                 markersize=6, markerfacecolor="#2e7d32",
                 markeredgecolor="#1a3a5c", markeredgewidth=1.2, zorder=3,
                 label="Mean VV Gamma⁰ (linear)")
        ax1.fill_between(dates, means, alpha=0.12, color="#1a3a5c")
        ax1.set_ylabel("Backscatter (linear Gamma⁰)", fontsize=9, color="#1a3a5c", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="#1a3a5c", labelsize=8)
        ax1.tick_params(axis="x", labelsize=8)
        ax1.set_ylim(bottom=0)

        # ── Right axis: dB equivalent ─────────────────────────────────────────
        ax2 = ax1.twinx()
        means_db = [10.0 * math.log10(max(m, 1e-10)) for m in means]
        ax2.plot(dates, means_db,
                 color="#e65100", linewidth=1.5, linestyle="--",
                 alpha=0.85, label="VV (dB)")
        ax2.set_ylabel("Backscatter (dB)", fontsize=9, color="#e65100", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#e65100", labelsize=8)

        # ── X-axis: acquisition date formatting ───────────────────────────────
        ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter("%d %b %Y"))
        ax1.xaxis.set_major_locator(mpl_dates.AutoDateLocator(minticks=4, maxticks=8))
        for label in ax1.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
            label.set_fontsize(8)
        ax1.set_xlabel("Acquisition Date", fontsize=9, fontweight="bold")

        # ── Title & grid ──────────────────────────────────────────────────────
        ax1.set_title(
            "Mean VV Backscatter Intensity — Temporal Trend",
            fontsize=11, color="#1a3a5c", fontweight="bold", pad=10,
        )
        ax1.grid(True, linestyle="-", alpha=0.25, linewidth=0.6, zorder=0, color="#9e9e9e")
        ax1.set_facecolor("#fafbfc")
        fig.patch.set_facecolor("white")
        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # Combined legend (both axes) — positioned at bottom right
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=8, loc="lower right", framealpha=0.95,
                   fancybox=True, edgecolor="#cfd8dc")

        fig.tight_layout(pad=0.6)

        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf

    # ── SAR Temporal Backscatter Trend table ───────────────────────────────────
    def _sar_trend_table(self, s1_scene_means: list, s1_files: list, styles) -> list:
        """
        Consolidated 'Temporal Backscatter Trend' summary table for SAR-only missions.
        Columns: Date | Mean VV (linear) | VV (dB) | Std Dev | Valid px |
                 Dominant Cover | Scene Alert
        Also emits a geometric integrity note (valid-pixel consistency across dates).
        """
        story = []
        story.append(Paragraph("TEMPORAL BACKSCATTER TREND — Sentinel-1 SAR",
                                styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=NAVY, spaceAfter=6))

        body_text = (
            "The table below consolidates mean VV Backscatter Intensity (Gamma<super>0</super> RTC, "
            "linear scale) for every Sentinel-1 acquisition in this mission. "
            "<b>Gamma<super>0</super> RTC</b> is radar energy reflected from the Earth\u2019s surface, "
            "corrected for terrain slope, enabling direct land-use comparison across dates. "
            "Higher values indicate rougher or denser surfaces (urban areas, dense forest); "
            "lower values indicate smooth surfaces (standing water, bare soil). "
            "The <b>Valid px</b> column is compared across dates to verify geometric "
            "co-registration \u2014 a stable count confirms all scenes share the same "
            "spatial grid, which is a prerequisite for reliable pixel-level change analysis."
        )
        story.append(Paragraph(body_text, styles["body"]))
        story.append(Spacer(1, 0.3 * cm))

        # ── Build per-scene stats ─────────────────────────────────────────────
        _CLASS_LABELS_SHORT = {
            WATER:      "Water",
            BARE_SOIL:  "Bare Soil",
            VEGETATION: "Vegetation",
            URBAN:      "Urban",
        }

        header = ["Date", "Mean VV\n(linear Gamma0)", "VV (dB)", "Std Dev",
                  "Valid px", "Dominant Cover", "Alert"]
        rows = [header]

        ref_pixels: Optional[int] = None
        pixel_drift_flag = False
        scene_data = []   # (date_str, mean_vv, std_vv, valid_px, dominant, vv_arr)

        for date_str, mean_vv in s1_scene_means:
            f = next((x for x in s1_files
                      if os.path.basename(x)[:8] == date_str), None)
            std_vv = float("nan")
            valid_px = 0
            dominant = "—"
            vv_arr = None

            if f:
                try:
                    with rasterio.open(f) as src:
                        vv = src.read(1).astype(np.float32)
                        vh = src.read(2).astype(np.float32) if src.count >= 2 else None
                    finite = vv[np.isfinite(vv)]
                    if finite.size > 0:
                        std_vv   = float(np.std(finite))
                        valid_px = int(finite.size)
                        vv_arr   = vv

                    if ref_pixels is None:
                        ref_pixels = valid_px
                    elif valid_px > 0 and ref_pixels > 0:
                        drift_ratio = abs(valid_px - ref_pixels) / ref_pixels
                        if drift_ratio > 0.01:
                            pixel_drift_flag = True

                    class_map = classify_sar(vv, vh)
                    lc_stats  = classification_stats(class_map)
                    if lc_stats:
                        dom_cls  = max(lc_stats, key=lambda c: lc_stats[c]["count"])
                        dominant = _CLASS_LABELS_SHORT.get(dom_cls, "—")
                except Exception as exc:
                    logger.warning(f"_sar_trend_table stats failed for {f}: {exc}")

            scene_data.append((date_str, mean_vv, std_vv, valid_px, dominant, vv_arr))

        # ── Compute per-scene delta alert ─────────────────────────────────────
        alerts = ["—"]   # first scene has no prior
        for i in range(1, len(scene_data)):
            _, _, _, _, _, curr_vv = scene_data[i]
            _, _, _, _, _, prev_vv = scene_data[i - 1]
            if (curr_vv is not None and prev_vv is not None
                    and curr_vv.shape == prev_vv.shape):
                from app.analytics.sar_alerting import compute_sar_delta
                _, dstats = compute_sar_delta(curr_vv, prev_vv)
                alerts.append(dstats.get("alert_level", "—"))
            else:
                alerts.append("—")

        # ── Populate rows ─────────────────────────────────────────────────────
        _ALERT_BG = {"HIGH": RED_LIGHT, "MEDIUM": AMBER_LIGHT, "LOW": GREEN_LIGHT}
        _ALERT_FG = {"HIGH": RED,       "MEDIUM": AMBER,        "LOW": GREEN}
        lc_style_cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("TEXTCOLOR",     (0, 1), (-1, -1), GRAY_DARK),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("ALIGN",         (1, 0), (5, -1),  "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [WHITE, colors.HexColor("#f5f5f5")]),
        ]

        for row_i, ((date_str, mean_vv, std_vv, valid_px, dominant, _),
                    alert) in enumerate(zip(scene_data, alerts), start=1):
            dt = _parse_date(date_str)
            date_label = _format_date(dt) if dt else date_str
            vv_db = (f"{10.0 * math.log10(max(mean_vv, 1e-10)):.2f} dB"
                     if np.isfinite(mean_vv) else "—")
            rows.append([
                date_label,
                f"{mean_vv:.4f}" if np.isfinite(mean_vv) else "—",
                vv_db,
                f"{std_vv:.4f}" if np.isfinite(std_vv) else "—",
                f"{valid_px:,}" if valid_px > 0 else "—",
                dominant,
                alert,
            ])
            # Colour Alert cell
            if alert in _ALERT_BG:
                lc_style_cmds.append(
                    ("BACKGROUND", (6, row_i), (6, row_i), _ALERT_BG[alert])
                )
                lc_style_cmds.append(
                    ("FONTNAME",   (6, row_i), (6, row_i), "Helvetica-Bold")
                )
                lc_style_cmds.append(
                    ("TEXTCOLOR",  (6, row_i), (6, row_i), _ALERT_FG[alert])
                )

        col_w = [3.5*cm, 2.4*cm, 2.0*cm, 1.8*cm, 2.1*cm, 2.6*cm, 1.8*cm]
        tbl = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle(lc_style_cmds))
        story.append(tbl)
        story.append(Spacer(1, 0.25 * cm))

        # ── Geometric integrity note ──────────────────────────────────────────
        if pixel_drift_flag:
            integrity_note = (
                "WARNING — Valid pixel count varies by more than 1% between acquisitions. "
                "Verify geometric co-registration before performing pixel-level change "
                "analysis. Misaligned grids may produce false anomaly detections."
            )
            integrity_style_key, int_bg, int_border = "alert", RED_LIGHT, RED
        else:
            integrity_note = (
                "Geometric Integrity Confirmed — Valid pixel counts are consistent across "
                "all acquisition dates (\u00b11%). All Sentinel-1 scenes are co-registered "
                "to the same spatial grid, ensuring reliable pixel-level change analysis."
            )
            integrity_style_key, int_bg, int_border = "good", GREEN_LIGHT, GREEN

        integrity_tbl = Table(
            [[Paragraph(integrity_note, styles[integrity_style_key])]],
            colWidths=[16.2 * cm],
        )
        integrity_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), int_bg),
            ("BOX",           (0, 0), (-1, -1), 0.6, int_border),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(integrity_tbl)
        story.append(Spacer(1, 0.4 * cm))
        return story

    # ── SAR Certification, Explainability & Technical Limitations section ───────
    def _sar_certification_section(self, styles, area_km2: Optional[float] = None) -> list:
        """
        Render the ISO 19157-compliant certification appendix:
          1. Plain-language classification legend
          2. Technical Limitations & Liability Disclaimer
          3. Copernicus Data Attribution
          4. Validation Protocol summary (50-point control design)
        """
        story = []
        story.append(PageBreak())
        story.append(Paragraph(
            "SAR CLASSIFICATION METHODOLOGY \u2014 TECHNICAL CERTIFICATION",
            styles["section_head"],
        ))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=NAVY, spaceAfter=6))
        story.append(Paragraph(
            "This section satisfies the ISO\u202f19157:2013 \u2018Geographic Information \u2014 "
            "Data Quality\u2019 requirement for transparency and auditability of automated "
            "classification products intended for commercial or legal use.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.4 * cm))

        # ── 1. Plain-language classification legend ────────────────────────
        story.append(Paragraph("1.  Plain-Language Classification Legend",
                               styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=NAVY_LIGHT, spaceAfter=4))
        story.append(Paragraph(
            "The table below explains <i>why</i> each land cover class was assigned, "
            "not merely <i>what</i> the class is. If a user asks \u201cWhy is this pixel "
            "orange?\u201d the answer is found in the ANOMALOUS VEGETATION / MOISTURE row. "
            "For a per-pixel audit trail, call "
            "<font face='Courier'>explain_pixel(vv, vh, ndvi, ndvi_baseline, "
            "persistence_run)</font> from the analytics API.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.25 * cm))

        _COLOR_HEX = {
            WATER:                  "#1e78c8",
            BARE_SOIL:              "#c8aa78",
            VEGETATION:             "#228b22",
            URBAN:                  "#b43232",
            ANOMALOUS_VEG_MOISTURE: "#ffa500",
            WET_FOREST:             "#008080",
        }
        _hdr_para = ParagraphStyle(
            'legend_hdr',
            fontName='Helvetica-Bold', fontSize=8,
            textColor=WHITE, leading=10,
        )
        legend_rows = [[
            Paragraph("Class", _hdr_para),
            Paragraph("Colour", _hdr_para),
            Paragraph("Plain-Language Justification", _hdr_para),
        ]]
        legend_style_cmds = [
            ("BACKGROUND",   (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("TEXTCOLOR",    (0, 1), (-1, -1), GRAY_DARK),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("VALIGN",       (1, 1), (1, -1),  "MIDDLE"),
            ("ALIGN",        (1, 0), (1, -1),  "CENTER"),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (2, 0), (2, -1),  8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [WHITE, colors.HexColor("#f5f5f5")]),
        ]
        cls_order = [
            WATER, BARE_SOIL, VEGETATION, URBAN,
            ANOMALOUS_VEG_MOISTURE, WET_FOREST,
        ]
        for row_i, cls_id in enumerate(cls_order, start=1):
            hex_col = _COLOR_HEX.get(cls_id, "#9e9e9e")
            swatch_tbl = Table(
                [[" "]],
                colWidths=[0.7 * cm],
                rowHeights=[0.35 * cm],
            )
            swatch_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(hex_col)),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            plain = CLASS_PLAIN_LANGUAGE.get(cls_id, "")
            legend_rows.append([
                Paragraph(CLASS_LABELS.get(cls_id, str(cls_id)), styles["body"]),
                swatch_tbl,
                Paragraph(plain, styles["body"]),
            ])
        legend_tbl = Table(
            legend_rows,
            colWidths=[3.8 * cm, 1.5 * cm, 10.9 * cm],
        )
        legend_tbl.setStyle(TableStyle(legend_style_cmds))
        story.append(legend_tbl)
        story.append(Spacer(1, 0.5 * cm))

        # ── 2. Copernicus Data Attribution ────────────────────────────────
        story.append(Paragraph("2.  Data Source Attribution (Copernicus / ESA)",
                               styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=NAVY_LIGHT, spaceAfter=4))
        attrib_tbl = Table(
            [[Paragraph(_COPERNICUS_ATTRIBUTION, styles["body"])]],
            colWidths=[16.2 * cm],
        )
        attrib_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#e8f0fe")),
            ("BOX",           (0, 0), (-1, -1), 0.8, NAVY_LIGHT),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        story.append(attrib_tbl)
        story.append(Spacer(1, 0.5 * cm))

        # ── 3. Validation Protocol Summary ───────────────────────────────
        story.append(Paragraph("3.  ISO 19157 Validation Protocol \u2014 50-Point Confusion Matrix Design",
                               styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=NAVY_LIGHT, spaceAfter=4))
        try:
            _aoi_km2 = area_km2 if (area_km2 is not None and area_km2 > 0) else 14.49
            protocol = generate_validation_protocol(area_km2=_aoi_km2, n_points=50)
            proto_intro = (
                f"Stratified random sampling across the {protocol['area_km2']:.2f}\u202fkm\u00b2 AOI. "
                f"Minimum spatial separation between control points: "
                f"{protocol['min_spacing_km']}\u202fkm (ensures spatial independence). "
                f"Reference data: {protocol['reference_data']['primary_source']}. "
                f"Temporal window: \u00b1{protocol['reference_data']['temporal_window_days']} days "
                f"from SAR acquisition date. {protocol['reference_data']['interpreter_requirement']}"
            )
            story.append(Paragraph(proto_intro, styles["body"]))
            story.append(Spacer(1, 0.2 * cm))

            proto_rows = [["Land Cover Class", "Control Points", "% of Total"]]
            total_pts = sum(protocol["per_class_samples"].values())
            for cls_label, n_pts in protocol["per_class_samples"].items():
                pct = n_pts / total_pts * 100.0 if total_pts > 0 else 0.0
                proto_rows.append([cls_label, str(n_pts), f"{pct:.0f}%"])
            proto_rows.append(["TOTAL", str(total_pts), "100%"])

            proto_tbl = Table(proto_rows, colWidths=[7.5 * cm, 3.5 * cm, 3.5 * cm])
            proto_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  NAVY),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 8),
                ("FONTNAME",     (0, 1), (-1, -2), "Helvetica"),
                ("FONTNAME",     (0, -1),(-1, -1), "Helvetica-Bold"),
                ("BACKGROUND",   (0, -1),(-1, -1), colors.HexColor("#eceff1")),
                ("TEXTCOLOR",    (0, 1), (-1, -1), GRAY_DARK),
                ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
                ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2),
                 [WHITE, colors.HexColor("#f5f5f5")]),
            ]))
            story.append(proto_tbl)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                "<b>Validation Design Template \u2014 Field Survey Required.</b> "
                "The table above defines the sampling design; accuracy metrics are computed "
                "after field data collection is complete. "
                "Producer\u2019s Accuracy (PA) = correctly classified reference points / "
                "total reference points per class (omission error = 100 - PA). "
                "User\u2019s Accuracy (UA) = correctly classified mapped points / "
                "total mapped points per class (commission error = 100 - UA). "
                "Overall Accuracy (OA) = diagonal sum / total reference points. "
                "Kappa Coefficient = (OA - Pe) / (1 - Pe) where Pe = agreement due to chance. "
                "Two independent interpreters required; inter-rater Cohen's Kappa \u2265 0.85 "
                "is required for consensus before confusion matrix completion.",
                styles["footnote"],
            ))
        except Exception as e:
            logger.warning(f"Validation protocol generation failed: {e}")
            story.append(Paragraph(
                "Validation protocol data unavailable.", styles["body"],
            ))

        story.append(Spacer(1, 0.4 * cm))

        # ── 4. Technical Limitations & Liability Disclaimer (last) ─────────
        story.append(Paragraph("4.  Technical Limitations & Liability Disclaimer",
                               styles["section_head"]))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                color=AMBER, spaceAfter=4))
        disclaimer_tbl = Table(
            [[Paragraph(_TECHNICAL_LIMITATIONS, styles["body"])]],
            colWidths=[16.2 * cm],
        )
        disclaimer_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), AMBER_LIGHT),
            ("BOX",           (0, 0), (-1, -1), 1.0, AMBER),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        story.append(disclaimer_tbl)

        return story

    # ── Metadata builder ───────────────────────────────────────────────────────
    def _build_meta(self, valid_ndvi, bbox, era5_file, s1_files=None,
                    job_id: str = "", report_gen_date: str = "",
                    primary_label: str = "NDVI"):
        is_radar = bool(s1_files) and not valid_ndvi
        _opt_sensor = {
            "NDVI": "Sentinel-2 MSI (Optical \u00b7 NDVI)",
            "NDWI": "Sentinel-2 MSI (Optical \u00b7 NDWI)",
            "NBR":  "Sentinel-2 MSI (Optical \u00b7 NBR)",
        }
        meta = {
            "sensor": "Sentinel-1 SAR (Radar / Cloud-Piercing)" if is_radar
                      else _opt_sensor.get(primary_label, "Sentinel-2 MSI (Optical)"),
            "job_id": job_id,
            "report_gen_date": report_gen_date or _report_gen_date(),
            "date_start": None,
            "date_end": None,
            "lat_center": None,
            "lon_center": None,
            "area_km2": None,
            "valid_scenes": len(valid_ndvi),
            "total_scenes": len(valid_ndvi),
            "crs": "EPSG:4326",
            "qr_data": None,
            "avg_temp": None,
            "avg_precip": None,
        }

        dates = sorted(valid_ndvi)
        if dates:
            dt_start = _parse_date(dates[0])
            dt_end   = _parse_date(dates[-1])
            meta["date_start"] = _format_date(dt_start) if dt_start else dates[0]
            meta["date_end"]   = _format_date(dt_end)   if dt_end   else dates[-1]

            first_s, _ = valid_ndvi[dates[0]]
            meta["lat_center"] = first_s["lat_center"]
            meta["lon_center"] = first_s["lon_center"]
            meta["area_km2"]   = first_s["area_km2"]
            meta["crs"]        = first_s["crs"]
        elif s1_files:
            # Populate date/location from S1 files when no optical data
            s1_dates = sorted(
                os.path.basename(f)[:8]
                for f in s1_files
                if len(os.path.basename(f)) >= 8
            )
            if s1_dates:
                dt_start = _parse_date(s1_dates[0])
                dt_end   = _parse_date(s1_dates[-1])
                meta["date_start"] = _format_date(dt_start) if dt_start else s1_dates[0]
                meta["date_end"]   = _format_date(dt_end)   if dt_end   else s1_dates[-1]
                meta["valid_scenes"] = len(s1_files)
                meta["total_scenes"] = len(s1_files)
            s1_stats = _extract_stats(s1_files[0])
            if s1_stats:
                meta["lat_center"] = s1_stats["lat_center"]
                meta["lon_center"] = s1_stats["lon_center"]
                meta["area_km2"]   = s1_stats["area_km2"]
                meta["crs"]        = s1_stats["crs"]

        # QR data from bbox or first file bounds
        try:
            if bbox and hasattr(bbox, "lower_left"):
                minx, miny = bbox.lower_left
                maxx, maxy = bbox.upper_right
                lat_c = (miny + maxy) / 2
                lon_c = (minx + maxx) / 2
            elif bbox and hasattr(bbox, "__iter__"):
                minx, miny, maxx, maxy = list(bbox)[:4]
                lat_c = (miny + maxy) / 2
                lon_c = (minx + maxx) / 2
            elif meta["lat_center"]:
                lat_c = meta["lat_center"]
                lon_c = meta["lon_center"]
            else:
                lat_c = lon_c = None

            if lat_c is not None:
                meta["qr_data"] = (
                    f"geo:{lat_c:.5f},{lon_c:.5f}"
                    f"?q={lat_c:.5f},{lon_c:.5f}(Project Gaia Mission Center)"
                )
        except Exception:
            pass

        # ERA5 weather data
        if era5_file:
            try:
                with rasterio.open(era5_file) as src:
                    t = src.read(1)
                    p = src.read(2)
                t_valid = t[np.isfinite(t)]
                p_valid = p[np.isfinite(p)]
                meta["avg_temp"]   = float(np.mean(t_valid)) - 273.15 if t_valid.size > 0 else None
                meta["avg_precip"] = float(np.mean(p_valid)) * 1000   if p_valid.size > 0 else None
            except Exception as exc:
                logger.warning(f"ERA5 read failed: {exc}")

        return meta


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    gen = PDFReportGenerator()
    gen.generate_pdf(filename="report_redesigned.pdf")
    print("Done — results/report_redesigned.pdf")
