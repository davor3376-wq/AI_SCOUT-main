import os
import io
import glob
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, FastAPI, HTTPException, Query, Response, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sentinelhub import SentinelHubCatalog, BBox, CRS, DataCollection
from app.ingestion.auth import SentinelHubAuth
import mercantile
import rasterio
import rasterio.errors
import rasterio.windows
import rasterio.warp
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import numpy as np
from PIL import Image

import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import (
    SCOUT_DB_PATH, ALLOWED_ORIGINS, RESULTS_DIR, STAC_DIR, PROCESSED_DATA_DIR,
    ENVIRONMENT, SECRET_KEY, BILLING_MODE,
)
from app.core.database import get_connection
from app.api.job_manager import JobManager
from app.api.audit import AuditLogger
from app.api.cleanup import run_cleanup
from app.api.dependencies import get_current_user, require_role
from app.auth.manager import UserManager
from app.auth.router import router as auth_router
from app.api.billing_router import router as billing_router
from app.core.mission_executor import process_mission_task
from app.core.scheduler import MissionScheduler
from app.api.usage_controller import (
    UsageController,
    RegionalLicenseViolation,
    DailyQuotaExhausted,
    InsufficientCredits,
)
from app.core.logging_config import setup_logging
from app.ingestion.retry import retry_call

setup_logging()
logger = logging.getLogger("main")

DB_PATH = SCOUT_DB_PATH

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

scheduler = MissionScheduler()
audit = AuditLogger(db_path=DB_PATH)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    if ENVIRONMENT == "production" and not SECRET_KEY:
        raise RuntimeError(
            "SCOUT_SECRET_KEY must be set in production. "
            "Refusing to start to prevent open-auth exposure."
        )

    admin_user = os.environ.get("SCOUT_ADMIN_USERNAME", "admin")
    admin_pass = os.environ.get("SCOUT_ADMIN_PASSWORD", "")
    if SECRET_KEY and admin_pass:
        um = UserManager(db_path=DB_PATH)
        if um.count_users() == 0:
            try:
                um.create_user(admin_user, admin_pass, role="admin")
                logger.info(f"Bootstrap: created admin user '{admin_user}'")
            except Exception as exc:
                logger.warning(f"Bootstrap admin creation failed: {exc}")

    asyncio.create_task(scheduler.start())
    logger.info(f"AI SCOUT started [env={ENVIRONMENT}].")

    yield

    # --- shutdown ---
    scheduler.stop()


app = FastAPI(
    title="AI SCOUT",
    version="1.0.0",
    description="Evidence-grade environmental monitoring platform.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_origins_list = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

if _origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ---------------------------------------------------------------------------
# Middleware — request correlation ID + security headers + HTTPS redirect
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Propagate or generate a per-request correlation ID."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Attach security headers; redirect HTTP → HTTPS in production (via proxy header)."""
    if (
        ENVIRONMENT == "production"
        and request.headers.get("X-Forwarded-Proto", "https") == "http"
    ):
        https_url = str(request.url).replace("http://", "https://", 1)
        return Response(status_code=301, headers={"Location": https_url})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
    return response


# ---------------------------------------------------------------------------
# Routers (versioned under /v1)
# ---------------------------------------------------------------------------
app.include_router(auth_router, prefix="/v1")
if BILLING_MODE != "disabled":
    app.include_router(billing_router, prefix="/v1")


# ---------------------------------------------------------------------------
# Versioned API sub-router — all non-health endpoints live under /v1
# ---------------------------------------------------------------------------
v1_router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class MissionRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    sensor: str
    recurrence: Optional[str] = None

    @field_validator("geometry")
    @classmethod
    def validate_geometry(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        geom_type = v.get("type")
        if geom_type not in ("Point", "Polygon"):
            raise ValueError("geometry.type must be 'Point' or 'Polygon'")
        coords = v.get("coordinates")
        if coords is None:
            raise ValueError("geometry.coordinates is required")

        def _check_coord(lon: float, lat: float) -> None:
            if not (-180 <= lon <= 180):
                raise ValueError(f"longitude {lon} out of range [-180, 180]")
            if not (-90 <= lat <= 90):
                raise ValueError(f"latitude {lat} out of range [-90, 90]")

        if geom_type == "Point":
            if not isinstance(coords, list) or len(coords) < 2:
                raise ValueError("Point coordinates must be [lon, lat]")
            _check_coord(coords[0], coords[1])
        elif geom_type == "Polygon":
            if not isinstance(coords, list) or not coords:
                raise ValueError("Polygon coordinates must be [[ring]]")
            for ring in coords:
                for pt in ring:
                    if not isinstance(pt, list) or len(pt) < 2:
                        raise ValueError("Each polygon coordinate must be [lon, lat]")
                    _check_coord(pt[0], pt[1])
        return v

    @field_validator("sensor")
    @classmethod
    def validate_sensor(cls, v: str) -> str:
        allowed = {"OPTICAL", "RADAR", "NDVI", "NDWI", "NBR", "S1_RTC"}
        if v.upper() not in allowed:
            raise ValueError(f"sensor must be one of {allowed}")
        return v.upper()

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("date must be ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        return v


# ---------------------------------------------------------------------------
# Helper: client IP from request
# ---------------------------------------------------------------------------
def _ip(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"


# ---------------------------------------------------------------------------
# Routes — system
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health():
    db_status = "ok"
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_status = "error"
    # Always return 200 — Railway/load-balancers use this to check liveness,
    # not DB readiness. DB failures are visible in the body for alerting.
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok" if db_status == "ok" else "degraded",
            "db": db_status,
            "environment": ENVIRONMENT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Routes — missions
# ---------------------------------------------------------------------------
@v1_router.post("/launch_custom_mission", tags=["missions"])
@limiter.limit("10/minute")
async def launch_mission(
    request: Request,
    body: MissionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_role("analyst", "admin")),
):
    auth_obj = SentinelHubAuth()
    catalog = SentinelHubCatalog(config=auth_obj.config)

    try:
        geom_type = body.geometry.get("type")
        if geom_type == "Point":
            lon, lat = body.geometry["coordinates"]
            delta = 0.05
            bbox = BBox(bbox=[lon - delta, lat - delta, lon + delta, lat + delta], crs=CRS.WGS84)
        elif geom_type == "Polygon":
            coords = body.geometry["coordinates"][0]
            lons, lats = [c[0] for c in coords], [c[1] for c in coords]
            bbox = BBox(bbox=[min(lons), min(lats), max(lons), max(lats)], crs=CRS.WGS84)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported geometry: {geom_type}")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid geometry structure")

    time_interval = (body.start_date, body.end_date)
    sensor = body.sensor
    tag = None
    results: List[Dict] = []
    search_results: Dict[str, Any] = {}

    if sensor in ("OPTICAL", "NDVI", "NDWI", "NBR"):
        s2_col = DataCollection.SENTINEL2_L2A.define_from(
            "CDSE_S2_L2A", service_url="https://sh.dataspace.copernicus.eu"
        )
        try:
            items = await asyncio.to_thread(
                lambda: list(retry_call(
                    catalog.search,
                    collection=s2_col, bbox=bbox, time=time_interval,
                ))
            )
        except Exception as exc:
            logger.warning(f"S2 catalog search failed after retries: {exc}")
            items = []

        if items:
            items = [x for x in items if x is not None]
            items.sort(key=lambda x: x["properties"].get("datetime", ""), reverse=True)
            latest = items[0]
            if latest["properties"].get("eo:cloud_cover", 100) > 80:
                sensor, tag = "RADAR", "CLOUD_PIERCED"
            else:
                for item in items:
                    assets = item.get("assets") or {}
                    preview = (
                        (assets.get("thumbnail") or {}).get("href")
                        or (assets.get("visual") or {}).get("href")
                    )
                    results.append({
                        "tile_id": item["id"], "date": item["properties"].get("datetime"),
                        "preview_url": preview, "bbox": item.get("bbox"),
                    })
                search_results = {
                    "tile_id": latest["id"],
                    "preview_url": results[0]["preview_url"] if results else None,
                    "results": results, "bbox": latest.get("bbox"),
                }
        else:
            sensor, tag = "RADAR", "NO_OPTICAL_DATA"

    if sensor in ("RADAR", "S1_RTC"):
        try:
            items = await asyncio.to_thread(
                lambda: list(retry_call(
                    catalog.search,
                    collection=DataCollection.SENTINEL1_IW, bbox=bbox,
                    time=time_interval,
                ))
            )
        except Exception as exc:
            logger.warning(f"S1 catalog search failed after retries: {exc}")
            items = []

        if items:
            items = [x for x in items if x is not None]
            items.sort(key=lambda x: x["properties"].get("datetime", ""), reverse=True)
            for item in items:
                preview = ((item.get("assets") or {}).get("thumbnail") or {}).get("href")
                results.append({
                    "tile_id": item["id"], "date": item["properties"].get("datetime"),
                    "preview_url": preview, "bbox": item.get("bbox"),
                })
            search_results = {
                "tile_id": items[0]["id"],
                "preview_url": results[0]["preview_url"] if results else None,
                "tag": tag, "results": results, "bbox": items[0].get("bbox"),
            }
        else:
            logger.warning(f"No radar data for bbox={bbox} time={time_interval}")

    if not search_results and not results:
        audit.log(
            "launch_mission", username=current_user.get("username"),
            details=f"no data for sensor={sensor}", ip_address=_ip(request), status="error"
        )
        raise HTTPException(status_code=404, detail="No data found for the given criteria.")

    # ------------------------------------------------------------------
    # Governance pre-flight (sensor is finalised by this point)
    # ------------------------------------------------------------------
    _username = current_user.get("username", "unknown")
    _uc = UsageController(db_path=DB_PATH)
    # Ensure a profile row exists for this user (idempotent)
    _uc.ensure_profile(current_user.get("id", _username), _username)

    # 1. Regional License check (HTTP 403)
    try:
        _uc.validate_region(_username, bbox)
    except RegionalLicenseViolation as _exc:
        audit.log(
            "launch_mission", username=_username,
            details=str(_exc), ip_address=_ip(request), status="denied",
        )
        raise HTTPException(status_code=403, detail=f"Regional License Violation: {_exc}")

    # Geometry metrics used by both cap and cost checks
    _bbox_list_preflight = [
        bbox.lower_left[0], bbox.lower_left[1],
        bbox.upper_right[0], bbox.upper_right[1],
    ]
    _area_km2 = _uc.calc_bbox_area_km2(*_bbox_list_preflight)

    # 2. Daily Surface Cap check (HTTP 429)
    try:
        _uc.check_daily_surface_cap(_username, _area_km2)
    except DailyQuotaExhausted as _exc:
        audit.log(
            "launch_mission", username=_username,
            details=str(_exc), ip_address=_ip(request), status="denied",
        )
        raise HTTPException(status_code=429, detail=f"Daily Quota Exhausted: {_exc}")

    # 3. Credit sufficiency check (HTTP 402) — read-only, no funds moved yet
    try:
        _start_dt = datetime.fromisoformat(body.start_date)
        _end_dt   = datetime.fromisoformat(body.end_date)
    except ValueError:
        _start_dt = _end_dt = datetime.now(timezone.utc)
    _temporal_depth = max(1, (_end_dt - _start_dt).days + 1)
    _mission_cost   = _uc.calculate_mission_cost(_area_km2, sensor, _temporal_depth)

    try:
        _uc.verify_credits(_username, _mission_cost)
    except InsufficientCredits as _exc:
        audit.log(
            "launch_mission", username=_username,
            details=str(_exc), ip_address=_ip(request), status="denied",
        )
        raise HTTPException(status_code=402, detail=f"Payment Required: {_exc}")
    # ------------------------------------------------------------------

    jm = JobManager(db_path=DB_PATH)
    bbox_list = [bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1]]
    job_id = jm.create_job({
        "sensor": sensor,
        "start_date": body.start_date,
        "end_date": body.end_date,
        "bbox": bbox_list,
        "recurrence": body.recurrence,
        "preview_url": search_results.get("preview_url"),
        "search_results": search_results,
        "tag": tag,
        "launched_by": current_user.get("username"),
    })

    background_tasks.add_task(process_mission_task, job_id, bbox, time_interval, sensor)

    audit.log(
        "launch_mission", username=current_user.get("username"),
        resource=job_id,
        details=f"sensor={sensor} bbox={bbox_list}",
        ip_address=_ip(request),
    )

    response = search_results.copy()
    response["job_id"] = job_id
    response["status"] = "PENDING"
    return response


# ---------------------------------------------------------------------------
# Routes — jobs
# ---------------------------------------------------------------------------
@v1_router.get("/jobs", tags=["jobs"])
@limiter.limit("60/minute")
async def list_jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    _user: Dict = Depends(require_role("viewer", "analyst", "admin")),
):
    jm = JobManager(db_path=DB_PATH)
    jobs = jm.list_jobs()
    if status:
        jobs = [j for j in jobs if j.get("status") == status.upper()]
    return {"total": len(jobs), "offset": offset, "limit": limit, "jobs": jobs[offset: offset + limit]}


@v1_router.get("/jobs/{job_id}", tags=["jobs"])
async def get_job(
    job_id: str,
    _user: Dict = Depends(require_role("viewer", "analyst", "admin")),
):
    jm = JobManager(db_path=DB_PATH)
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@v1_router.delete("/jobs/{job_id}", tags=["jobs"])
async def delete_job(
    request: Request,
    job_id: str,
    current_user: Dict = Depends(require_role("analyst", "admin")),
):
    jm = JobManager(db_path=DB_PATH)
    if not jm.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    jm.delete_job(job_id)
    audit.log(
        "delete_job", username=current_user.get("username"),
        resource=job_id, ip_address=_ip(request),
    )
    return {"deleted": job_id}


@v1_router.get("/jobs/{job_id}/pdf", tags=["jobs"])
async def download_pdf(
    request: Request,
    job_id: str,
    current_user: Dict = Depends(require_role("viewer", "analyst", "admin")),
):
    jm = JobManager(db_path=DB_PATH)
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job not complete")
    pdf_path = job.get("results", {}).get("pdf_report")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found")

    audit.log(
        "download_pdf", username=current_user.get("username"),
        resource=job_id, ip_address=_ip(request),
    )
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))


# ---------------------------------------------------------------------------
# Routes — STAC
# ---------------------------------------------------------------------------
@v1_router.get("/stac", tags=["stac"])
async def get_stac_catalog():
    path = os.path.join(STAC_DIR, "catalog.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="STAC catalog not found")
    return FileResponse(path, media_type="application/json")


@v1_router.get("/stac/{item_id}", tags=["stac"])
async def get_stac_item(item_id: str):
    safe_id = os.path.basename(item_id)
    path = os.path.join(STAC_DIR, safe_id, f"{safe_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="STAC item not found")
    return FileResponse(path, media_type="application/json")


# ---------------------------------------------------------------------------
# Routes — tiles
# ---------------------------------------------------------------------------
@v1_router.get("/tiles/{z}/{x}/{y}", tags=["tiles"])
async def get_tile(
    z: int, x: int, y: int,
    file: Optional[str] = None,
    job_id: Optional[str] = None,
):
    if not (0 <= z <= 28):
        raise HTTPException(status_code=400, detail="z must be 0-28")
    max_xy = 2 ** z
    if not (0 <= x < max_xy and 0 <= y < max_xy):
        raise HTTPException(status_code=400, detail=f"x,y must be in [0, {max_xy})")

    tif_path: Optional[str] = None

    if job_id:
        jm = JobManager(db_path=DB_PATH)
        job = jm.get_job(job_id)
        if job and job.get("status") == "COMPLETED":
            processed = job.get("results", {}).get("processed_files", [])
            if processed:
                ndvi = [p for p in processed if "NDVI" in p]
                tif_path = ndvi[0] if ndvi else processed[0]

    if not tif_path and file:
        safe = os.path.basename(file)
        for base in (RESULTS_DIR, PROCESSED_DATA_DIR):
            candidate = os.path.join(base, safe)
            if os.path.exists(candidate):
                tif_path = candidate
                break

    if not tif_path and not job_id and not file:
        candidates: List[str] = []
        for pattern in (os.path.join(RESULTS_DIR, "*.tif"), os.path.join(PROCESSED_DATA_DIR, "*.tif")):
            candidates.extend(glob.glob(pattern))
        if candidates:
            tif_path = max(candidates, key=os.path.getmtime)

    def _transparent() -> Response:
        buf = io.BytesIO()
        Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    if not tif_path:
        return _transparent()

    try:
        mtime = os.path.getmtime(tif_path)
        png = _render_tile_cached(tif_path, z, x, y, mtime)
        if png is None:
            return _transparent()
        return Response(content=png, media_type="image/png")
    except Exception as exc:
        logger.error(f"Tile render error z={z} x={x} y={y}: {exc}", exc_info=True)
        return _transparent()


@lru_cache(maxsize=512)
def _render_tile_cached(tif_path: str, z: int, x: int, y: int, _mtime: float) -> Optional[bytes]:
    import matplotlib.pyplot as plt  # lazy: avoids 3-6 s startup cost; cached after first call
    bounds = mercantile.xy_bounds(x, y, z)
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    dst_crs = "EPSG:3857"
    dst_transform = from_bounds(left, bottom, right, top, 256, 256)

    with rasterio.open(tif_path) as src:
        src_bounds = rasterio.warp.transform_bounds(
            dst_crs, src.crs.to_string(), left, bottom, right, top
        )
        try:
            window = src.window(*src_bounds).intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
        except (rasterio.errors.WindowError, Exception):
            return None
        if window.width <= 0 or window.height <= 0:
            return None

        data = np.full((src.count, 256, 256), np.nan, dtype=np.float32)
        for i in range(src.count):
            reproject(
                source=src.read(i + 1, window=window, out_dtype=np.float32),
                destination=data[i],
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    if np.all(np.isnan(data)):
        return None

    basename = os.path.basename(tif_path).upper()

    if data.shape[0] == 1:
        band = data[0]
        if "NDWI" in basename:
            cmap = plt.get_cmap("Blues")
            norm = plt.Normalize(vmin=-0.3, vmax=0.8)
        elif "NBR" in basename:
            cmap = plt.get_cmap("RdYlGn")
            norm = plt.Normalize(vmin=-0.5, vmax=0.8)
        else:
            cmap = plt.get_cmap("RdYlGn")
            norm = plt.Normalize(vmin=-0.2, vmax=1.0)
        colored = cmap(norm(band))
        colored[np.isnan(band)] = [0, 0, 0, 0]
        img = Image.fromarray((colored * 255).astype(np.uint8), "RGBA")
    elif data.shape[0] == 2:
        vv = data[0]
        no_data = np.isnan(vv) | (vv <= 0)
        safe_vv = np.where(~no_data, vv, 1.0)
        vv_db = np.where(~no_data, 10.0 * np.log10(safe_vv), np.nan)
        colored = plt.get_cmap("gray")(plt.Normalize(vmin=-25.0, vmax=0.0)(vv_db))
        colored[no_data] = [0, 0, 0, 0]
        img = Image.fromarray((colored * 255).astype(np.uint8), "RGBA")
    elif data.shape[0] >= 3:
        rgb = np.moveaxis(data[:3], 0, -1)
        max_val = np.nanmax(rgb)
        if max_val is not None and max_val <= 1.5:
            rgb = np.clip(rgb, 0.0, 1.0) * 255
        rgb = np.nan_to_num(rgb).astype(np.uint8)
        alpha = np.full((256, 256), 255, dtype=np.uint8)
        alpha[np.all(rgb == 0, axis=2)] = 0
        img = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
    else:
        return None

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Routes — admin
# ---------------------------------------------------------------------------
@v1_router.get("/admin/audit-logs", tags=["admin"])
async def get_audit_logs(
    username: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    _admin: Dict = Depends(require_role("admin")),
):
    return audit.query(username=username, action=action, limit=limit, offset=offset)


@v1_router.delete("/admin/cleanup", tags=["admin"])
async def trigger_cleanup(
    retention_days: int = Query(default=int(os.environ.get("DATA_RETENTION_DAYS", "90")), ge=1),
    current_user: Dict = Depends(require_role("admin")),
    request: Request = None,
):
    result = run_cleanup(retention_days=retention_days)
    audit.log(
        "cleanup", username=current_user.get("username"),
        details=f"retention_days={retention_days} deleted={result['deleted_count']}",
        ip_address=_ip(request) if request else None,
    )
    return result


@v1_router.get("/admin/users", tags=["admin"])
async def list_users_admin(
    _admin: Dict = Depends(require_role("admin")),
):
    return UserManager(db_path=DB_PATH).list_users()


# ---------------------------------------------------------------------------
# Routes — emergency wipe (NGO panic button)
# ---------------------------------------------------------------------------
class WipeRequest(BaseModel):
    confirmation_phrase: str


@v1_router.delete("/admin/emergency-wipe", tags=["admin"])
async def emergency_wipe(
    body: WipeRequest,
    request: Request,
    current_user: Dict = Depends(require_role("admin")),
):
    """
    Irreversibly delete all job data, audit logs, usage records, and result
    files, then rotate every API key.  Designed for field use when a device
    is at risk of seizure.

    Requires:  { "confirmation_phrase": "WIPE ALL DATA" }
    """
    if body.confirmation_phrase != "WIPE ALL DATA":
        raise HTTPException(
            status_code=400,
            detail="Send confirmation_phrase='WIPE ALL DATA' to confirm the irreversible wipe.",
        )

    deleted_files: List[str] = []
    file_errors: List[str] = []

    wipe_patterns = [
        os.path.join(RESULTS_DIR, "**", "*"),
        os.path.join(PROCESSED_DATA_DIR, "**", "*"),
        os.path.join(SCOUT_DB_PATH.replace("jobs.db", ""), "data", "raw", "**", "*"),
    ]
    import glob as _glob
    for pattern in wipe_patterns:
        for filepath in _glob.glob(pattern, recursive=True):
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    deleted_files.append(filepath)
                except Exception as exc:
                    file_errors.append(f"{filepath}: {exc}")

    with get_connection() as conn:
        for table in ("jobs", "audit_logs", "usage_log",
                      "daily_surface_log", "credit_ledger"):
            try:
                conn.execute(f"DELETE FROM {table}")
            except Exception:
                pass

    keys_rotated = UserManager(db_path=DB_PATH).rotate_all_api_keys()

    audit.log(
        "emergency_wipe",
        username=current_user.get("username"),
        details=(
            f"files_deleted={len(deleted_files)} "
            f"file_errors={len(file_errors)} "
            f"keys_rotated={keys_rotated}"
        ),
        ip_address=_ip(request),
        status="ok",
    )

    return {
        "status": "wiped",
        "files_deleted": len(deleted_files),
        "file_errors": len(file_errors),
        "keys_rotated": keys_rotated,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operator": current_user.get("username"),
    }


# Register the versioned router (must come before the SPA wildcard mount)
app.include_router(v1_router)

# Must be last — catches all unmatched paths for the SPA
_platform_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "platform")
)
if os.path.isdir(_platform_dir):
    app.mount("/", StaticFiles(directory=_platform_dir, html=True), name="platform")
else:
    logger.warning(f"SPA not served — platform directory not found: {_platform_dir}")
