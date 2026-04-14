"""
Pay-as-you-go billing via Stripe Checkout.

Flow:
  1. Client calls POST /billing/checkout {"package_id": "starter", "country_code": "BRA"}
  2. We create a Stripe Checkout session with metadata {username, credits, country_code}
  3. User pays on Stripe's hosted page
  4. Stripe fires POST /billing/webhook (checkout.session.completed)
  5. We verify the signature, call topup_credits() + set_geofence() with country polygon

Country registry is built at startup from the Natural Earth dataset bundled with
geopandas (already a project dependency). Country codes are ISO 3166-1 alpha-3.

Endpoints:
  GET  /billing/packages          available credit packages
  GET  /billing/countries         searchable list of all licensable countries
  GET  /billing/countries/{code}  single country details + GeoJSON
  GET  /billing/balance           credit balance + licensed country + daily surface used
  GET  /billing/history           ledger entries for the authenticated user
  POST /billing/checkout          create Stripe Checkout session → {checkout_url}
  POST /billing/webhook           Stripe webhook (no auth — verified by signature)
  POST /billing/topup             admin: manually add credits to any user
  PUT  /billing/region            admin: set country or custom polygon without payment
"""

import json
import os
import logging
import pathlib
import warnings

from app.core.config import BILLING_MODE
from typing import Dict, Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Header
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, require_role
from app.api.usage_controller import UsageController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
SCOUT_FRONTEND_URL    = os.environ.get("SCOUT_FRONTEND_URL", "http://localhost:3000")
DB_PATH               = os.environ.get("SCOUT_DB_PATH", "jobs.db")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


# ---------------------------------------------------------------------------
# Country registry — built once at import time from Natural Earth / geopandas
#
# geopandas ships with naturalearth_lowres (110 m resolution, ~250 countries).
# The public API changed across versions so we use a fallback chain:
#   1. gpd.datasets.get_path()      — geopandas < 0.14
#   2. Walk the package directory   — geopandas >= 0.14 (file still present)
#   3. Empty dict + warning         — admin must set regions manually via API
# ---------------------------------------------------------------------------

def _build_country_registry() -> dict[str, dict]:
    """
    Load country geometries from Natural Earth and index by ISO alpha-3 code.
    Returns an empty dict (with a logged warning) if loading fails.
    """
    try:
        import geopandas as gpd

        world = None

        # Attempt 1: old public API (geopandas < 0.14)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            except Exception:
                pass

        # Attempt 2: find the shapefile directly in the package tree
        if world is None:
            pkg_root = pathlib.Path(gpd.__file__).parent
            for candidate in pkg_root.rglob("naturalearth_lowres.shp"):
                try:
                    world = gpd.read_file(str(candidate))
                    break
                except Exception:
                    continue

        if world is None:
            raise RuntimeError("naturalearth_lowres shapefile not found in geopandas package.")

        registry: dict[str, dict] = {}
        for _, row in world.iterrows():
            iso  = str(row.get("iso_a3", "")).strip()
            name = str(row.get("name",   "")).strip()
            if not iso or iso == "-99" or not name:
                continue
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            registry[iso] = {
                "code":    iso,
                "name":    name,
                "geojson": geom.__geo_interface__,   # Polygon or MultiPolygon
            }

        logger.info(f"Country registry loaded: {len(registry)} countries from Natural Earth.")
        return registry

    except Exception as exc:
        logger.warning(
            f"Country registry could not be loaded from Natural Earth: {exc}. "
            "The /billing/countries endpoint will return an empty list. "
            "Admins can still set country geofences manually via PUT /billing/region."
        )
        return {}


_COUNTRY_REGISTRY: dict[str, dict] | None = None


def _get_registry() -> dict[str, dict]:
    """Load the country registry on first access (lazy — avoids blocking startup)."""
    global _COUNTRY_REGISTRY
    if _COUNTRY_REGISTRY is None:
        _COUNTRY_REGISTRY = _build_country_registry()
    return _COUNTRY_REGISTRY


# ---------------------------------------------------------------------------
# Credit packages
# ---------------------------------------------------------------------------
CREDIT_PACKAGES: list[dict] = [
    {
        "id":          "starter",
        "label":       "Starter",
        "credits":     1_000,
        "price_cents": 900,       # $9.00  → $0.009 / credit
        "description": "Good for ~3 optical scans of a 10 km² AOI over 30 days",
    },
    {
        "id":          "standard",
        "label":       "Standard",
        "credits":     6_000,
        "price_cents": 4_900,     # $49.00 → $0.0082 / credit
        "description": "Good for ~20 optical scans or ~8 radar scans",
    },
    {
        "id":          "professional",
        "label":       "Professional",
        "credits":     25_000,
        "price_cents": 17_900,    # $179.00 → $0.0072 / credit
        "description": "Unlimited experimentation; covers large-AOI radar time-series",
    },
]

_PACKAGES_BY_ID: dict[str, dict] = {p["id"]: p for p in CREDIT_PACKAGES}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/billing", tags=["billing"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CheckoutRequest(BaseModel):
    package_id: str = Field(..., description="One of: starter, standard, professional")
    country_code: Optional[str] = Field(
        None,
        description=(
            "ISO 3166-1 alpha-3 country code (e.g. BRA, DEU, USA). "
            "Required on first purchase to license a territory. "
            "Omit on subsequent top-ups to keep the existing licence."
        ),
    )


class AdminTopupRequest(BaseModel):
    username: str
    amount: float = Field(..., gt=0)
    note: Optional[str] = "admin top-up"


class AdminSetRegionRequest(BaseModel):
    username: str
    country_code: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-3 code. Takes precedence unless geojson is also set.",
    )
    geojson: Optional[dict] = Field(
        None,
        description=(
            "Custom GeoJSON Polygon, MultiPolygon, or Feature. "
            "Use for sub-national AOIs or countries missing from Natural Earth."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_country(code: str) -> dict:
    """Lookup a country by ISO alpha-3, case-insensitive. Raises 404 if missing."""
    entry = _get_registry().get(code.upper())
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Country code '{code}' not found. "
                "Use GET /billing/countries to browse available codes."
            ),
        )
    return entry


def _geofence_from_country(entry: dict) -> dict:
    """
    Build the GeoJSON Feature that gets stored in client_profiles.licensed_geofence.
    Wraps geometry in a Feature so validate_region() can extract .geometry cleanly.
    """
    return {
        "type":       "Feature",
        "properties": {"code": entry["code"], "name": entry["name"]},
        "geometry":   entry["geojson"],
    }


def _resolve_licensed_country(stored_geofence) -> Optional[dict]:
    """Return {code, name} from the stored geofence Feature properties, or None."""
    if not stored_geofence:
        return None
    raw = stored_geofence if isinstance(stored_geofence, dict) else json.loads(stored_geofence)
    props = raw.get("properties", {})
    code  = props.get("code")
    name  = props.get("name")
    if code and name:
        return {"code": code, "name": name}
    return {"code": "custom", "name": "Custom Region"}


# ---------------------------------------------------------------------------
# GET /billing/packages
# ---------------------------------------------------------------------------
@router.get("/packages")
async def list_packages():
    return {
        "packages": CREDIT_PACKAGES,
        "note": (
            "Credit cost formula: area_km2 × sensor_multiplier × temporal_days. "
            "Sentinel-2 multiplier=1.0, Sentinel-1 multiplier=2.5. Minimum 1 credit."
        ),
    }


# ---------------------------------------------------------------------------
# GET /billing/countries
# ---------------------------------------------------------------------------
@router.get("/countries")
async def list_countries(
    search: Optional[str] = Query(None, description="Filter by country name (case-insensitive)"),
):
    """
    Return all licensable countries from Natural Earth.
    Optionally filter by partial name match (e.g. ?search=bra → Brazil).
    GeoJSON geometries are excluded here to keep the list lightweight;
    use GET /billing/countries/{code} for the full polygon.
    """
    registry = _get_registry()
    countries = sorted(registry.values(), key=lambda x: x["name"])
    if search:
        q = search.lower()
        countries = [c for c in countries if q in c["name"].lower()]

    return {
        "count":     len(countries),
        "countries": [{"code": c["code"], "name": c["name"]} for c in countries],
    }


# ---------------------------------------------------------------------------
# GET /billing/countries/{code}
# ---------------------------------------------------------------------------
@router.get("/countries/{code}")
async def get_country(code: str):
    """Return a single country's details including its GeoJSON boundary polygon."""
    entry = _get_country(code)
    return {"code": entry["code"], "name": entry["name"], "geojson": entry["geojson"]}


# ---------------------------------------------------------------------------
# GET /billing/balance
# ---------------------------------------------------------------------------
@router.get("/balance")
async def get_balance(current_user: Dict = Depends(get_current_user)):
    username = current_user.get("username", "")
    uc = UsageController(db_path=DB_PATH)
    uc.ensure_profile(current_user.get("id", username), username)

    profile       = uc.get_profile(username) or {}
    balance       = uc.get_credit_balance(username)
    surface_today = uc.get_daily_surface_used(username)

    return {
        "username":                username,
        "credit_balance":          round(balance, 4),
        "licensed_country":        _resolve_licensed_country(profile.get("licensed_geofence")),
        "daily_surface_used_km2":  round(surface_today, 4),
        "daily_surface_cap_km2":   profile.get("daily_surface_cap_km2"),
    }


# ---------------------------------------------------------------------------
# GET /billing/history
# ---------------------------------------------------------------------------
@router.get("/history")
async def get_history(
    limit: int = 50,
    current_user: Dict = Depends(get_current_user),
):
    username = current_user.get("username", "")
    uc = UsageController(db_path=DB_PATH)
    return {
        "username": username,
        "ledger":   uc.get_ledger(username, limit=limit),
        "usage":    uc.get_usage_summary(username, limit=limit),
    }


# ---------------------------------------------------------------------------
# POST /billing/checkout
# ---------------------------------------------------------------------------
@router.post("/checkout")
async def create_checkout_session(
    body: CheckoutRequest,
    current_user: Dict = Depends(get_current_user),
):
    """
    Create a Stripe Checkout session.

    - First purchase: country_code required — no territory, no access.
    - Subsequent top-ups: omit country_code to keep existing licence.
    - Changing territory: include a different country_code; takes effect on payment.
    """
    if BILLING_MODE == "grant":
        raise HTTPException(
            status_code=403,
            detail=(
                "This instance uses grant-based access. "
                "Contact your administrator to have credits added to your account."
            ),
        )
    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=503,
            detail="Payment processing not configured (STRIPE_SECRET_KEY missing).",
        )

    package = _PACKAGES_BY_ID.get(body.package_id)
    if not package:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown package '{body.package_id}'. Options: {list(_PACKAGES_BY_ID)}",
        )

    username = current_user.get("username", "")
    uc = UsageController(db_path=DB_PATH)
    uc.ensure_profile(current_user.get("id", username), username)
    profile = uc.get_profile(username) or {}

    # Validate country code if provided
    country_entry = None
    if body.country_code:
        country_entry = _get_country(body.country_code)   # raises 404 if invalid

    # Require a country on first purchase
    if not profile.get("licensed_geofence") and not country_entry:
        raise HTTPException(
            status_code=400,
            detail=(
                "No licensed territory on file. "
                "Include country_code in your first purchase to unlock access. "
                "Use GET /billing/countries to find your ISO alpha-3 code."
            ),
        )

    # Build Stripe metadata
    metadata: dict[str, str] = {
        "username":   username,
        "credits":    str(package["credits"]),
        "package_id": package["id"],
    }
    if country_entry:
        metadata["country_code"] = country_entry["code"]

    # Product description surfaced on the Stripe receipt
    desc = package["description"]
    if country_entry:
        existing = _resolve_licensed_country(profile.get("licensed_geofence"))
        action = "Territory change" if existing else "Licensed territory"
        desc = f"{desc}. {action}: {country_entry['name']}."

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "quantity": 1,
                "price_data": {
                    "currency":     "usd",
                    "unit_amount":  package["price_cents"],
                    "product_data": {
                        "name": f"AI Scout — {package['label']} Credits",
                        "description": desc,
                    },
                },
            }],
            metadata=metadata,
            success_url=(
                f"{SCOUT_FRONTEND_URL}/billing/success"
                "?session_id={CHECKOUT_SESSION_ID}"
            ),
            cancel_url=f"{SCOUT_FRONTEND_URL}/billing/cancel",
        )
    except stripe.StripeError as exc:
        logger.error(f"Stripe checkout failed for {username}: {exc}")
        raise HTTPException(status_code=502, detail=f"Stripe error: {exc.user_message}")

    logger.info(
        f"Checkout created | user={username} package={package['id']} "
        f"country={country_entry['code'] if country_entry else 'unchanged'} "
        f"session={session.id}"
    )
    return {"checkout_url": session.url, "session_id": session.id}


# ---------------------------------------------------------------------------
# POST /billing/webhook  (no JWT — verified by Stripe signature)
# ---------------------------------------------------------------------------
@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook not configured.")

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, STRIPE_WEBHOOK_SECRET
        )
    except stripe.SignatureVerificationError:
        logger.warning("Stripe webhook signature verification failed.")
        raise HTTPException(status_code=400, detail="Invalid Stripe signature.")

    if event["type"] == "checkout.session.completed":
        session      = event["data"]["object"]
        meta         = session.get("metadata", {})
        username     = meta.get("username", "")
        credits      = float(meta.get("credits", 0))
        pkg_id       = meta.get("package_id", "unknown")
        country_code = meta.get("country_code")

        if not username or credits <= 0:
            logger.error(f"Webhook incomplete metadata: {meta}")
            return {"status": "ignored", "reason": "incomplete metadata"}

        uc = UsageController(db_path=DB_PATH)
        uc.ensure_profile(username, username)

        # 1. Top up credits
        uc.topup_credits(
            username=username,
            amount=credits,
            note=(
                f"Stripe | package={pkg_id} "
                f"country={country_code or 'unchanged'} "
                f"session={session['id']}"
            ),
        )

        # 2. Set geofence to the purchased country
        if country_code:
            entry = _get_registry().get(country_code.upper())
            if entry:
                uc.set_geofence(username, _geofence_from_country(entry))
                logger.info(f"Geofence set | user={username} country={country_code}")
            else:
                logger.warning(
                    f"country_code '{country_code}' in webhook metadata not in registry — "
                    "geofence not updated."
                )

        logger.info(
            f"Payment processed | user={username} credits={credits} "
            f"package={pkg_id} country={country_code or 'unchanged'}"
        )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /billing/topup  (admin only)
# ---------------------------------------------------------------------------
@router.post("/topup", tags=["admin"])
async def admin_topup(
    body: AdminTopupRequest,
    _admin: Dict = Depends(require_role("admin")),
):
    """Manually add credits to any user's account."""
    uc = UsageController(db_path=DB_PATH)
    if not uc.get_profile(body.username):
        raise HTTPException(status_code=404, detail=f"No profile for '{body.username}'.")
    uc.topup_credits(body.username, body.amount, note=body.note or "admin top-up")
    return {
        "username":    body.username,
        "added":       body.amount,
        "new_balance": uc.get_credit_balance(body.username),
    }


# ---------------------------------------------------------------------------
# PUT /billing/region  (admin only)
# ---------------------------------------------------------------------------
@router.put("/region", tags=["admin"])
async def admin_set_region(
    body: AdminSetRegionRequest,
    _admin: Dict = Depends(require_role("admin")),
):
    """
    Set or override the licensed territory for any user without payment.
    Use for enterprise contracts, corrections, or sub-national custom AOIs.

    Priority: geojson > country_code.
    """
    uc = UsageController(db_path=DB_PATH)
    if not uc.get_profile(body.username):
        raise HTTPException(status_code=404, detail=f"No profile for '{body.username}'.")

    if body.geojson:
        geofence = body.geojson
        label    = body.geojson.get("properties", {}).get("name", "custom polygon")
    elif body.country_code:
        entry    = _get_country(body.country_code)
        geofence = _geofence_from_country(entry)
        label    = entry["name"]
    else:
        raise HTTPException(status_code=400, detail="Provide country_code or geojson.")

    uc.set_geofence(body.username, geofence)
    logger.info(f"Admin set region | user={body.username} territory={label}")
    return {"username": body.username, "territory": label, "status": "updated"}
