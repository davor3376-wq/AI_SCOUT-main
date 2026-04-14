"""
Credit & Region Gatekeeper for Project Gaia.

Enforces four governance layers in order:
  1. Regional Licensing  — bbox vs. licensed GeoJSON geofence (HTTP 403)
  2. Daily Surface Cap   — km² consumed today vs. per-client quota (HTTP 429)
  3. Credit Verification — balance check before any data fetch (HTTP 402)
  4. Credit Lock         — pend → deduct (success) / release (failure) pattern

Usage flow (called by mission_executor.py):
  validate_region()          → raises RegionalLicenseViolation
  check_daily_surface_cap()  → raises DailyQuotaExhausted
  cost = calculate_mission_cost()
  verify_credits()           → raises InsufficientCredits
  ledger_id = pend_credits() → holds credits before S1/S2 fetch
  ... S1Client / S2Client download + PDF generation ...
  deduct_credits()           → finalise on PDF success
  OR release_credits()       → unwind on any failure
  log_usage()                → write immutable audit row
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from math import radians, cos
from typing import Optional

from shapely.geometry import shape, Point

from app.core.config import SCOUT_DB_PATH
from app.core.database import get_connection, init_schema, insert_returning_id, is_postgres, get_raw_connection

logger = logging.getLogger(__name__)

# Sensor credit multipliers
SENSOR_MULTIPLIERS: dict[str, float] = {
    "OPTICAL": 1.0,
    "NDVI":    1.0,
    "NDWI":    1.0,
    "NBR":     1.0,
    "RADAR":   2.5,
    "S1_RTC":  2.5,
}


# ---------------------------------------------------------------------------
# Custom exceptions (caught by the API layer and mapped to HTTP status codes)
# ---------------------------------------------------------------------------

class RegionalLicenseViolation(Exception):
    """HTTP 403 — request bbox falls outside the client's licensed geofence."""


class DailyQuotaExhausted(Exception):
    """HTTP 429 — client has consumed their daily km² surface allowance."""


class InsufficientCredits(Exception):
    """HTTP 402 — credit balance too low for this mission cost."""


# ---------------------------------------------------------------------------
# DB helpers — same WAL-SQLite pattern as the rest of the codebase
# ---------------------------------------------------------------------------

_USAGE_SCHEMA = """
    -- Per-client licensing & credit pool
    CREATE TABLE IF NOT EXISTS client_profiles (
        user_id               TEXT PRIMARY KEY,
        username              TEXT UNIQUE NOT NULL,
        credit_balance        REAL NOT NULL DEFAULT 0.0,
        daily_surface_cap_km2 REAL NOT NULL DEFAULT 10000.0,
        licensed_geofence     TEXT        -- GeoJSON Feature/Geometry, NULL = unrestricted
    );

    -- Immutable credit ledger (PEND / DEDUCT / RELEASE / TOPUP)
    CREATE TABLE IF NOT EXISTS credit_ledger (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    TEXT    NOT NULL,
        job_id     TEXT    NOT NULL,
        entry_type TEXT    NOT NULL,  -- PEND | DEDUCT | RELEASE | TOPUP
        amount     REAL    NOT NULL,  -- negative = debit, positive = credit
        note       TEXT,
        created_at TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ledger_user
        ON credit_ledger(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_ledger_job
        ON credit_ledger(job_id);

    -- Daily surface consumption (km2 per user per UTC day)
    CREATE TABLE IF NOT EXISTS daily_surface_log (
        user_id          TEXT NOT NULL,
        date             TEXT NOT NULL,  -- YYYY-MM-DD UTC
        surface_used_km2 REAL NOT NULL DEFAULT 0.0,
        PRIMARY KEY (user_id, date)
    );

    -- Immutable per-job usage audit
    CREATE TABLE IF NOT EXISTS usage_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id              TEXT    NOT NULL UNIQUE,
        client_id           TEXT    NOT NULL,
        username            TEXT    NOT NULL,
        sensor              TEXT    NOT NULL,
        bbox_area_km2       REAL    NOT NULL,
        temporal_depth_days INTEGER NOT NULL,
        credits_consumed    REAL    NOT NULL DEFAULT 0.0,
        timestamp           TEXT    NOT NULL,
        status              TEXT    NOT NULL DEFAULT 'PENDING'
    );
    CREATE INDEX IF NOT EXISTS idx_usage_client
        ON usage_log(client_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_usage_job
        ON usage_log(job_id);
"""


def _init_usage_db(db_path: str) -> None:
    init_schema(_USAGE_SCHEMA, db_path)


# ---------------------------------------------------------------------------
# UsageController
# ---------------------------------------------------------------------------

class UsageController:
    """
    Central enforcement point for regional licensing, credit ledger,
    daily surface throttling, and usage accountability.
    """

    def __init__(self, db_path: str = SCOUT_DB_PATH):
        self.db_path = db_path
        _init_usage_db(db_path)

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_profile(self, username: str) -> Optional[dict]:
        """Return the client_profiles row for *username*, or None."""
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM client_profiles WHERE username=?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def ensure_profile(self, user_id: str, username: str) -> dict:
        """
        Upsert a default profile so every user has a row.
        Call this at user-creation time (or lazily on first mission launch).
        """
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO client_profiles (user_id, username) VALUES (?,?) "
                "ON CONFLICT (user_id) DO NOTHING",
                (user_id, username),
            )
        return self.get_profile(username)  # type: ignore[return-value]

    def set_geofence(self, username: str, geofence_geojson: dict) -> None:
        """
        Set or replace the licensed GeoJSON geofence for *username*.
        Accepts a GeoJSON Feature or bare Geometry dict.
        """
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE client_profiles SET licensed_geofence=? WHERE username=?",
                (json.dumps(geofence_geojson), username),
            )

    def topup_credits(
        self, username: str, amount: float, note: str = "admin top-up"
    ) -> None:
        """Add *amount* credits to the balance and record a TOPUP ledger entry."""
        profile = self.get_profile(username)
        if not profile:
            raise ValueError(f"No client profile for '{username}'.")
        now = datetime.now(timezone.utc).isoformat()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE client_profiles SET credit_balance = credit_balance + ? "
                "WHERE username=?",
                (amount, username),
            )
            conn.execute(
                "INSERT INTO credit_ledger "
                "(user_id, job_id, entry_type, amount, note, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (profile["user_id"], "TOPUP", "TOPUP", amount, note, now),
            )
        logger.info(f"Topped up {amount:.2f} credits for '{username}'. Note: {note}")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calc_bbox_area_km2(
        lon_min: float, lat_min: float, lon_max: float, lat_max: float
    ) -> float:
        """
        Approximate area of a WGS-84 bounding box in km².
        Uses spherical Earth (R=6371 km) with mid-latitude width correction.
        """
        R = 6371.0
        lat_mid = radians((lat_min + lat_max) / 2.0)
        width_km  = R * radians(abs(lon_max - lon_min)) * cos(lat_mid)
        height_km = R * radians(abs(lat_max - lat_min))
        return abs(width_km * height_km)

    @staticmethod
    def _unpack_bbox(bbox) -> tuple[float, float, float, float]:
        """Accept sentinelhub.BBox or a 4-element sequence."""
        if hasattr(bbox, "lower_left"):
            lon_min, lat_min = bbox.lower_left
            lon_max, lat_max = bbox.upper_right
        else:
            lon_min, lat_min, lon_max, lat_max = bbox
        return lon_min, lat_min, lon_max, lat_max

    # ------------------------------------------------------------------
    # 1. Regional Licensing (Geofencing)
    # ------------------------------------------------------------------

    def validate_region(self, username: str, bbox) -> None:
        """
        Verify that the request bbox does not violate the client's
        licensed_geofence.  Both the centre point AND all four corners
        must fall inside the licensed polygon.

        Raises RegionalLicenseViolation if any point is outside.
        Silently passes when no geofence is configured (unrestricted licence).
        """
        profile = self.get_profile(username)
        if not profile or not profile.get("licensed_geofence"):
            return  # no restriction

        raw = profile["licensed_geofence"]
        geojson = json.loads(raw) if isinstance(raw, str) else raw
        # Accept both GeoJSON Feature and bare Geometry
        geom_dict = geojson.get("geometry", geojson)
        fence = shape(geom_dict)

        lon_min, lat_min, lon_max, lat_max = self._unpack_bbox(bbox)

        checkpoints: dict[str, tuple[float, float]] = {
            "centre": ((lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0),
            "SW":     (lon_min, lat_min),
            "SE":     (lon_max, lat_min),
            "NW":     (lon_min, lat_max),
            "NE":     (lon_max, lat_max),
        }
        for label, (lon, lat) in checkpoints.items():
            if not fence.contains(Point(lon, lat)):
                raise RegionalLicenseViolation(
                    f"BBox {label} corner ({lon:.4f}, {lat:.4f}) falls outside "
                    f"the licensed region for client '{username}'."
                )

    # ------------------------------------------------------------------
    # 2. Consumption Logic (Credit Ledger)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_mission_cost(
        area_km2: float, sensor: str, temporal_depth_days: int
    ) -> float:
        """
        Mission cost = Area (km²) × Sensor_Multiplier × Temporal_Depth (days).

        Multipliers:
          Sentinel-2 family (OPTICAL/NDVI/NDWI/NBR): 1.0
          Sentinel-1 family (RADAR/S1_RTC):           2.5

        Minimum cost is 1.0 credit regardless of area.
        """
        multiplier = SENSOR_MULTIPLIERS.get(sensor.upper(), 1.0)
        cost = area_km2 * multiplier * max(1, temporal_depth_days)
        return max(1.0, round(cost, 4))

    def verify_credits(self, username: str, cost: float) -> None:
        """
        Read-only balance check — raises InsufficientCredits if the client
        cannot cover *cost*.  Does NOT modify any state.
        """
        profile = self.get_profile(username)
        if not profile:
            raise InsufficientCredits(
                f"No client profile found for '{username}'. "
                "Contact your administrator to provision an account."
            )
        balance = profile["credit_balance"]
        if balance < cost:
            raise InsufficientCredits(
                f"Insufficient credits for '{username}': "
                f"required {cost:.2f}, available {balance:.2f}."
            )

    def pend_credits(self, username: str, job_id: str, cost: float) -> int:
        """
        Credit Lock — atomically verify and hold *cost* credits before any
        data is fetched from Copernicus/CDSE endpoints.

        Uses BEGIN IMMEDIATE to serialise concurrent requests against the
        same client profile and prevent double-spending.

        Returns the ledger row id so the caller can later settle or release.
        Raises InsufficientCredits if the balance is insufficient.
        """
        now = datetime.now(timezone.utc).isoformat()

        conn, release = get_raw_connection(self.db_path)
        try:
            if not is_postgres():
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute("BEGIN")

            # For PostgreSQL, row-level lock via FOR UPDATE
            lock_suffix = " FOR UPDATE" if is_postgres() else ""
            row = conn.execute(
                "SELECT user_id, credit_balance "
                f"FROM client_profiles WHERE username=?{lock_suffix}",
                (username,),
            ).fetchone()

            if not row:
                conn.execute("ROLLBACK")
                raise InsufficientCredits(
                    f"No client profile for '{username}'. Cannot pend credits."
                )

            if row["credit_balance"] < cost:
                conn.execute("ROLLBACK")
                raise InsufficientCredits(
                    f"Insufficient credits for job {job_id}: "
                    f"need {cost:.2f}, have {row['credit_balance']:.2f}."
                )

            user_id = row["user_id"]

            conn.execute(
                "UPDATE client_profiles "
                "SET credit_balance = credit_balance - ? WHERE username=?",
                (cost, username),
            )

            ledger_id: int = insert_returning_id(
                conn,
                "INSERT INTO credit_ledger "
                "(user_id, job_id, entry_type, amount, note, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (user_id, job_id, "PEND", -cost,
                 f"credit hold for job {job_id}", now),
            )

            conn.execute("COMMIT")

        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            release()

        logger.info(
            f"Credits pended: {cost:.2f} for {username}/{job_id} "
            f"(ledger_id={ledger_id})"
        )
        return ledger_id

    def deduct_credits(
        self,
        username: str,
        job_id: str,
        ledger_id: int,
        area_km2: float,
        cost: float,
    ) -> None:
        """
        Finalise the credit hold after successful PDF generation.

        The balance was already reduced by pend_credits(); here we:
          • Write the confirmatory DEDUCT ledger entry (audit trail).
          • Materialise the surface-area consumption in daily_surface_log.
        """
        now = datetime.now(timezone.utc).isoformat()
        profile = self.get_profile(username)
        if not profile:
            logger.warning(f"deduct_credits: no profile for '{username}', skipping.")
            return

        today = datetime.now(timezone.utc).date().isoformat()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO credit_ledger "
                "(user_id, job_id, entry_type, amount, note, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    profile["user_id"], job_id, "DEDUCT", -cost,
                    f"confirmed deduction for job {job_id} "
                    f"(pend_ledger_id={ledger_id})",
                    now,
                ),
            )
            # Upsert daily surface consumption
            conn.execute(
                """
                INSERT INTO daily_surface_log (user_id, date, surface_used_km2)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, date) DO UPDATE
                  SET surface_used_km2 = surface_used_km2 + excluded.surface_used_km2
                """,
                (profile["user_id"], today, area_km2),
            )

        logger.info(
            f"Credits deducted: {cost:.2f} for {username}/{job_id} | "
            f"surface logged: {area_km2:.2f} km² on {today}"
        )

    def release_credits(
        self, username: str, job_id: str, ledger_id: int, cost: float
    ) -> None:
        """
        Unwind the credit hold after a pipeline failure.

        Restores the balance and writes a RELEASE ledger entry so the
        ledger remains balanced and auditable.
        """
        now = datetime.now(timezone.utc).isoformat()
        profile = self.get_profile(username)
        if not profile:
            logger.warning(f"release_credits: no profile for '{username}', skipping.")
            return

        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE client_profiles "
                "SET credit_balance = credit_balance + ? WHERE username=?",
                (cost, username),
            )
            conn.execute(
                "INSERT INTO credit_ledger "
                "(user_id, job_id, entry_type, amount, note, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    profile["user_id"], job_id, "RELEASE", cost,
                    f"hold released for failed job {job_id} "
                    f"(pend_ledger_id={ledger_id})",
                    now,
                ),
            )

        logger.info(
            f"Credits released: {cost:.2f} back to {username} "
            f"(job {job_id} failed, ledger_id={ledger_id})"
        )

    # ------------------------------------------------------------------
    # 3. Daily Surface Cap
    # ------------------------------------------------------------------

    def check_daily_surface_cap(
        self, username: str, requested_area_km2: float
    ) -> None:
        """
        Verify the client hasn't exhausted their daily km² scanning quota.

        Raises DailyQuotaExhausted if adding *requested_area_km2* to
        today's total would breach the cap.

        Note: this is a soft check only — the area is NOT reserved here.
        Surface consumption is materialised in deduct_credits() on success,
        meaning failed jobs do NOT count against the daily cap.
        """
        profile = self.get_profile(username)
        if not profile:
            return  # no profile → no cap enforced

        cap = profile["daily_surface_cap_km2"]
        if cap <= 0:
            return  # 0 or negative cap means unlimited

        today = datetime.now(timezone.utc).date().isoformat()
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT surface_used_km2 FROM daily_surface_log "
                "WHERE user_id=? AND date=?",
                (profile["user_id"], today),
            ).fetchone()

        used      = row["surface_used_km2"] if row else 0.0
        remaining = cap - used

        if requested_area_km2 > remaining:
            raise DailyQuotaExhausted(
                f"Daily surface cap reached for '{username}': "
                f"cap={cap:.0f} km², used={used:.2f} km², "
                f"requested={requested_area_km2:.2f} km², "
                f"remaining={remaining:.2f} km². "
                f"Quota resets at midnight UTC."
            )

    # ------------------------------------------------------------------
    # 4. Usage Accountability Log
    # ------------------------------------------------------------------

    def log_usage(
        self,
        job_id: str,
        username: str,
        credits_consumed: float,
        bbox_area_km2: float,
        sensor: str,
        temporal_depth_days: int,
        status: str = "COMPLETED",
    ) -> None:
        """
        Write the final, immutable usage record for *job_id*.

        Every Job ID is logged with: Client_ID, Credits_Consumed,
        Timestamp, BBox_Area, Sensor, Temporal_Depth, and Status.
        Uses INSERT OR REPLACE so a retry on transient DB error is safe.
        """
        profile = self.get_profile(username)
        client_id = profile["user_id"] if profile else username
        now = datetime.now(timezone.utc).isoformat()

        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO usage_log
                    (job_id, client_id, username, sensor, bbox_area_km2,
                     temporal_depth_days, credits_consumed, timestamp, status)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT (job_id) DO UPDATE SET
                    client_id           = excluded.client_id,
                    username            = excluded.username,
                    sensor              = excluded.sensor,
                    bbox_area_km2       = excluded.bbox_area_km2,
                    temporal_depth_days = excluded.temporal_depth_days,
                    credits_consumed    = excluded.credits_consumed,
                    timestamp           = excluded.timestamp,
                    status              = excluded.status
                """,
                (
                    job_id, client_id, username, sensor.upper(), bbox_area_km2,
                    temporal_depth_days, credits_consumed, now, status,
                ),
            )

        logger.info(
            f"Usage logged | job={job_id} client={client_id} "
            f"credits={credits_consumed:.2f} area={bbox_area_km2:.2f} km² "
            f"sensor={sensor} temporal={temporal_depth_days}d status={status}"
        )

    # ------------------------------------------------------------------
    # Read helpers (admin / reporting use)
    # ------------------------------------------------------------------

    def get_usage_summary(self, username: str, limit: int = 50) -> list:
        """Return the most recent *limit* usage_log rows for *username*."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM usage_log WHERE username=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_surface_used(self, username: str) -> float:
        """Return km² consumed today (UTC) for *username*."""
        profile = self.get_profile(username)
        if not profile:
            return 0.0
        today = datetime.now(timezone.utc).date().isoformat()
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT surface_used_km2 FROM daily_surface_log "
                "WHERE user_id=? AND date=?",
                (profile["user_id"], today),
            ).fetchone()
        return row["surface_used_km2"] if row else 0.0

    def get_credit_balance(self, username: str) -> float:
        """Return the current (post-pend) credit balance for *username*."""
        profile = self.get_profile(username)
        return profile["credit_balance"] if profile else 0.0

    def get_ledger(self, username: str, limit: int = 100) -> list:
        """Return the most recent *limit* credit ledger entries for *username*."""
        profile = self.get_profile(username)
        if not profile:
            return []
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM credit_ledger WHERE user_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (profile["user_id"], limit),
            ).fetchall()
        return [dict(r) for r in rows]
