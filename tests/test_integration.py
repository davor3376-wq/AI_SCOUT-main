"""
Integration tests for AI SCOUT — uses FastAPI TestClient against an
in-memory SQLite database so no external services or files are needed.

Run:
    pytest tests/test_integration.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_file(tmp_path_factory):
    """Temporary SQLite database file, shared for the whole module."""
    f = tmp_path_factory.mktemp("scout_db") / "test_jobs.db"
    os.environ["SCOUT_DB_PATH"] = str(f)
    yield str(f)
    # cleanup handled by tmp_path_factory


@pytest.fixture(scope="module")
def client(db_file):
    """TestClient with lifespan disabled (no scheduler background task)."""
    from app.api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def admin_token(client):
    """Bootstrap the first admin user and return a valid JWT."""
    resp = client.post("/v1/auth/setup", json={
        "username": "testadmin",
        "password": "TestAdmin1!",
        "role": "admin",
    })
    # 201 = created; 409 = already exists (re-run)
    assert resp.status_code in (201, 409), resp.text

    resp = client.post(
        "/v1/auth/login",
        data={"username": "testadmin", "password": "TestAdmin1!"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body


# ---------------------------------------------------------------------------
# /auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_setup_conflict_on_second_call(self, client, admin_token):
        resp = client.post("/v1/auth/setup", json={
            "username": "another",
            "password": "AnotherPass1!",
        })
        assert resp.status_code == 409

    def test_login_wrong_password(self, client):
        resp = client.post(
            "/v1/auth/login",
            data={"username": "testadmin", "password": "wrong"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post(
            "/v1/auth/login",
            data={"username": "nobody", "password": "NoPass1!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 401

    def test_me_returns_profile(self, client, auth_headers):
        resp = client.get("/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "testadmin"
        assert body["role"] == "admin"
        # api_key must NOT be in the response
        assert "api_key" not in body
        # hashed_pw must NOT be in the response
        assert "hashed_pw" not in body

    def test_me_unauthenticated(self, client):
        resp = client.get("/v1/auth/me")
        assert resp.status_code == 401

    def test_create_user_as_admin(self, client, auth_headers):
        resp = client.post("/v1/auth/users", json={
            "username": "analyst1",
            "password": "Analyst1Pass!",
            "role": "analyst",
        }, headers=auth_headers)
        assert resp.status_code == 201
        body = resp.json()
        assert body["role"] == "analyst"
        assert "api_key" in body

    def test_create_user_weak_password(self, client, auth_headers):
        resp = client.post("/v1/auth/users", json={
            "username": "weakuser",
            "password": "short",
            "role": "analyst",
        }, headers=auth_headers)
        assert resp.status_code == 400

    def test_create_user_invalid_role(self, client, auth_headers):
        resp = client.post("/v1/auth/users", json={
            "username": "roleuser",
            "password": "ValidPass1!",
            "role": "superuser",
        }, headers=auth_headers)
        assert resp.status_code == 400

    def test_list_users_as_admin(self, client, auth_headers):
        resp = client.get("/v1/auth/users", headers=auth_headers)
        assert resp.status_code == 200
        users = resp.json()
        assert isinstance(users, list)
        usernames = [u["username"] for u in users]
        assert "testadmin" in usernames

    def test_rotate_key_returns_new_key(self, client, auth_headers):
        resp = client.post("/v1/auth/rotate-key", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "api_key" in body
        assert body["api_key"].startswith("sk-")

    def test_api_key_auth(self, client, auth_headers):
        rotate = client.post("/v1/auth/rotate-key", headers=auth_headers)
        new_key = rotate.json()["api_key"]
        resp = client.get("/v1/auth/me", headers={"X-API-Key": new_key})
        assert resp.status_code == 200

    def test_refresh_token(self, client):
        login = client.post(
            "/v1/auth/login",
            data={"username": "testadmin", "password": "TestAdmin1!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        refresh_token = login.json()["refresh_token"]
        resp = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body

    def test_refresh_with_invalid_token(self, client):
        resp = client.post("/v1/auth/refresh", json={"refresh_token": "invalid.token.here"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /jobs
# ---------------------------------------------------------------------------

class TestJobs:
    def test_list_jobs_empty(self, client, auth_headers):
        resp = client.get("/v1/jobs", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "jobs" in body
        assert isinstance(body["jobs"], list)

    def test_list_jobs_pagination_params(self, client, auth_headers):
        resp = client.get("/v1/jobs?limit=10&offset=0", headers=auth_headers)
        assert resp.status_code == 200

    def test_list_jobs_unauthenticated(self, client):
        resp = client.get("/v1/jobs")
        assert resp.status_code == 401

    def test_get_nonexistent_job(self, client, auth_headers):
        resp = client.get("/v1/jobs/JOB-DOESNOTEXIST", headers=auth_headers)
        assert resp.status_code == 404

    def test_delete_nonexistent_job(self, client, auth_headers):
        resp = client.delete("/v1/jobs/JOB-DOESNOTEXIST", headers=auth_headers)
        assert resp.status_code == 404

    def test_download_pdf_nonexistent_job(self, client, auth_headers):
        resp = client.get("/v1/jobs/JOB-DOESNOTEXIST/pdf", headers=auth_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /admin
# ---------------------------------------------------------------------------

class TestAdmin:
    def test_audit_logs_accessible_to_admin(self, client, auth_headers):
        resp = client.get("/v1/admin/audit-logs", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_audit_logs_forbidden_to_viewer(self, client, auth_headers):
        viewer_resp = client.post("/v1/auth/users", json={
            "username": "viewer1",
            "password": "Viewer1Pass!",
            "role": "viewer",
        }, headers=auth_headers)
        assert viewer_resp.status_code == 201

        login = client.post(
            "/v1/auth/login",
            data={"username": "viewer1", "password": "Viewer1Pass!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        viewer_token = login.json()["access_token"]
        resp = client.get(
            "/v1/admin/audit-logs",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403

    def test_list_users_admin_endpoint(self, client, auth_headers):
        resp = client.get("/v1/admin/users", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# /auth/credits
# ---------------------------------------------------------------------------

class TestCredits:
    def test_credits_endpoint_accessible(self, client, auth_headers):
        resp = client.get("/v1/auth/credits", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "credit_balance" in body
        assert "daily_surface_cap_km2" in body
        assert "recent_usage" in body


# ---------------------------------------------------------------------------
# Mission validation (no Sentinel Hub calls — purely Pydantic validation)
# ---------------------------------------------------------------------------

class TestMissionValidation:
    """Test input validation for /launch_custom_mission without hitting SH."""

    def test_invalid_sensor_rejected(self, client, auth_headers):
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "Point", "coordinates": [16.37, 48.21]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sensor": "INVALID_SENSOR",
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_invalid_geometry_type_rejected(self, client, auth_headers):
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sensor": "OPTICAL",
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_out_of_range_longitude_rejected(self, client, auth_headers):
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "Point", "coordinates": [200.0, 48.21]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sensor": "OPTICAL",
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_out_of_range_latitude_rejected(self, client, auth_headers):
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "Point", "coordinates": [16.37, 100.0]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sensor": "OPTICAL",
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_invalid_date_format_rejected(self, client, auth_headers):
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "Point", "coordinates": [16.37, 48.21]},
            "start_date": "not-a-date",
            "end_date": "2024-01-31",
            "sensor": "OPTICAL",
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_unauthenticated_mission_rejected(self, client):
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "Point", "coordinates": [16.37, 48.21]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sensor": "OPTICAL",
        })
        assert resp.status_code == 401

    def test_viewer_cannot_launch_mission(self, client, auth_headers):
        login = client.post(
            "/v1/auth/login",
            data={"username": "viewer1", "password": "Viewer1Pass!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if login.status_code != 200:
            pytest.skip("viewer1 not yet created")
        viewer_token = login.json()["access_token"]
        resp = client.post("/v1/launch_custom_mission", json={
            "geometry": {"type": "Point", "coordinates": [16.37, 48.21]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "sensor": "OPTICAL",
        }, headers={"Authorization": f"Bearer {viewer_token}"})
        assert resp.status_code == 403
