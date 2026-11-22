"""Optional local MFA/RBAC for the Streamlit prototype.

Enable with FINANCE_REQ_AUTH_REQUIRED=1 and point FINANCE_REQ_USERS_FILE at a
private JSON file created with ``python -m src.security add-user``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import time
from getpass import getpass
from urllib.parse import quote


AUTH_REQUIRED = os.getenv("FINANCE_REQ_AUTH_REQUIRED", "0").strip().lower() in {"1", "true", "yes"}
USERS_FILE = os.getenv("FINANCE_REQ_USERS_FILE", "")


def load_users() -> dict:
    if not USERS_FILE:
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def password_record(password: str, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return {"salt": base64.b64encode(salt).decode(), "password_hash": base64.b64encode(derived).decode()}


def verify_login(users: dict, username: str, password: str, code: str) -> dict | None:
    user = users.get(username)
    if not user or user.get("disabled"):
        return None
    try:
        salt = base64.b64decode(user["salt"])
        expected = base64.b64decode(user["password_hash"])
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
        secret = user["totp_secret"]
    except (KeyError, ValueError):
        return None
    if not hmac.compare_digest(actual, expected) or not verify_totp(secret, code):
        return None
    return {"username": username, "role": user.get("role", "viewer")}


def verify_totp(secret: str, code: str, now: int | None = None) -> bool:
    if not code.isdigit() or len(code) != 6:
        return False
    try:
        key = base64.b32decode(secret.upper() + "=" * ((8 - len(secret) % 8) % 8))
    except (ValueError, base64.binascii.Error):
        return False
    tick = int((now if now is not None else time.time()) // 30)
    return any(hmac.compare_digest(_totp(key, tick + offset), code) for offset in (-1, 0, 1))


def _totp(key: bytes, tick: int) -> str:
    digest = hmac.new(key, struct.pack(">Q", tick), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % 1_000_000
    return f"{number:06d}"


def can_run(role: str) -> bool:
    return role in {"admin", "analyst", "business_analyst", "product_owner"}


def can_approve(role: str) -> bool:
    return role in {"admin", "approver", "product_owner", "compliance", "security", "architect", "project_manager", "operations"}


def _add_user(path: str, username: str, role: str) -> None:
    users = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as stream:
            users = json.load(stream)
    if username in users:
        raise SystemExit(f"User {username!r} already exists.")
    password = getpass("Password (minimum 14 characters): ")
    if len(password) < 14:
        raise SystemExit("Password must be at least 14 characters.")
    confirm = getpass("Confirm password: ")
    if password != confirm:
        raise SystemExit("Passwords did not match.")
    secret = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")
    users[username] = {**password_record(password), "totp_secret": secret, "role": role, "disabled": False}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(users, stream, indent=2)
    print(f"Created {username} ({role}). Add this TOTP secret to an authenticator: {secret}")
    print(f"otpauth://totp/FinanceReq:{quote(username)}?secret={secret}&issuer=FinanceReq")
    print("Protect the user file with operating-system permissions and back it up securely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage local Finance Requirements workspace users")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add-user")
    add.add_argument("--file", required=True)
    add.add_argument("--username", required=True)
    add.add_argument("--role", required=True, choices=["admin", "analyst", "business_analyst", "product_owner", "approver", "compliance", "security", "architect", "project_manager", "operations", "auditor", "viewer"])
    args = parser.parse_args()
    _add_user(args.file, args.username, args.role)
