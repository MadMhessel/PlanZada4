"""Zero-knowledge encryption helpers using Fernet."""
from __future__ import annotations

import os
from base64 import urlsafe_b64encode
from typing import Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=default_backend(),
    )
    return urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_text(password: str, text: str) -> str:
    """Encrypt text with password-derived key."""
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    token = Fernet(key).encrypt(text.encode())
    return f"{salt.hex()}${token.decode()}"


def decrypt_text(password: str, encrypted: str) -> str:
    """Decrypt text that was produced by encrypt_text."""
    try:
        salt_hex, token_str = encrypted.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        key = _derive_key(password, salt)
        return Fernet(key).decrypt(token_str.encode()).decode()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid password or data") from exc


__all__ = ["encrypt_text", "decrypt_text"]
