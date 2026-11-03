"""Supabase Storage over REST, server side only (the browser never sees a Supabase key).

- Signed upload URL:  POST /storage/v1/object/upload/sign/{bucket}/{key}   -> {"url": "/object/upload/sign/...?token=..."}
  The browser then PUTs the raw file to that URL (valid ~2 h, one object, no key needed).
- Download:           GET  /storage/v1/object/{bucket}/{key}
- Signed download:    POST /storage/v1/object/sign/{bucket}/{key} {expiresIn} -> {"signedURL": "/object/sign/...?token=..."}
- Delete:             DELETE /storage/v1/object/{bucket} {"prefixes": [key]}

New-style secret keys (sb_secret_...) are not JWTs and go in the `apikey` header only; a legacy
service_role JWT (eyJ...) is also sent as `Authorization: Bearer`.
"""
from urllib.parse import parse_qs, quote, urlsplit

import httpx


class StorageError(RuntimeError):
    """`transient`: a 5xx, timeout or network failure, worth retrying later."""

    def __init__(self, message: str, status: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status = status
        self.transient = transient


class NotFound(StorageError):
    pass


def _key(path: str) -> str:
    return quote(path.lstrip("/"), safe="/")


class SupabaseStorage:
    def __init__(self, url: str, secret_key: str, bucket: str, client: httpx.Client | None = None):
        self.base = url.rstrip("/") + "/storage/v1"
        self.bucket = bucket
        self.headers = {"apikey": secret_key}
        if secret_key.startswith("eyJ"):
            self.headers["Authorization"] = f"Bearer {secret_key}"
        self.client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    def _send(self, what: str, method: str, url: str, timeout: float | None = None, **kwargs) -> httpx.Response:
        if timeout is not None:
            t = max(1.0, min(60.0, timeout))
            kwargs["timeout"] = httpx.Timeout(t, connect=min(10.0, t))
        try:
            return self.client.request(method, url, headers=self.headers, **kwargs)
        except httpx.HTTPError as exc:
            raise StorageError(f"{what}: Supabase Storage unreachable ({exc.__class__.__name__})",
                               transient=True) from exc

    def _check(self, resp: httpx.Response, what: str) -> None:
        if resp.status_code < 400:
            return
        text = resp.text[:300]
        # Storage reports a missing object as 400 {"statusCode":"404",...} or a plain 404.
        if resp.status_code == 404 or '"404"' in text or "not_found" in text.lower() or "Object not found" in text:
            raise NotFound(f"{what}: object not found", status=404)
        raise StorageError(f"{what}: Supabase Storage HTTP {resp.status_code}: {text}", status=resp.status_code,
                           transient=resp.status_code >= 500 or resp.status_code == 429)

    def create_upload_url(self, path: str) -> dict:
        resp = self._send("create signed upload URL", "POST",
                          f"{self.base}/object/upload/sign/{self.bucket}/{_key(path)}", json={})
        self._check(resp, "create signed upload URL")
        rel = (resp.json() or {}).get("url") or ""
        full = self.base + rel if rel.startswith("/") else rel
        token = (parse_qs(urlsplit(full).query).get("token") or [""])[0]
        if not token:
            raise StorageError("create signed upload URL: no token returned")
        return {"url": full, "token": token}

    def download(self, path: str, timeout: float | None = None) -> bytes:
        resp = self._send("download", "GET", f"{self.base}/object/{self.bucket}/{_key(path)}", timeout=timeout)
        self._check(resp, "download")
        return resp.content

    def signed_download_url(self, path: str, expires_in: int = 300) -> str:
        resp = self._send("create signed download URL", "POST", f"{self.base}/object/sign/{self.bucket}/{_key(path)}",
                          json={"expiresIn": int(expires_in)})
        self._check(resp, "create signed download URL")
        data = resp.json() or {}
        rel = data.get("signedURL") or data.get("signedUrl") or ""
        if not rel:
            raise StorageError("create signed download URL: no URL returned")
        return self.base + rel if rel.startswith("/") else rel

    def delete(self, path: str) -> None:
        resp = self._send("delete", "DELETE", f"{self.base}/object/{self.bucket}",
                          json={"prefixes": [path.lstrip("/")]})
        self._check(resp, "delete")
