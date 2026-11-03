"""Gemini request shape, normalisation and 429 handling; Supabase Storage REST calls (fake HTTP)."""
import json
import math

import httpx
import pytest

from core.embed import EmbedError, Gemini, RateLimited, l2_normalise, to_pgvector
from core.storage import NotFound, StorageError, SupabaseStorage


def _gemini(handler, dims=768):
    return Gemini("test-key", "gemini-embedding-001", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_batch_request_shape_batching_and_normalisation():
    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append((req.url, req.headers, body))
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0] + [0.0] * 766}
                                                        for _ in body["requests"]]})

    g = _gemini(handler)
    vecs = g.embed_documents([f"text {i}" for i in range(250)], title="notes.pdf")
    assert [len(b["requests"]) for _, _, b in seen] == [100, 100, 50]
    url, headers, body = seen[0]
    assert str(url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    assert headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(url)                       # key never in the URL
    r0 = body["requests"][0]
    assert r0 == {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "text 0"}]},
                  "taskType": "RETRIEVAL_DOCUMENT", "outputDimensionality": 768, "title": "notes.pdf"}
    assert len(vecs) == 250
    assert vecs[0][:2] == pytest.approx([0.6, 0.8])
    assert math.isclose(math.sqrt(sum(v * v for v in vecs[0])), 1.0)


def test_query_request_shape():
    seen = []

    def handler(req):
        seen.append((str(req.url), json.loads(req.content)))
        return httpx.Response(200, json={"embedding": {"values": [1.0] * 768}})

    v = _gemini(handler).embed_query("what is osmosis?")
    url, body = seen[0]
    assert url.endswith("/models/gemini-embedding-001:embedContent")
    assert body == {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "what is osmosis?"}]},
                    "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 768}
    assert math.isclose(sum(x * x for x in v), 1.0)


def test_429_is_rate_limited_with_retry_delay():
    def handler(req):
        return httpx.Response(429, json={"error": {"code": 429, "message": "Resource exhausted", "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}})

    with pytest.raises(RateLimited) as e:
        _gemini(handler).embed_documents(["a"], title="t")
    assert e.value.transient and e.value.retry_after == 17.0 and e.value.status == 429


def test_errors():
    def wrong_dims(req):
        return httpx.Response(200, json={"embeddings": [{"values": [1.0] * 3072}]})

    with pytest.raises(EmbedError, match="3072 dimensions"):
        _gemini(wrong_dims).embed_documents(["a"], title="t")

    def bad_key(req):
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    with pytest.raises(EmbedError) as e:
        _gemini(bad_key).embed_query("q")
    assert not e.value.transient and "API key not valid" in str(e.value)

    def overloaded(req):
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    with pytest.raises(EmbedError) as e:
        _gemini(overloaded).embed_query("q")
    assert e.value.transient

    with pytest.raises(EmbedError):
        l2_normalise([0.0, 0.0])


def test_pgvector_literal():
    assert to_pgvector([0.5, -0.25, 1e-9]) == "[0.5,-0.25,1e-09]"


# --- storage ---------------------------------------------------------------------------------
def _storage(handler, key="sb_secret_abc"):
    return SupabaseStorage("https://proj.supabase.co/", key, "cortex-docs",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_storage_signed_upload_download_delete():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        p = req.url.path
        if p.startswith("/storage/v1/object/upload/sign/"):
            return httpx.Response(200, json={"url": "/object/upload/sign/cortex-docs/docs/u/a.pdf?token=TOK"})
        if p.startswith("/storage/v1/object/sign/"):
            assert json.loads(req.content) == {"expiresIn": 300}
            return httpx.Response(200, json={"signedURL": "/object/sign/cortex-docs/docs/u/a.pdf?token=DL"})
        if req.method == "GET":
            return httpx.Response(200, content=b"%PDF-1.7")
        if req.method == "DELETE":
            assert json.loads(req.content) == {"prefixes": ["docs/u/a.pdf"]}
            return httpx.Response(200, json=[{"name": "docs/u/a.pdf"}])
        return httpx.Response(500)

    s = _storage(handler)
    up = s.create_upload_url("docs/u/a.pdf")
    assert up == {"url": "https://proj.supabase.co/storage/v1/object/upload/sign/cortex-docs/docs/u/a.pdf?token=TOK",
                  "token": "TOK"}
    assert s.download("docs/u/a.pdf") == b"%PDF-1.7"
    assert s.signed_download_url("docs/u/a.pdf").endswith("/storage/v1/object/sign/cortex-docs/docs/u/a.pdf?token=DL")
    s.delete("docs/u/a.pdf")
    assert [r.method + " " + r.url.path for r in seen] == [
        "POST /storage/v1/object/upload/sign/cortex-docs/docs/u/a.pdf",
        "GET /storage/v1/object/cortex-docs/docs/u/a.pdf",
        "POST /storage/v1/object/sign/cortex-docs/docs/u/a.pdf",
        "DELETE /storage/v1/object/cortex-docs",
    ]
    for r in seen:  # new secret keys: apikey header only, never Bearer
        assert r.headers["apikey"] == "sb_secret_abc"
        assert "authorization" not in r.headers


def test_storage_legacy_jwt_key_also_sent_as_bearer_and_errors():
    def handler(req):
        assert req.headers["authorization"] == "Bearer eyJhbGciOi.x.y"
        if req.method == "GET":
            return httpx.Response(400, json={"statusCode": "404", "error": "not_found", "message": "Object not found"})
        return httpx.Response(403, json={"message": "signature verification failed"})

    s = _storage(handler, key="eyJhbGciOi.x.y")
    with pytest.raises(NotFound):
        s.download("docs/u/missing.pdf")
    with pytest.raises(StorageError) as e:
        s.create_upload_url("docs/u/a.pdf")
    assert e.value.status == 403


def test_daily_quota_is_recognised():
    def per_day(req):
        return httpx.Response(429, json={"error": {"code": 429, "message": "Quota exceeded", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaMetric": "generativelanguage.googleapis.com/embed_content_requests",
                 "quotaId": "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}})

    with pytest.raises(RateLimited) as e:
        _gemini(per_day).embed_documents(["a"], title="t")
    assert e.value.daily and "daily quota" in str(e.value)

    def long_delay(req):
        return httpx.Response(429, json={"error": {"details": [{"retryDelay": "3600s"}]}})

    with pytest.raises(RateLimited) as e:
        _gemini(long_delay).embed_query("q")
    assert e.value.daily and e.value.retry_after == 3600.0

    def per_minute(req):
        return httpx.Response(429, json={"error": {"details": [
            {"violations": [{"quotaId": "EmbedContentRequestsPerMinutePerProjectPerModel"}]},
            {"retryDelay": "17s"}]}})

    with pytest.raises(RateLimited) as e:
        _gemini(per_minute).embed_query("q")
    assert not e.value.daily and e.value.retry_after == 17.0


def test_per_call_timeout_is_capped():
    seen = []

    def handler(req):
        seen.append(req.extensions["timeout"])
        return httpx.Response(200, json={"embeddings": [{"values": [1.0] * 768}]})

    g = _gemini(handler)
    g.embed_documents(["a"], title="t", timeout=7.5)
    g.embed_documents(["a"], title="t", timeout=500)
    g.embed_documents(["a"], title="t", timeout=-3)
    assert [t["read"] for t in seen] == [7.5, 60.0, 1.0]
    assert seen[0]["connect"] == 7.5 and seen[1]["connect"] == 10.0


def test_storage_transient_errors():
    def busy(req):
        return httpx.Response(503, text="upstream busy")

    with pytest.raises(StorageError) as e:
        _storage(busy).download("docs/u/a.pdf")
    assert e.value.transient and e.value.status == 503

    def unreachable(req):
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(StorageError) as e:
        _storage(unreachable).download("docs/u/a.pdf", timeout=5)
    assert e.value.transient

    def forbidden(req):
        return httpx.Response(403, json={"message": "nope"})

    with pytest.raises(StorageError) as e:
        _storage(forbidden).delete("docs/u/a.pdf")
    assert not e.value.transient
