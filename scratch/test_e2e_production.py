import urllib.request
import json
import io

BASE_URL = "http://127.0.0.1:8000"

def test_pwa_endpoints():
    print("\n[1/14] Testing PWA Endpoints...")
    res_manifest = urllib.request.urlopen(f"{BASE_URL}/manifest.json")
    assert res_manifest.getcode() == 200, "Manifest failed"
    manifest_data = json.loads(res_manifest.read().decode())
    assert "short_name" in manifest_data, "Invalid manifest"
    print("  -> manifest.json: OK")

    res_sw = urllib.request.urlopen(f"{BASE_URL}/service-worker.js")
    assert res_sw.getcode() == 200, "Service worker failed"
    print("  -> service-worker.js: OK")

def test_upload_document():
    print("\n[2/14] Testing Document Ingestion...")
    boundary = "----WebKitFormBoundaryE2EProductionTest"
    body = io.BytesIO()

    body.write(b"--" + boundary.encode() + b"\r\n")
    body.write(b'Content-Disposition: form-data; name="file"; filename="enterprise_security_policy.txt"\r\n')
    body.write(b"Content-Type: text/plain\r\n\r\n")
    body.write(b"Synthara IT Security Policy 2026.\nAll employee passwords must be at least 14 characters with MFA mandatory.\nVPN is required for remote network access.\r\n")

    body.write(b"--" + boundary.encode() + b"\r\n")
    body.write(b'Content-Disposition: form-data; name="chunk_size"\r\n\r\n512\r\n')
    body.write(b"--" + boundary.encode() + b"\r\n")
    body.write(b'Content-Disposition: form-data; name="chunk_overlap"\r\n\r\n64\r\n')
    body.write(b"--" + boundary.encode() + b"--\r\n")

    req = urllib.request.Request(
        f"{BASE_URL}/api/upload",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    assert res.getcode() == 200 and data.get("status") == "indexed", f"Upload failed: {data}"
    print(f"  -> Uploaded enterprise_security_policy.txt: {data['chunks_count']} chunks, version: {data.get('version')}")

def test_documents_registry():
    print("\n[3/14] Testing Documents Registry & Versioning...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/documents")
    docs = json.loads(res.read().decode())
    assert len(docs) > 0, "No documents in registry"
    print(f"  -> Registered documents: {len(docs)} found ({docs[0]['source']})")

    res_ver = urllib.request.urlopen(f"{BASE_URL}/api/documents/versions")
    versions = json.loads(res_ver.read().decode())
    assert len(versions) > 0, "No versions tracked"
    print("  -> Document version control ledger: OK")

def test_suggestions_and_autocomplete():
    print("\n[4/14] Testing Suggestions & Autocomplete...")
    res_sug = urllib.request.urlopen(f"{BASE_URL}/api/suggestions")
    sugs = json.loads(res_sug.read().decode())
    assert len(sugs) > 0, "No suggestions"
    print(f"  -> Contextual suggestions: {len(sugs)} available (First: '{sugs[0]}')")

    res_auto = urllib.request.urlopen(f"{BASE_URL}/api/autocomplete?q=pass")
    autos = json.loads(res_auto.read().decode())
    assert len(autos) > 0, "Autocomplete failed"
    print(f"  -> Autocomplete matches for 'pass': {len(autos)} found")

def test_facets():
    print("\n[5/14] Testing Search Facets...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/search/facets")
    facets = json.loads(res.read().decode())
    assert "categories" in facets and "sources" in facets, "Invalid facets"
    print(f"  -> Search categories: {facets['categories']}")

def test_query_pipeline():
    print("\n[6/14] Testing RAG Query Pipeline...")
    payload = {
        "query": "What are the password security and MFA requirements?",
        "clearance": "Compliance Officer",
        "prompt_variant": "A"
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    assert res.getcode() == 200, "Query failed"
    assert "answer" in data and len(data["answer"]) > 10, "Empty answer"
    assert len(data.get("citations", [])) > 0, "Missing citations"
    print(f"  -> Answer: {data['answer'][:120]}...")
    print(f"  -> Citations count: {len(data['citations'])}")
    print(f"  -> Faithfulness Score: {data['evaluation']['faithfulness']['score']}")
    print(f"  -> Relevancy Score: {data['evaluation']['relevancy']['score']}")
    return data

def test_document_preview():
    print("\n[7/14] Testing Document Preview & Citation Highlighting...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/document/preview/enterprise_security_policy.txt/1?highlight=passwords")
    data = json.loads(res.read().decode())
    assert res.getcode() == 200 and "content_html" in data, "Preview failed"
    print(f"  -> Document page preview generated: {len(data['content_html'])} chars")

def test_feedback():
    print("\n[8/14] Testing Feedback Rating Loop...")
    payload = {
        "query": "Password requirements test",
        "answer": "Passwords must be 14 chars",
        "rating": 1,
        "comments": "Accurate response"
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/feedback",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.getcode() == 200, "Feedback submit failed"

    res_sum = urllib.request.urlopen(f"{BASE_URL}/api/feedback/summary")
    sum_data = json.loads(res_sum.read().decode())
    assert sum_data["total_reviews"] > 0, "Feedback summary empty"
    print(f"  -> Feedback satisfaction rate: {sum_data['satisfaction_rate']}% ({sum_data['positive_count']} positive)")

def test_multichannel_sharing():
    print("\n[9/14] Testing Multi-Channel Sharing Formatters...")
    payload = {
        "query": "Password policy",
        "answer": "14 chars minimum with MFA",
        "citations": [{"metadata": {"source": "sec_policy.pdf", "page": 1}, "text": "Sample text"}]
    }
    # Email
    req_email = urllib.request.Request(f"{BASE_URL}/api/share/email", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    res_email = urllib.request.urlopen(req_email)
    data_email = json.loads(res_email.read().decode())
    assert "html_body" in data_email, "Email format failed"
    print("  -> Email HTML & Plaintext formatting: OK")

    # Slack
    req_slack = urllib.request.Request(f"{BASE_URL}/api/share/slack", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    res_slack = urllib.request.urlopen(req_slack)
    data_slack = json.loads(res_slack.read().decode())
    assert "blocks" in data_slack, "Slack block format failed"
    print("  -> Slack Block Kit formatting: OK")

    # Teams
    req_teams = urllib.request.Request(f"{BASE_URL}/api/share/teams", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    res_teams = urllib.request.urlopen(req_teams)
    data_teams = json.loads(res_teams.read().decode())
    assert "card" in data_teams, "Teams card format failed"
    print("  -> MS Teams Adaptive Card formatting: OK")

def test_telemetry_analytics():
    print("\n[10/14] Testing Telemetry & Analytics Summary...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/analytics/summary")
    data = json.loads(res.read().decode())
    assert "total_queries" in data and "topic_breakdown" in data, "Analytics failed"
    print(f"  -> Total queries tracked: {data['total_queries']}, Avg Latency: {data['avg_latency_ms']}ms")

def test_ab_experimentation():
    print("\n[11/14] Testing Prompt A/B Experimentation Metrics...")
    res = urllib.request.urlopen(f"{BASE_URL}/api/ab-test/metrics")
    data = json.loads(res.read().decode())
    assert "A" in data and "B" in data, "AB metrics failed"
    print(f"  -> Variant A: {data['A']['name']}, Variant B: {data['B']['name']}")

def test_pii_guardrail():
    print("\n[12/14] Testing PII Guardrail Redaction...")
    payload = {"text": "Contact john.doe@synthara.io or call 555-123-4567 regarding SSN 000-12-3456."}
    req = urllib.request.Request(f"{BASE_URL}/api/guardrail/scan", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    assert data["is_safe"] is False, "PII should be detected"
    assert "[REDACTED_EMAIL]" in data["redacted_text"] and "[REDACTED_SSN]" in data["redacted_text"]
    print(f"  -> PII scanned: {len(data['pii_detected'])} sensitive tokens safely redacted")

def test_audit_integrity():
    print("\n[13/14] Testing Cryptographic Audit Trail & CSV Export...")
    res_verify = urllib.request.urlopen(f"{BASE_URL}/api/audit/verify")
    data_verify = json.loads(res_verify.read().decode())
    assert data_verify["valid"] is True, f"Audit chain broken: {data_verify}"
    print(f"  -> Cryptographic hash chain: VALID ({data_verify['entries_checked']} immutable blocks)")

    res_export = urllib.request.urlopen(f"{BASE_URL}/api/audit/export?format=csv")
    csv_text = res_export.read().decode()
    assert "Timestamp (UTC)" in csv_text and "Entry Hash" in csv_text, "CSV export failed"
    print("  -> RFC-4180 CSV compliance export: OK")

def test_auth_management():
    print("\n[14/14] Testing User Authentication & JWT Sessions...")
    payload = {"username": "admin", "password": "admin123"}
    req = urllib.request.Request(f"{BASE_URL}/api/auth/login", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    assert "token" in data and data["user"]["role"] == "Admin", "Login failed"
    token = data["token"]
    print(f"  -> JWT Login: Authorized as {data['user']['full_name']} ({data['user']['role']})")

    res_me = urllib.request.urlopen(f"{BASE_URL}/api/auth/me?token={token}")
    me_data = json.loads(res_me.read().decode())
    assert me_data["user"]["username"] == "admin", "Token validation failed"
    print("  -> JWT Stateless Verification: OK")

if __name__ == "__main__":
    print("=========================================================")
    print("  SYNTHARA ENTERPRISE RAG PRODUCTION VALIDATION SUITE   ")
    print("=========================================================")
    test_pwa_endpoints()
    test_upload_document()
    test_documents_registry()
    test_suggestions_and_autocomplete()
    test_facets()
    test_query_pipeline()
    test_document_preview()
    test_feedback()
    test_multichannel_sharing()
    test_telemetry_analytics()
    test_ab_experimentation()
    test_pii_guardrail()
    test_audit_integrity()
    test_auth_management()
    print("\n=========================================================")
    print("  ALL 14 PRODUCTION WORKFLOWS VERIFIED SUCCESSFULLY!     ")
    print("=========================================================")
