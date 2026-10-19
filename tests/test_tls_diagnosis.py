"""A TLS failure names itself: the pin the box no longer presents, or why its certificate did not verify.

Issue #90 (2026-10-18, a site on 0.44.2): PVE renewed a node's certificate on its own, the pin
broke, and every call read "ConnectError" until an operator ran openssl by hand. On 0.45 both doors
already print the pin error's text; the doctor and every ledger row kept only the type, because
`spine.wire_error` drops the message (it carries the request URL). These tests drive real TLS on a
loopback socket through the doctor, end to end, never a built exception chain: the shape is what
httpx makes of what ssl raises, and only the wire knows it.
"""
from __future__ import annotations

import datetime
import hashlib
import http.server
import ipaddress
import json
import ssl
import threading

import pytest

x509 = pytest.importorskip("cryptography.x509")
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from proximo.core import doctor  # noqa: E402

_DAY = datetime.timedelta(days=1)


def _cert(name: str, *, san_ip: str, issuer=None, ca: bool = False):
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.UTC)
    signer_key, signer_name = (issuer[1], issuer[0].subject) if issuer else (key, subject)
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(signer_name)
         .public_key(key.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - _DAY).not_valid_after(now + _DAY)
         .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(san_ip))]), False)
         # SKI and AKI as PVE writes them: without them a strict verifier stops at "Missing Authority
         # Key Identifier" before it ever reaches the keyUsage question #90 is about.
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(signer_key.public_key()), False))
    if ca:
        # PVE's own root: CA:TRUE, critical, and NO keyUsage (the shape #90 measured on a stock cluster).
        b = b.add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    return b.sign(signer_key, hashes.SHA256()), key


def _pem(cert) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


class _Version(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 -- the stdlib's name
        body = b'{"data":{"version":"9.2","release":"9.2"}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def serve(tmp_path):
    """Serve `/version` over TLS on a loopback port with the given leaf; yield its base URL."""
    servers = []

    def start(leaf, key, chain=()):
        crt, k = tmp_path / f"leaf{len(servers)}.pem", tmp_path / f"key{len(servers)}.pem"
        crt.write_bytes(_pem(leaf) + b"".join(_pem(c) for c in chain))
        k.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Version)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, k)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"https://127.0.0.1:{srv.server_address[1]}/api2/json"

    yield start
    for s in servers:
        s.shutdown()


@pytest.fixture
def plane(tmp_path, monkeypatch):
    """Configure the env pve plane at `url`, with a pin or a CA bundle, and run the doctor."""
    token = tmp_path / "token"
    token.write_text("tls@pve!t=" + "0" * 64 + "\n")
    token.chmod(0o600)

    def run(url, *, pin=None, ca=None, strict=None):
        monkeypatch.setenv("PROXIMO_API_BASE_URL", url)
        monkeypatch.setenv("PROXIMO_TOKEN_PATH", str(token))
        for var, val in (("PROXIMO_FINGERPRINT", pin), ("PROXIMO_CA_BUNDLE", ca), ("PROXIMO_TLS_STRICT", strict)):
            if val:
                monkeypatch.setenv(var, val)
            else:
                monkeypatch.delenv(var, raising=False)
        report = doctor.run()
        flag = next((f for f in report["flags"] if f.startswith("pve: no reach")), None)
        return report["planes"]["pve"]["reach"]["version"], flag, report

    return run


def test_a_pin_the_box_no_longer_presents_names_both_fingerprints(serve, plane):
    leaf, key = _cert("pve0", san_ip="127.0.0.1")
    url = serve(leaf, key)
    presented = hashlib.sha256(leaf.public_bytes(serialization.Encoding.DER)).hexdigest()
    version, flag, _ = plane(url, pin="AB:" * 31 + "AB")
    assert version["tls"] == "pin mismatch"
    assert version["observed"] == presented and version["pinned"] == "ab" * 32
    assert presented in flag and "ab" * 32 in flag and "re-pin" in flag, flag


def test_the_right_pin_reaches_the_box(serve, plane):
    leaf, key = _cert("pve0", san_ip="127.0.0.1")
    url = serve(leaf, key)
    version, flag, _ = plane(url, pin=hashlib.sha256(leaf.public_bytes(serialization.Encoding.DER)).hexdigest())
    assert version == "9.2" and flag is None


def test_an_untrusted_certificate_says_why_it_did_not_verify(serve, plane, tmp_path):
    leaf, key = _cert("pve0", san_ip="127.0.0.1")
    other, _ = _cert("someone-else", san_ip="127.0.0.1", ca=True)
    url = serve(leaf, key)
    (tmp_path / "other.pem").write_bytes(_pem(other))
    version, flag, _ = plane(url, ca=str(tmp_path / "other.pem"))
    assert version["tls"] == "certificate not verified"
    assert isinstance(version["verify_code"], int) and version["verify_message"], version
    assert version["verify_message"] in flag, flag


def test_the_stock_pve_root_is_named_when_a_strict_interpreter_refuses_it(serve, plane, tmp_path):
    """The #90 case itself: a CA bundle holding PVE's root (CA:TRUE, no keyUsage). Where the
    interpreter verifies strictly (3.13 sets VERIFY_X509_STRICT by default) the doctor says the
    reason OpenSSL gives; where it does not (3.12), the same bundle simply reaches the box."""
    root, root_key = _cert("pve-root-ca", san_ip="127.0.0.1", ca=True)
    leaf, key = _cert("pve0", san_ip="127.0.0.1", issuer=(root, root_key))
    url = serve(leaf, key)
    (tmp_path / "pve-root-ca.pem").write_bytes(_pem(root))
    version, flag, _ = plane(url, ca=str(tmp_path / "pve-root-ca.pem"))
    if ssl.create_default_context().verify_flags & ssl.VERIFY_X509_STRICT:
        assert version["tls"] == "certificate not verified"
        assert "key usage" in version["verify_message"], version
        assert "key usage" in flag, flag
    else:
        assert version == "9.2" and flag is None


def test_a_name_mismatch_keeps_the_code_and_never_the_target(serve, plane, tmp_path):
    """The axis the other cases do not vary: OpenSSL's hostname and IP mismatch messages are the
    ones Python appends the target to. A cert for another address, trusted as its own anchor, is
    refused by name; the row keeps the code and no part of the address it dialed."""
    leaf, key = _cert("elsewhere", san_ip="192.0.2.10")
    url = serve(leaf, key)
    (tmp_path / "elsewhere.pem").write_bytes(_pem(leaf))
    version, flag, report = plane(url, ca=str(tmp_path / "elsewhere.pem"))
    assert version["tls"] == "certificate not verified"
    assert version["verify_code"] in (62, 64) and "verify_message" not in version, version
    assert "127.0.0.1" not in json.dumps(version) and "127.0.0.1" not in flag, flag


# ---- tls_strict = false: the operator's opt-in for PVE's own root (John's call on #90, 2026-10-19) ----

def _stock_root(serve, tmp_path):
    root, root_key = _cert("pve-root-ca", san_ip="127.0.0.1", ca=True)
    leaf, key = _cert("pve0", san_ip="127.0.0.1", issuer=(root, root_key))
    (tmp_path / "pve-root-ca.pem").write_bytes(_pem(root))
    return serve(leaf, key), str(tmp_path / "pve-root-ca.pem")


def test_tls_strict_false_trusts_the_stock_pve_root_and_says_so(serve, plane, tmp_path):
    url, root = _stock_root(serve, tmp_path)
    version, flag, report = plane(url, ca=root, strict="false")
    assert version == "9.2" and flag is None, (version, flag)
    assert report["planes"]["pve"]["tls_strict"] is False
    assert any("tls_strict=false" in f for f in report["flags"]), report["flags"]


def test_tls_strict_false_still_checks_the_name(serve, plane, tmp_path):
    """Strictness off is not verification off: a certificate for another address is refused."""
    leaf, key = _cert("elsewhere", san_ip="192.0.2.10")
    url = serve(leaf, key)
    (tmp_path / "elsewhere.pem").write_bytes(_pem(leaf))
    version, _, _ = plane(url, ca=str(tmp_path / "elsewhere.pem"), strict="false")
    assert version["tls"] == "certificate not verified" and version["verify_code"] in (62, 64), version


@pytest.mark.parametrize("pin", [None, "AB:" * 31 + "AB"])
def test_tls_strict_false_without_a_ca_bundle_refuses_instead_of_meaning_nothing(serve, plane, pin):
    leaf, key = _cert("pve0", san_ip="127.0.0.1")
    url = serve(leaf, key)
    version, _, _ = plane(url, pin=pin, strict="false")
    assert version["error"] == "refused" and "TLS_STRICT" in version["reason"], version


def test_a_target_reads_tls_strict_from_its_toml_bool(tmp_path):
    from proximo.core import wire  # noqa: PLC0415
    token = tmp_path / "token"
    token.write_text("t@pve!x=" + "0" * 64 + "\n")
    token.chmod(0o600)
    base = {"base_url": "https://192.0.2.10:8006/api2/json", "token_path": str(token), "ca_bundle": "x.pem"}
    assert wire.plane_from_target("pve", "edge", {**base, "tls_strict": False}).tls_strict is False
    assert wire.plane_from_target("pve", "edge", base).tls_strict is True


# ---- the lens of 2026-10-19: a message is text anyone could build; the row reads a table ----

def _verify_error(code, message):
    e = ssl.SSLCertVerificationError(1, "certificate verify failed")
    e.verify_code, e.verify_message = code, message
    return e


def test_an_unknown_code_keeps_the_code_and_never_the_message():
    from proximo.core.spine import wire_error  # noqa: PLC0415
    row = wire_error(_verify_error(99, "Hostname mismatch, certificate is not valid for 'node7.example.invalid'."))
    assert row == {"error": "SSLCertVerificationError", "tls": "certificate not verified", "verify_code": 99}


def test_a_known_code_carries_the_openssl_reason_not_the_exception_text():
    from proximo.core.spine import wire_error  # noqa: PLC0415
    row = wire_error(_verify_error(92, "planted: node7.example.invalid"))
    assert row["verify_message"] == "CA cert does not include key usage extension", row


@pytest.mark.parametrize("code", [None, True, "92"])
def test_a_code_that_is_not_an_int_is_no_code(code):
    from proximo.core.spine import wire_error  # noqa: PLC0415
    row = wire_error(_verify_error(code, "planted: node7.example.invalid"))
    assert "verify_code" not in row and "verify_message" not in row, row


def test_a_hand_built_error_with_no_attributes_or_one_that_raises_still_leaves_a_row():
    from proximo.core.spine import wire_error  # noqa: PLC0415

    class Raises(ssl.SSLCertVerificationError):
        @property
        def verify_code(self):
            raise RuntimeError("a property that will not say")

    assert wire_error(ssl.SSLCertVerificationError("boom"))["tls"] == "certificate not verified"
    assert wire_error(Raises("boom")) == {"error": "Raises"}
