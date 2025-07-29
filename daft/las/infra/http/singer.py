from __future__ import annotations

import hashlib
import hmac
import logging
import posixpath
import re
import shlex
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote, unquote, unquote_plus, urlparse

if TYPE_CHECKING:
    import httpx

    from daft.las.infra.credentials import Credentials, CredentialsProvider

DATE_KEYS = ["x-date", "date"]

logger = logging.getLogger(__name__)


class VolcSigningKey:
    def __init__(
        self, secret_key: str, region: str, service: str, date: str | None = None, request: str = "request"
    ) -> None:
        self.region = region
        self.service = service
        self.date = date or datetime.utcnow().strftime("%Y%m%d")
        self.scope = f"{self.date}/{self.region}/{self.service}/{request}"
        self.secret_key = secret_key
        self.key = self.generate_key(secret_key, self.region, self.service, self.date)

    @classmethod
    def generate_key(cls, secret_key: str, region: str, service: str, date: str) -> bytes:
        init_key = secret_key.encode("utf-8")
        date_key = cls.sign_sha256(init_key, date)
        region_key = cls.sign_sha256(date_key, region)
        service_key = cls.sign_sha256(region_key, service)
        return cls.sign_sha256(service_key, "request")

    @staticmethod
    def sign_sha256(key: bytes, msg: bytes | str) -> bytes:
        if isinstance(msg, str):
            msg = msg.encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).digest()


class Singer:
    def __init__(
        self,
        service: str,
        region: str,
        credential_provider: CredentialsProvider,
        signing_algorithm: str = "HMAC-SHA256",
        date_key: str = "x-date",
        content_hash_key: str = "x-content-sha256",
        signed_headers: set[str] | None = None,
        date: str | None = None,
    ) -> None:
        self.service = service
        self.region = region
        self.credential_provider = credential_provider

        self.signing_algorithm = signing_algorithm
        self.date_key = date_key
        self.content_hash_key = content_hash_key
        self.signed_headers = signed_headers or {"host", "content-type", "date", "x-*"}

        credentials = self._get_credentials()
        self.signing_key = VolcSigningKey(credentials.secret_key, region, service, date)
        if credentials.session_token:
            self.signed_headers.add("x-security-token")

    def _get_credentials(self) -> Credentials:
        credential = self.credential_provider.get_credentials()
        if credential is None:
            raise RuntimeError("Cannot found valid credentials.")
        return credential

    def sign_request(self, req: httpx.Request) -> httpx.Request:
        """Generate signature of request.

        According to the VolcanoEngine signature algorithm https://www.volcengine.com/docs/6369/67269.

        """
        req_url = str(req.url)
        parsed_url = urlparse(req_url)
        headers = req.headers

        scope_date = self.extract_scope_date(headers)

        content = req.content if req.content is not None else b""
        payload_hash = hashlib.sha256(content).hexdigest()
        headers[self.content_hash_key] = payload_hash

        credential = self._get_credentials()
        if credential.session_token:
            headers["x-security-token"] = credential.session_token

        if "host" in headers:
            headers["host"] = parsed_url.netloc.split(":")[0]

        cano_headers, signed_headers = self.extract_canonical_headers(headers)
        cano_req = self.gen_canonical_string(req_url, req.method, cano_headers, signed_headers, payload_hash)
        scope = self.signing_key.scope
        sig_string = self.get_sig_string(scope_date, cano_req, scope)
        hsh = hmac.new(self.signing_key.key, sig_string, hashlib.sha256)
        sig = hsh.hexdigest()

        auth_str = f"{self.signing_algorithm} "
        auth_str += f"Credential={credential.access_key}/{scope}, "
        auth_str += f"SignedHeaders={signed_headers}, "
        auth_str += f"Signature={sig}"
        headers["Authorization"] = auth_str

        return req

    def gen_scope(self, date: str) -> str:
        return "/".join([date, self.region, self.service, "request"])

    def get_sig_string(self, date: str, cano_req: str, scope: str) -> bytes:
        hsh = hashlib.sha256(cano_req.encode())
        sig_items = [self.signing_algorithm, date, scope, hsh.hexdigest()]
        sig_string = "\n".join(sig_items)
        return sig_string.encode("utf-8")

    def gen_canonical_string(
        self, url: str, method: str, cano_headers: str, signed_headers: str, payload_hash: str
    ) -> str:
        r"""Generate a canonical string for signing.

        The pattern is:

        HTTPRequestMethod + '\n' +
        CanonicalURI + '\n' +
        CanonicalQueryString + '\n' +
        CanonicalHeaders + '\n' +
        SignedHeaders + '\n' +
        HexEncode(Hash(RequestPayload))
        """
        path = self.volc_cano_path(urlparse(url).path)

        split = url.split("?", 1)
        qs = split[1] if len(split) == 2 else ""
        qs = self.volc_cano_querystring(qs)

        req_parts = [method.upper(), path, qs, cano_headers, signed_headers]
        if payload_hash:
            req_parts.append(payload_hash)

        cano_req = "\n".join(req_parts)
        return cano_req

    def volc_cano_path(self, path: str) -> str:
        safe_chars = "/~"
        qs = ""
        fixed_path = path

        if "?" in fixed_path:
            fixed_path, qs = fixed_path.split("?", 1)
        fixed_path = posixpath.normpath(fixed_path)
        fixed_path = re.sub("/+", "/", fixed_path)

        if path.endswith("/") and not fixed_path.endswith("/"):
            fixed_path += "/"
        full_path = fixed_path

        # S3 seems to require unquoting first. "host" service is used in
        if self.service in ["s3", "host"]:
            full_path = unquote(full_path)
        full_path = quote(full_path, safe=safe_chars)

        if qs:
            full_path = "?".join((full_path, qs))

        return full_path

    def extract_canonical_headers(self, headers: httpx.Headers | dict[str, str]) -> tuple[str, str]:
        # extract headers need to be signed
        cano_headers_dict: dict[str, list[str]] = {}
        for hdr, val in headers.items():
            hdr = hdr.strip().lower()
            val = self.volc_norm_whitespace(val).strip()

            if (
                hdr in self.signed_headers
                or "*" in self.signed_headers
                or ("x-*" in self.signed_headers and hdr.startswith("x-") and not hdr == "x-client-context")
            ):
                vals = cano_headers_dict.setdefault(hdr, [])
                vals.append(val)

        # sorted signed headers
        cano_headers = ""
        signed_headers_list = []
        for hdr in sorted(cano_headers_dict):
            vals = cano_headers_dict[hdr]
            val = ",".join(sorted(vals))

            cano_headers += f"{hdr}:{val}\n"
            signed_headers_list.append(hdr)

        return cano_headers, ";".join(signed_headers_list)

    @staticmethod
    def volc_cano_querystring(qs: str) -> str:
        safe_qs_chars = "&="
        safe_qs_unresvd = "-_.~"

        qs = unquote_plus(qs)
        qs = quote(qs, safe=safe_qs_chars)

        qs_items = {}
        for name, vals in parse_qs(qs, keep_blank_values=True).items():
            name = quote(name, safe=safe_qs_unresvd)
            vals = [quote(val, safe=safe_qs_unresvd) for val in vals]
            qs_items[name] = vals

        qs = ""
        for key in sorted(qs_items.keys()):
            if isinstance(qs_items[key], list):
                for k in qs_items[key]:
                    qs = qs + key + "=" + k + "&"
            else:
                qs = qs + key + "=" + str(qs_items[key]) + "&"

        qs = qs[:-1]
        return qs.replace("+", "%20")

    @staticmethod
    def volc_norm_whitespace(text: str) -> str:
        """Replace runs of whitespace with a single space.

        Ignore text enclosed in quotes.
        """
        return " ".join(shlex.split(text, posix=False))

    def extract_scope_date(self, headers: httpx.Headers | dict[str, str]) -> str:
        def reset_and_get_date_header() -> str:
            date_val = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            for header in DATE_KEYS:
                if header in headers:
                    date_val = headers[header]
                    break

            # Clear all date headers
            for header in DATE_KEYS:
                if header in headers:
                    headers.pop(header)

            # Rewrite date header
            headers[self.date_key] = date_val
            return date_val

        # Rewrite scope date if doesn't match with header date.
        req_date = reset_and_get_date_header()
        scope_date = self.parse_date(req_date) or datetime.utcnow().strftime("%Y%m%d")
        if self.signing_key.date != scope_date:
            self.signing_key = VolcSigningKey(self._get_credentials().secret_key, self.region, self.service, scope_date)

        return req_date

    @staticmethod
    def parse_date(date_str: str) -> str | None:
        """Parse a date string to '%Y%m%d' format."""
        months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        formats = {
            # RFC 7231, e.g. "Mon, 09 Sep 2011 23:36:00 GMT"
            r"^(?:\w{3}, )?(\d{2}) (\w{3}) (\d{4})\D.*$": lambda m: f"{m.group(3)}{months.index(m.group(2).lower()) + 1:02d}{m.group(1)}",
            # RFC 850 (e.g. Sunday, 06-Nov-94 08:49:37 GMT)
            # assumes current century
            r"^\w+day, (\d{2})-(\w{3})-(\d{2})\D.*$": lambda m: f"{str(datetime.today().year)[:2]}{m.group(3)}"
            f"{months.index(m.group(2).lower()) + 1:02d}{m.group(1)}",
            # C time, e.g. "Wed Dec 4 00:00:00 2002"
            r"^\w{3} (\w{3}) (\d{1,2}) \d{2}:\d{2}:\d{2} (\d{4})$": lambda m: f"{m.group(3)}{months.index(m.group(1).lower()) + 1:02d}{int(m.group(2)):02d}",
            # x-date format dates, e.g. 20100325T010101Z
            r"^(\d{4})(\d{2})(\d{2})T\d{6}Z$": lambda m: "{}{}{}".format(*m.groups()),
            # ISO 8601 / RFC 3339, e.g. "2009-03-25T10:11:12.13-01:00"
            r"^(\d{4})-(\d{2})-(\d{2})(?:[Tt].*)?$": lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}",
        }

        for regex, xform in formats.items():
            m = re.search(regex, date_str)
            if m:
                return xform(m)  # type: ignore[no-untyped-call]

        return None
