"""Detached SSH signatures over an evidence package's canonical digest.

`evidence_export` produces a deterministic, Git-bound ZIP and checks that it is
internally consistent.  Internal consistency is not provenance: a partner who
receives the archive can confirm that nothing inside it moved, but not who
produced it.  ``docs/EXPORT-RESTORE.md`` said so in terms.

This module closes that gap without touching the archive.  The signed payload
is a small canonical JSON statement naming the archive's SHA-256 together with
the profile, freeze, and commit it claims to be, and the signature lives in a
sidecar file beside the archive.  Archive bytes are therefore identical with
and without signing, and the determinism gate is unaffected.

**Absence is a state, never an inference.**  An unsigned archive reports
``absent``; it is never reported as valid, and it never becomes valid by
default.  A caller that supplies an ``allowed_signers`` file has asked for
authenticity, and for that caller ``absent``, ``invalid``, ``unknown_signer``
and ``malformed`` are all failures with distinct exit codes.  A caller that
supplies no signers file gets the state reported and nothing gated, because a
signature no one asked to check is not evidence of anything.

The mechanism is `ssh-keygen -Y`, the same primitive the portfolio's release
workflows and Git itself use, so an operator needs no new key material and no
new tool.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SIGNATURE_NAMESPACE",
    "SIGNATURE_SCHEMA",
    "SIGNATURE_SUFFIX",
    "SignatureState",
    "SignatureStatus",
    "archive_digest",
    "check_signature",
    "default_allowed_signers",
    "sidecar_path",
    "sign_archive",
    "statement_matches_manifest",
]

#: Sidecar name is the archive name plus this suffix, so the pair travels
#: together and neither can be mistaken for a package member.
SIGNATURE_SUFFIX = ".sig.json"

#: `ssh-keygen -Y` namespace.  Distinct per purpose so a signature made over a
#: Git commit or a release tag can never be replayed as an evidence package
#: signature, and vice versa.
SIGNATURE_NAMESPACE = "evidence-export@permit-bearings"

SIGNATURE_SCHEMA = 1

#: `ssh-keygen -Y sign` on a 32 MiB-capped archive digest is fast; the bound
#: exists so a wedged subprocess fails rather than hanging a build.
_SUBPROCESS_TIMEOUT_SECONDS = 60

#: A signature blob and a signers file are both small.  Refuse anything larger
#: rather than reading an arbitrary file into memory.
_MAX_SIDECAR_BYTES = 64 * 1024


class SignatureState:
    """The states a package's signature can be in.

    ``ABSENT`` is deliberately first and deliberately not an error on its own:
    the project's existing unsigned archives stay valid, and the state is
    reported so a reader can see that nothing was claimed rather than assume
    something was.
    """

    ABSENT = "absent"
    VERIFIED = "verified"
    INVALID = "invalid"
    UNKNOWN_SIGNER = "unknown_signer"
    MALFORMED = "malformed"
    #: A sidecar exists and is well-formed, but no signers file was supplied,
    #: so nothing was checked.  Never reported as verified.
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class SignatureStatus:
    """What was found, what was checked, and what it does not establish."""

    state: str
    signer: str | None = None
    detail: str = ""
    archive_sha256: str | None = None
    #: The statement the sidecar says was signed, when one was readable. Kept
    #: so a caller can check it against a manifest recovered *later*, without
    #: having to open the archive before checking the signature.
    signed_statement: dict[str, Any] | None = None

    @property
    def is_verified(self) -> bool:
        return self.state == SignatureState.VERIFIED

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "signer": self.signer,
            "detail": self.detail,
            "archive_sha256": self.archive_sha256,
        }


def sidecar_path(archive: Path) -> Path:
    """Where the detached signature for ``archive`` lives."""

    return archive.with_name(archive.name + SIGNATURE_SUFFIX)


def default_allowed_signers(root: Path) -> Path | None:
    """`.github/allowed_signers` in a checkout, when it is a regular file."""

    candidate = Path(root) / ".github" / "allowed_signers"
    if candidate.is_file() and not candidate.is_symlink():
        return candidate
    return None


def archive_digest(archive: Path) -> str:
    """``sha256:<hex>`` over the archive's bytes, read in bounded chunks."""

    digest = hashlib.sha256()
    with Path(archive).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _signed_statement(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    """The bytes a signature covers.

    The digest alone would be a signature over an opaque number.  Binding the
    package identity into the same statement means a signature cannot be moved
    from one freeze to another even if two builds somehow shared a digest, and
    it means a reader of the sidecar can see what was attested without opening
    the archive.
    """

    package = manifest.get("package") if isinstance(manifest, dict) else None
    freeze = manifest.get("freeze") if isinstance(manifest, dict) else None
    profile = manifest.get("profile") if isinstance(manifest, dict) else None
    package = package if isinstance(package, dict) else {}
    freeze = freeze if isinstance(freeze, dict) else {}
    profile = profile if isinstance(profile, dict) else {}
    return {
        "signature_schema": SIGNATURE_SCHEMA,
        "namespace": SIGNATURE_NAMESPACE,
        "archive_sha256": digest,
        "package_id": package.get("package_id"),
        "archive_root": package.get("archive_root"),
        "freeze_id": freeze.get("freeze_id"),
        "frozen_on": freeze.get("frozen_on"),
        "repository_commit_sha": freeze.get("repository_commit_sha"),
        "profile_path": profile.get("path"),
        "profile_sha256": profile.get("sha256"),
    }


def _statement_bytes(statement: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            statement, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


def _ssh_keygen() -> str:
    path = shutil.which("ssh-keygen")
    if path is None:
        raise ValueError(
            "ssh-keygen was not found; evidence-package signing needs OpenSSH"
        )
    return path


def _run(argv: list[str], payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # The argv is built entirely in this module from a fixed program path and
    # values this process controls; no shell is used and nothing from the
    # archive or the sidecar reaches the command line.
    return subprocess.run(  # noqa: S603  # nosec B603
        argv,
        input=payload,
        capture_output=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def _signer_principal(key_path: Path) -> str:
    """The principal recorded in the sidecar, from the key's own comment.

    A principal is a claim about which line of an ``allowed_signers`` file
    should apply.  It is never trusted on its own: `ssh-keygen -Y verify`
    fails when the signature was not made by a key listed for that principal,
    so a sidecar naming a principal it did not sign for cannot verify.
    """

    result = _run([_ssh_keygen(), "-l", "-f", str(key_path)], b"")
    if result.returncode != 0:
        raise ValueError(
            f"{key_path}: could not read the signing key "
            f"({result.stderr.decode('utf-8', 'replace').strip()})"
        )
    fields = result.stdout.decode("utf-8", "replace").strip().split(" ")
    # `ssh-keygen -l` prints "<bits> <fingerprint> <comment> (<type>)"; the
    # comment is conventionally the signer's email.
    if len(fields) >= 3 and fields[2] != "no":
        return fields[2]
    raise ValueError(
        f"{key_path}: the signing key has no comment to use as a principal; "
        "pass --signer-identity"
    )


def sign_archive(
    archive: Path,
    manifest: dict[str, Any],
    *,
    key_path: Path,
    signer_identity: str | None = None,
) -> dict[str, Any]:
    """Sign ``archive``'s canonical digest and write the sidecar beside it.

    The archive is not opened for writing and not rewritten.  Returns the
    sidecar payload.
    """

    archive = Path(archive)
    key_path = Path(key_path)
    if not key_path.is_file():
        raise ValueError(f"{key_path}: expected a readable private signing key")
    digest = archive_digest(archive)
    statement = _signed_statement(manifest, digest)
    payload = _statement_bytes(statement)
    identity = signer_identity or _signer_principal(key_path)

    result = _run(
        [
            _ssh_keygen(),
            "-Y",
            "sign",
            "-q",
            "-f",
            str(key_path),
            "-n",
            SIGNATURE_NAMESPACE,
            "-",
        ],
        payload,
    )
    if result.returncode != 0:
        raise ValueError(
            "evidence package signing failed: "
            + result.stderr.decode("utf-8", "replace").strip()
        )
    sidecar = {
        "signature_schema": SIGNATURE_SCHEMA,
        "namespace": SIGNATURE_NAMESPACE,
        "signer_identity": identity,
        "signed_statement": statement,
        "signature": result.stdout.decode("utf-8"),
    }
    target = sidecar_path(archive)
    _write_sidecar(target, sidecar)
    return sidecar


def _write_sidecar(target: Path, sidecar: dict[str, Any]) -> None:
    if target.exists() or target.is_symlink():
        raise ValueError(f"{target}: refusing to overwrite an existing signature")
    body = (
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=".evidence-signature-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
        os.chmod(temporary, 0o644)
        # `link` then `unlink` rather than `replace`, so a signature that
        # appeared concurrently is never silently overwritten.
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise ValueError(
                f"{target}: refusing to overwrite an existing signature"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_sidecar(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a regular signature file")
    if path.stat().st_size > _MAX_SIDECAR_BYTES:
        raise ValueError(f"{path}: signature file exceeds the size limit")
    loaded = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: signature file is not a JSON object")
    return loaded


def check_signature(
    archive: Path,
    manifest: dict[str, Any] | None = None,
    *,
    allowed_signers: Path | None = None,
) -> SignatureStatus:
    """Report the archive's signature state without ever inferring validity.

    ``allowed_signers`` is what turns reporting into checking.  Without it a
    present signature is ``not_checked``, because a signature verified against
    no declared set of signers establishes nothing about who produced the
    package.

    ``manifest`` is optional so this can run *before* the archive is opened.
    A single altered byte changes the digest and is reported here as a broken
    signature, rather than surfacing later as a changed member -- which is what
    a partner asking "did this come from you unchanged" needs to hear first.
    Pass :func:`statement_matches_manifest` the manifest afterwards to confirm
    the signed statement describes the package the verifier actually read.
    """

    archive = Path(archive)
    digest = archive_digest(archive)
    target = sidecar_path(archive)
    if not target.exists() and not target.is_symlink():
        return SignatureStatus(
            state=SignatureState.ABSENT,
            detail=(
                "no detached signature accompanies this archive; its hashes "
                "show that nothing inside it moved, and say nothing about who "
                "produced it"
            ),
            archive_sha256=digest,
        )
    try:
        sidecar = _read_sidecar(target)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return SignatureStatus(
            state=SignatureState.MALFORMED,
            detail=f"the signature file could not be read: {error}",
            archive_sha256=digest,
        )

    claimed = sidecar.get("signed_statement")
    signer = sidecar.get("signer_identity")
    signature = sidecar.get("signature")
    if not isinstance(signer, str) or not isinstance(signature, str):
        return SignatureStatus(
            state=SignatureState.MALFORMED,
            detail="the signature file is missing its signer identity or signature",
            archive_sha256=digest,
        )
    if not isinstance(claimed, dict):
        return SignatureStatus(
            state=SignatureState.MALFORMED,
            signer=signer,
            detail="the signature file carries no signed statement",
            archive_sha256=digest,
        )
    if claimed.get("archive_sha256") != digest:
        # Reported as invalid, not malformed: the file is well-formed and
        # attests to a different archive than this one. This is the branch a
        # single altered byte takes, and it is reached before the archive is
        # opened.
        return SignatureStatus(
            state=SignatureState.INVALID,
            signer=signer,
            detail=(
                "the signed statement covers a different archive digest, so "
                "these bytes are not the bytes that were signed"
            ),
            archive_sha256=digest,
            signed_statement=claimed,
        )
    if manifest is not None and claimed != _signed_statement(manifest, digest):
        return SignatureStatus(
            state=SignatureState.INVALID,
            signer=signer,
            detail=(
                "the signed statement does not describe this package's "
                "manifest; the signature covers a different freeze or profile"
            ),
            archive_sha256=digest,
            signed_statement=claimed,
        )
    if allowed_signers is None:
        return SignatureStatus(
            state=SignatureState.NOT_CHECKED,
            signer=signer,
            detail=(
                "a signature is present and describes this archive, but no "
                "allowed-signers file was supplied, so no signer was checked"
            ),
            archive_sha256=digest,
            signed_statement=claimed,
        )
    return _verify_with_openssh(
        claimed, signature, signer, Path(allowed_signers), digest
    )


def statement_matches_manifest(
    status: SignatureStatus, manifest: dict[str, Any]
) -> bool:
    """Does the signed statement describe the manifest the verifier read?

    Called after structural verification, so a signature that checks out over
    the right bytes still has to describe the right package.
    """

    if status.signed_statement is None or status.archive_sha256 is None:
        return False
    return status.signed_statement == _signed_statement(manifest, status.archive_sha256)


def _allowed_principals(allowed_signers: Path) -> list[str]:
    """The principal patterns an ``allowed_signers`` file declares.

    Format is ``principals [options] keytype key [comment]``; principals is a
    comma-separated list of patterns. Only this first field is parsed, and only
    to answer "is this identity listed at all".
    """

    patterns: list[str] = []
    for raw in allowed_signers.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        field = line.split(None, 1)[0].strip('"')
        patterns.extend(part for part in field.split(",") if part)
    return patterns


def _is_listed(signer: str, patterns: list[str]) -> bool:
    """Whether OpenSSH would consider ``signer`` named by any pattern.

    Negations (``!pattern``) veto, matching OpenSSH's own precedence. A file
    that lists no patterns at all lists nobody: an empty signers file accepts
    no one, and must never read as "cannot tell, so allow".
    """

    matched = False
    for pattern in patterns:
        if pattern.startswith("!"):
            if fnmatch.fnmatchcase(signer, pattern[1:]):
                return False
        elif fnmatch.fnmatchcase(signer, pattern):
            matched = True
    return matched


def _verify_with_openssh(
    statement: dict[str, Any],
    signature: str,
    signer: str,
    allowed_signers: Path,
    digest: str,
) -> SignatureStatus:
    if allowed_signers.is_symlink() or not allowed_signers.is_file():
        raise ValueError(f"{allowed_signers}: expected a regular allowed-signers file")
    # Decided here rather than by reading OpenSSH's stderr. `ssh-keygen -Y
    # verify` prints the same "Could not verify signature." for an unlisted
    # principal as for a signature that does not check out, and a partner needs
    # those told apart: one is "we do not accept this producer", the other is
    # "these bytes are not what was signed".
    if not _is_listed(signer, _allowed_principals(allowed_signers)):
        return SignatureStatus(
            state=SignatureState.UNKNOWN_SIGNER,
            signer=signer,
            detail=(
                f"{signer} is not listed in {allowed_signers}, so this "
                "signature establishes nothing here"
            ),
            archive_sha256=digest,
            signed_statement=statement,
        )
    with tempfile.TemporaryDirectory(prefix="evidence-signature-") as workspace:
        signature_file = Path(workspace) / "signature"
        signature_file.write_text(signature, encoding="utf-8")
        result = _run(
            [
                _ssh_keygen(),
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                signer,
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                str(signature_file),
            ],
            _statement_bytes(statement),
        )
    if result.returncode == 0:
        return SignatureStatus(
            state=SignatureState.VERIFIED,
            signer=signer,
            detail=(
                f"signed by {signer}, listed in {allowed_signers}. This "
                "attests who produced the package; it is not an endorsement "
                "of its contents"
            ),
            archive_sha256=digest,
            signed_statement=statement,
        )
    message = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
    return SignatureStatus(
        state=SignatureState.INVALID,
        signer=signer,
        detail=f"the signature did not verify ({message})",
        archive_sha256=digest,
        signed_statement=statement,
    )
