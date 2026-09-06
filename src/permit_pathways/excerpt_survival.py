"""Whether the words a rule quotes still appear after its source changed.

The weekly watch classifies a source by content hash, and a moved hash stales
every rule that depends on that source. That is correct and deliberately
coarse: a footer edit on the HCD Handbook stales seventeen rules exactly as a
rewritten height section would. This module adds the finer, purely mechanical
question a maintainer asks next — *does the text this rule quotes still occur
in the new document, word for word?* — so the answer is computed rather than
guessed at from a diff.

Three things it deliberately does not do:

* **It un-stales nothing.** A rule whose excerpt survives is still on hold
  until a person re-verifies it. Surrounding text can change what the same
  sentence means, and no string test can see that. Survival reorders a
  worklist; it never clears one.
* **It never reports a guess as a finding.** A source whose bytes cannot be
  turned into text — a scanned image, an unknown media type, a PDF with the
  ``ai`` extra absent — is ``not_checkable``, carrying the reason. That is the
  same discipline the watch already applies to a fetch it could not make: a
  check that did not run must never read as a check that passed.
* **It does not re-implement matching.** Normalization is
  :func:`permit_pathways.ai.corpus.normalize_for_match`, the same function the
  AI layer's citation verifier uses, so "occurs verbatim" means one thing in
  this project rather than two.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .ai.corpus import CorpusError, extract_text, normalize_for_match

if TYPE_CHECKING:
    from .screening import Rule

ExcerptSurvivalStatus = Literal["excerpt_survives", "excerpt_lost", "not_checkable"]

EXCERPT_SURVIVAL_STATUSES: frozenset[str] = frozenset(
    {"excerpt_survives", "excerpt_lost", "not_checkable"}
)

# Suffixes `ai.corpus.extract_text` can turn into text. Anything else is
# `not_checkable` by media type rather than by a failed attempt.
_EXTRACTABLE_SUFFIXES = frozenset({".html", ".htm", ".txt", ".pdf"})


@dataclass(frozen=True)
class RuleExcerptSurvival:
    """One rule's excerpt, held against the new text of a source it cites."""

    rule_id: str
    status: ExcerptSurvivalStatus
    # Set only when ``status`` is ``not_checkable``: why the question could
    # not be asked. A survived or lost result is the answer itself and needs
    # no explanation.
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"rule_id": self.rule_id, "status": self.status}
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


def _excerpt_of(rule: Rule) -> str | None:
    excerpt = rule.citation.excerpt
    if not isinstance(excerpt, str) or not excerpt.strip():
        return None
    return excerpt


def rules_depending_on(source_id: str, rules: Iterable[Rule]) -> tuple[Rule, ...]:
    """Every rule that names ``source_id`` in ``source_dependencies``."""

    matched = [rule for rule in rules if source_id in rule.source_dependencies]
    return tuple(sorted(matched, key=lambda rule: rule.rule_id))


def survival_for_source(
    source_id: str,
    source_url: str,
    rules: Iterable[Rule],
    *,
    new_text: str | None,
    previous_text: str | None = None,
    not_checkable_reason: str | None = None,
) -> tuple[RuleExcerptSurvival, ...]:
    """Hold every dependent rule's excerpt against a source's new text.

    ``source_url`` is the watched source's registry key, and it is required
    rather than optional on purpose. **Depending on a source is not the same
    as quoting it.** Thirteen of the fifteen rules that depend on the HCD ADU
    Handbook quote a statute instead and merely lean on the Handbook for
    context; their excerpts were never in that PDF, so testing them against it
    would report thirteen rules as having lost text they never carried. A rule
    is given a verdict only when the changed document is the one it quotes —
    the same "cites" versus "depends on" line ``citationSourceId()`` already
    draws in the browser. The rest are ``not_checkable``: still stale, because
    the source they depend on moved, but with nothing here to say about them.

    ``previous_text`` is the retained copy the recorded digest was taken from,
    and it is what makes ``excerpt_lost`` mean anything. This project's
    excerpts are curated citations, not raw quotations: many carry editorial
    brackets condensing a list, so they do not occur verbatim in the source
    even now. Measured over the committed corpus, twelve of nineteen would
    report ``excerpt_lost`` against a document that had not changed at all. So
    a verdict is issued only for an excerpt that occurs verbatim in the
    retained copy *first*; the rest are ``not_checkable``. ``excerpt_lost``
    therefore means "this text was there and is not any more" — a fact about
    the change — rather than "this text is not there", which was already true.

    ``new_text`` is ``None`` when the document could not be read at all. Every
    dependent rule is then ``not_checkable`` with ``not_checkable_reason`` —
    never ``excerpt_lost``, which would report "the words are gone" on the
    strength of never having looked.
    """

    dependents = rules_depending_on(source_id, rules)
    if new_text is None:
        reason = not_checkable_reason or "the changed document could not be read"
        return tuple(
            RuleExcerptSurvival(
                rule_id=rule.rule_id,
                status="not_checkable",
                reason=reason,
            )
            for rule in dependents
        )

    haystack = normalize_for_match(new_text)
    baseline = normalize_for_match(previous_text) if previous_text is not None else None
    results: list[RuleExcerptSurvival] = []
    for rule in dependents:
        rule_id = rule.rule_id
        if rule.citation.url != source_url:
            results.append(
                RuleExcerptSurvival(
                    rule_id=rule_id,
                    status="not_checkable",
                    reason=(
                        "the rule depends on this source but quotes a different "
                        "one, so its excerpt is not in this document"
                    ),
                )
            )
            continue
        excerpt = _excerpt_of(rule)
        if excerpt is None:
            results.append(
                RuleExcerptSurvival(
                    rule_id=rule_id,
                    status="not_checkable",
                    reason="the rule quotes no excerpt from this source",
                )
            )
            continue
        needle = normalize_for_match(excerpt)
        if not needle:
            results.append(
                RuleExcerptSurvival(
                    rule_id=rule_id,
                    status="not_checkable",
                    reason="the excerpt normalizes to nothing matchable",
                )
            )
            continue
        if baseline is None:
            results.append(
                RuleExcerptSurvival(
                    rule_id=rule_id,
                    status="not_checkable",
                    reason=(
                        "no retained copy was available to establish what this "
                        "excerpt looked like before the change"
                    ),
                )
            )
            continue
        if needle not in baseline:
            # Not a defect in the rule: an excerpt may legitimately be an
            # edited citation. It just cannot be tracked by verbatim survival,
            # and saying `excerpt_lost` about it would report the editing as
            # if the source had dropped the text.
            results.append(
                RuleExcerptSurvival(
                    rule_id=rule_id,
                    status="not_checkable",
                    reason=(
                        "the recorded excerpt is an edited citation rather than "
                        "a verbatim quote of the retained copy, so verbatim "
                        "survival cannot be tested"
                    ),
                )
            )
            continue
        results.append(
            RuleExcerptSurvival(
                rule_id=rule_id,
                status="excerpt_survives" if needle in haystack else "excerpt_lost",
            )
        )
    return tuple(results)


def text_from_bytes(payload: bytes, *, suffix: str) -> tuple[str | None, str | None]:
    """Extract text from freshly fetched bytes.

    Returns ``(text, None)`` on success and ``(None, reason)`` when the bytes
    are not something this project can read. The reason is carried into
    ``not_checkable`` so a maintainer is told which of the two it was.
    """

    normalized_suffix = suffix.lower()
    if normalized_suffix not in _EXTRACTABLE_SUFFIXES:
        return None, f"no text extractor for {normalized_suffix or 'an unnamed type'}"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"fetched{normalized_suffix}"
        path.write_bytes(payload)
        try:
            text = extract_text(path)
        except CorpusError as error:
            return None, str(error)
        except Exception as error:
            # A malformed PDF or undecodable payload is a fact about the
            # document, not a reason to end the watch. It is reported as
            # `not_checkable`, which is what it is.
            return None, f"the changed document could not be read: {error}"
    return (text, None) if text.strip() else (None, "the document extracted no text")


def counts(survival: Sequence[RuleExcerptSurvival]) -> dict[str, int]:
    """Survived / lost / not-checkable totals, for a one-line report."""

    tally = dict.fromkeys(sorted(EXCERPT_SURVIVAL_STATUSES), 0)
    for item in survival:
        tally[item.status] += 1
    return tally


def summarize(source_id: str, survival: Sequence[RuleExcerptSurvival]) -> str:
    """One line per changed source, then one line per rule that lost its text."""

    tally = counts(survival)
    lines = [
        f"  {source_id}: {tally['excerpt_survives']} excerpt(s) survive, "
        f"{tally['excerpt_lost']} lost, {tally['not_checkable']} not checkable"
    ]
    for item in survival:
        if item.status == "excerpt_lost":
            lines.append(
                f"    EXCERPT LOST: {item.rule_id} — the text this rule quotes no "
                "longer occurs in the changed document; re-verify it first"
            )
        elif item.status == "not_checkable" and item.reason:
            lines.append(f"    not checkable: {item.rule_id} — {item.reason}")
    return "\n".join(lines)
