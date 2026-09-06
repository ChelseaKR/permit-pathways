"""Transit-proximity determinations from GTFS data.

Two State ADU Law standards turn on transit proximity, and both are
computable from a jurisdiction's own GTFS feed instead of applicant
self-attestation:

- Parking exemption — Gov. Code § 66322(a)(1): no parking may be required
  for an ADU "located within one-half mile walking distance of public
  transit"; § 66313(m) defines public transit broadly (any bus stop or
  train station with fixed-route, set-fare public service).
- 18-foot height allowance — Gov. Code § 66321(b)(4)(B): applies within a
  half-mile walking distance of a "major transit stop" (Pub. Res. Code
  § 21064.3: rail/BRT station, ferry with bus or rail service, or the
  intersection of two or more major bus routes with ≤20-minute peak
  service) or a "high-quality transit corridor" (Pub. Res. Code
  § 21155(b): fixed-route bus service with ≤15-minute peak intervals).

Honesty model. Distances here are straight-line (haversine). Walking
distance is never shorter than straight-line, so a supplied stop farther than
the threshold can be eliminated. That does not prove every relevant operator,
stop, or service record is present. A stop within the threshold is a
CANDIDATE "yes" pending a walking-network check (production deployments
should confirm with a router). Headways are measured within the peak windows
6-9 AM and 4-7 PM, using the maximum gap between consecutive trips - a
screening approximation of the statutes' "service interval" language, and
only as current and complete as the supplied feed.

Service dates. Both statutory screens turn on service that actually operates
(PRC § 21064.3 "existing"; § 21155(b) headways "during peak commute
periods"), so a headway is a fact about a *date*, not about a file. Every
headway here is measured over the services `calendar.txt` and
`calendar_dates.txt` say run on one stated date, and no headway is produced
without one. Earlier revisions of this module read `calendar.txt` alone,
kept whichever service_id had the most trips, and ignored `feed_info.txt`
entirely -- so a summer session, a holiday, and a feed that expired years ago
all screened identically, and the answer silently depended on which of the
feed's service periods happened to be larger. That is a measurement standing
in for a measurement nobody made. Four states now withhold instead:

- ``no_as_of``            no date was supplied, so no service set is defined;
- ``outside_feed_window`` the date falls outside `feed_info.txt` validity;
- ``no_calendar``         the feed ships neither calendar file;
- ``no_service_on_date``  the calendar is readable and says nothing runs.

Only the last is a negative the feed can support: it is the feed answering,
not the feed being unreadable. In the other three the feed-derived screens
report `unknown`, never `no`. A candidate that rests on the separate
statewide Caltrans dataset is unaffected, because that dataset carries its
own currency and is not scoped by this feed's calendar.

Planned facilities. The statewide Caltrans dataset mixes two different
statutory definitions in one column, and `hqta_details` is the only field
that tells them apart. Cal-ITP's published methodology for the dataset
(https://github.com/cal-itp/data-analyses/tree/main/high_quality_transit_areas,
README.md and technical_notes.md, both retrieved 2026-08-27) lists planned
major stops under "Planned Major Stops (future service, provided by MPOs)"
and states that they "must be included in the *currently adopted* regional
transportation plan", that "the only statutory criteria for including these
stops is that they are included in the RTP", and that "Caltrans does not
validate or further process them." The same README quotes PRC § 21155's
definition, "A major transit stop is as defined in Section 21064.3, except
that, for purposes of this section, it also includes major transit stops
that are included in the applicable regional transportation plan", and
PRC § 21064.3(a), "An existing rail or bus rapid transit station."

So a row marked `mpo_rtp_planned_major_stop` is a major transit stop for
§ 21155 and is not established as one by § 21064.3. This module does not
decide which definition a given standard incorporates. It refuses to
collapse the two: a planned row never produces a candidate, and it is
always reported by count, agency, type and distance so the reader can ask
the transit agency whether the facility is in service.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

HALF_MILE = 0.5
HQTC_MAX_GAP_MIN = 15  # PRC § 21155(b)
MAJOR_STOP_MAX_GAP_MIN = 20  # PRC § 21064.3(c)
PEAKS = ((6 * 60, 9 * 60), (16 * 60, 19 * 60))
STOP_CLUSTER_MILES = 0.1  # stops this close count as one "intersection"
RAIL_ROUTE_TYPES = {"0", "1", "2"}  # tram, metro, rail
FERRY_ROUTE_TYPES = {"4"}
BUS_ROUTE_TYPES = {"3", "11"}  # bus and trolleybus
# The one documented `hqta_details` value marking a row an MPO submitted as
# planned in its adopted regional transportation plan rather than one Caltrans
# derived from published GTFS service. See the module docstring.
PLANNED_MAJOR_STOP_DETAIL = "mpo_rtp_planned_major_stop"

#: `calendar.txt` day columns in `date.weekday()` order (Monday first), so the
#: column is selected by index rather than by a name lookup that could silently
#: miss and read as "this service does not run".
GTFS_DAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
GTFS_SERVICE_ADDED = "1"  # calendar_dates.txt exception_type
GTFS_SERVICE_REMOVED = "2"

#: Calendar states in which a feed-derived screen may be reported at all.
CALENDAR_RESOLVED = "resolved"
#: The feed answered "nothing runs that day". A negative the feed supports.
CALENDAR_NO_SERVICE = "no_service_on_date"
#: States in which the feed could not be read for the date: report `unknown`.
CALENDAR_NO_AS_OF = "no_as_of"
CALENDAR_OUTSIDE_WINDOW = "outside_feed_window"
CALENDAR_NO_CALENDAR = "no_calendar"


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class StopService:
    stop_id: str
    name: str
    lat: float
    lon: float
    route_max_gaps: dict[str, float] = field(default_factory=dict)
    bus_routes: set[str] = field(default_factory=set)
    rail: bool = False
    ferry: bool = False

    def hqtc_routes(self) -> list[str]:
        return [r for r, g in self.route_max_gaps.items() if g <= HQTC_MAX_GAP_MIN]

    def major_candidate_routes(self) -> list[str]:
        return [
            r for r, g in self.route_max_gaps.items() if g <= MAJOR_STOP_MAX_GAP_MIN
        ]


def _read(z: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with z.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")))


def _read_optional(z: zipfile.ZipFile, name: str) -> list[dict[str, str]] | None:
    """Rows, or ``None`` when the feed does not ship the file at all.

    ``None`` and ``[]`` are different facts — a missing `calendar.txt` and an
    empty one lead to different calendar states — so the absence is returned,
    not flattened into an empty list.
    """
    try:
        return _read(z, name)
    except KeyError:
        return None


def _gtfs_date(value: str | None) -> date | None:
    """Parse a GTFS ``YYYYMMDD`` field, or ``None`` when it is absent or junk."""
    text = (value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


@dataclass(frozen=True)
class ServiceCalendar:
    """Which of a feed's services run on one stated date, and why not.

    ``status`` is the whole point of this record. A screen may read a feed
    stop only when the calendar resolved to real service; a *negative* screen
    may additionally rest on :data:`CALENDAR_NO_SERVICE`, because that is the
    feed answering rather than the feed being unreadable.
    """

    status: str
    service_date: str | None = None  # ISO 8601, or None when none was supplied
    service_ids_active: tuple[str, ...] = ()
    feed_valid_from: str | None = None  # from feed_info.txt, ISO 8601
    feed_valid_to: str | None = None
    calendar_source: str = "none"  # "calendar" | "calendar_dates" | "none"
    service_ids_added_by_exception: tuple[str, ...] = ()
    service_ids_removed_by_exception: tuple[str, ...] = ()
    #: `frequencies.txt` is not expanded. Recorded so a headway measured over
    #: template trips alone is never presented as a schedule.
    frequency_based_trips: bool = False

    @property
    def headways_measurable(self) -> bool:
        """True only when a real service set was resolved for the date."""
        return self.status == CALENDAR_RESOLVED

    @property
    def supports_a_negative(self) -> bool:
        """True when a feed-derived "no candidate" would be an answer.

        False for every state in which the feed simply could not be read for
        the requested date; those must report ``unknown``.
        """
        return self.status in {CALENDAR_RESOLVED, CALENDAR_NO_SERVICE}

    def reason(self) -> str:
        """One sentence naming what was withheld and why."""
        window = ""
        if self.feed_valid_from and self.feed_valid_to:
            window = f" The feed declares validity {self.feed_valid_from} to {self.feed_valid_to}."
        if self.status == CALENDAR_NO_AS_OF:
            return (
                "Peak headways were not measured: no service date was supplied, "
                "and the feed's services do not all run on the same days, so "
                "there is no single schedule to measure." + window
            )
        if self.status == CALENDAR_OUTSIDE_WINDOW:
            return (
                f"Peak headways were not measured: {self.service_date} falls "
                "outside the validity window the feed publishes for itself, so "
                "this feed does not state what runs on that date." + window
            )
        if self.status == CALENDAR_NO_CALENDAR:
            return (
                "Peak headways were not measured: the feed ships neither "
                "calendar.txt nor calendar_dates.txt, so no service date can be "
                "resolved."
            )
        if self.status == CALENDAR_NO_SERVICE:
            removed = ", ".join(self.service_ids_removed_by_exception)
            exception = (
                f" calendar_dates.txt removes service {removed} on that date."
                if removed
                else ""
            )
            return (
                f"The feed states that no service runs on {self.service_date}."
                + exception
            )
        active = ", ".join(self.service_ids_active) or "none"
        note = (
            f"Peak headways measured over service {active} running on "
            f"{self.service_date} (from {self.calendar_source})."
        )
        if self.service_ids_added_by_exception or self.service_ids_removed_by_exception:
            added = ", ".join(self.service_ids_added_by_exception) or "none"
            removed = ", ".join(self.service_ids_removed_by_exception) or "none"
            note += (
                f" calendar_dates.txt adds {added} and removes {removed} on that date."
            )
        if self.frequency_based_trips:
            note += (
                " The feed also ships frequencies.txt; headway-defined trips are "
                "not expanded here, so any route defined only that way is absent "
                "from these measurements."
            )
        return note


def _feed_validity(z: zipfile.ZipFile) -> tuple[date | None, date | None]:
    rows = _read_optional(z, "feed_info.txt") or []
    if not rows:
        return None, None
    row = rows[0]
    return _gtfs_date(row.get("feed_start_date")), _gtfs_date(row.get("feed_end_date"))


def _services_scheduled_on(rows: list[dict[str, str]], as_of: date) -> set[str]:
    """`calendar.txt` service ids whose day column and date range cover ``as_of``."""
    day_column = GTFS_DAY_COLUMNS[as_of.weekday()]
    active: set[str] = set()
    for row in rows:
        service_id = (row.get("service_id") or "").strip()
        if not service_id or row.get(day_column) != "1":
            continue
        start, end = _gtfs_date(row.get("start_date")), _gtfs_date(row.get("end_date"))
        # An unparsable bound is treated as unbounded on that side rather than
        # as a reason to drop the row: dropping it would remove real service
        # and read as a smaller schedule than the publisher declared.
        if (start and as_of < start) or (end and as_of > end):
            continue
        active.add(service_id)
    return active


def _service_exceptions_on(
    rows: list[dict[str, str]], as_of: date
) -> tuple[set[str], set[str]]:
    """`calendar_dates.txt` service ids added and removed on ``as_of``."""
    added: set[str] = set()
    removed: set[str] = set()
    for row in rows:
        if _gtfs_date(row.get("date")) != as_of:
            continue
        service_id = (row.get("service_id") or "").strip()
        if not service_id:
            continue
        exception = (row.get("exception_type") or "").strip()
        if exception == GTFS_SERVICE_ADDED:
            added.add(service_id)
        elif exception == GTFS_SERVICE_REMOVED:
            removed.add(service_id)
    return added, removed


def resolve_service_calendar(z: zipfile.ZipFile, as_of: date | None) -> ServiceCalendar:
    """Decide which services run on ``as_of``, or why that cannot be said."""
    valid_from, valid_to = _feed_validity(z)
    from_iso = valid_from.isoformat() if valid_from else None
    to_iso = valid_to.isoformat() if valid_to else None
    calendar_rows = _read_optional(z, "calendar.txt")
    exception_rows = _read_optional(z, "calendar_dates.txt")
    if calendar_rows:
        calendar_source = "calendar"
    elif exception_rows:
        calendar_source = "calendar_dates"
    else:
        calendar_source = "none"
    frequency_based = _read_optional(z, "frequencies.txt") is not None

    if as_of is None:
        return ServiceCalendar(
            status=CALENDAR_NO_AS_OF,
            calendar_source=calendar_source,
            frequency_based_trips=frequency_based,
            feed_valid_from=from_iso,
            feed_valid_to=to_iso,
        )
    stated = as_of.isoformat()
    if calendar_source == "none":
        return ServiceCalendar(
            status=CALENDAR_NO_CALENDAR,
            service_date=stated,
            calendar_source=calendar_source,
            frequency_based_trips=frequency_based,
            feed_valid_from=from_iso,
            feed_valid_to=to_iso,
        )
    if (valid_from and as_of < valid_from) or (valid_to and as_of > valid_to):
        return ServiceCalendar(
            status=CALENDAR_OUTSIDE_WINDOW,
            service_date=stated,
            calendar_source=calendar_source,
            frequency_based_trips=frequency_based,
            feed_valid_from=from_iso,
            feed_valid_to=to_iso,
        )

    scheduled = _services_scheduled_on(calendar_rows or [], as_of)
    added, removed = _service_exceptions_on(exception_rows or [], as_of)
    active = (scheduled | added) - removed

    status = CALENDAR_RESOLVED if active else CALENDAR_NO_SERVICE
    return ServiceCalendar(
        status=status,
        service_date=stated,
        service_ids_active=tuple(sorted(active)),
        calendar_source=calendar_source,
        service_ids_added_by_exception=tuple(sorted(added)),
        service_ids_removed_by_exception=tuple(sorted(removed)),
        frequency_based_trips=frequency_based,
        feed_valid_from=from_iso,
        feed_valid_to=to_iso,
    )


def _minutes(hms: str) -> int | None:
    parts = hms.strip().split(":")
    if len(parts) < 2:
        return None
    return int(parts[0]) * 60 + int(parts[1])


def _worst_peak_gap(times: list[int]) -> float | None:
    """Worst max-gap across the peak windows; None if any peak window has
    fewer than 2 trips (can't establish an interval at all).

    Window-edge gaps count. Otherwise service at 6:15 and every 15 minutes
    through 8:45 would be mislabeled as continuous 15-minute service across
    the full 6-9 AM window.
    """
    worst = 0.0
    for start, end in PEAKS:
        window = sorted(t for t in times if start <= t <= end)
        if len(window) < 2:
            return None
        gaps = [
            window[0] - start,
            *(b - a for a, b in itertools.pairwise(window)),
            end - window[-1],
        ]
        worst = max(worst, max(gaps))
    return worst


def _route_mode(route_type: str) -> str:
    if route_type in RAIL_ROUTE_TYPES:
        return "rail"
    if route_type in FERRY_ROUTE_TYPES:
        return "ferry"
    if route_type in BUS_ROUTE_TYPES:
        return "bus"
    # GTFS extended route types 700-799 are bus services.
    try:
        numeric = int(route_type)
    except (TypeError, ValueError):
        return "other"
    return "bus" if 700 <= numeric <= 799 else "other"


def _trips_on_date(
    z: zipfile.ZipFile, service_ids: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    """Every trip whose service runs on the resolved date.

    All of them, not the busiest one. Keeping only the largest service_id was
    how the previous implementation coped with not knowing the date, and it
    silently discarded supplemental service that shares a day with a base
    schedule.
    """
    wanted = set(service_ids)
    return {
        row["trip_id"]: row
        for row in _read(z, "trips.txt")
        if row["service_id"] in wanted
    }


def _arrivals(
    z: zipfile.ZipFile,
    trips: dict[str, dict[str, str]],
) -> dict[tuple[str, str, str], list[int]]:
    arrivals: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for stop_time in _read(z, "stop_times.txt"):
        trip = trips.get(stop_time["trip_id"])
        if not trip:
            continue
        time = stop_time.get("arrival_time") or stop_time.get("departure_time") or ""
        minutes = _minutes(time)
        if minutes is None:
            continue
        key = (
            stop_time["stop_id"],
            trip["route_id"],
            trip.get("direction_id", ""),
        )
        arrivals[key].append(minutes)
    return arrivals


def _apply_arrivals(
    stops: dict[str, StopService],
    route_types: dict[str, str],
    arrivals: dict[tuple[str, str, str], list[int]],
) -> None:
    for (stop_id, route_id, _direction), times in arrivals.items():
        stop = stops.get(stop_id)
        if stop is None:
            continue
        mode = _route_mode(route_types.get(route_id, "3"))
        if mode in {"rail", "ferry"}:
            setattr(stop, mode, True)
            continue
        if mode != "bus":
            continue
        stop.bus_routes.add(route_id)
        gap = _worst_peak_gap(times)
        best = stop.route_max_gaps.get(route_id)
        if gap is not None and (best is None or gap < best):
            stop.route_max_gaps[route_id] = gap


@dataclass(frozen=True)
class FeedScreen:
    """A feed's stops together with the calendar state they were read under.

    The two travel together deliberately. A bare ``list[StopService]`` cannot
    say whether its headways are a measurement or a blank, and a caller handed
    only the list will read empty ``route_max_gaps`` as "no qualifying
    service" when it may mean "no date was supplied".
    """

    stops: list[StopService]
    calendar: ServiceCalendar


def load_feed(gtfs_zip: Path, *, as_of: date | None = None) -> FeedScreen:
    """Load stops, and measure peak headways only for a resolvable date.

    Without ``as_of`` -- or for a date the feed does not cover -- the stops are
    returned with no headways at all, and ``calendar.status`` says why. That is
    the difference between "this stop has no qualifying service" and "this feed
    was never asked about a day".
    """
    with zipfile.ZipFile(gtfs_zip) as z:
        stops = _load_stops(z)
        calendar = resolve_service_calendar(z, as_of)
        # An early-out, not the invariant. The invariant is that an unresolved
        # calendar carries no `service_ids_active`, so the measurement below
        # would find no trips even without this branch; skipping the two
        # largest files in the archive is the reason the branch is here.
        if calendar.headways_measurable:
            route_types = {
                row["route_id"]: row.get("route_type", "3")
                for row in _read(z, "routes.txt")
            }
            _apply_arrivals(
                stops,
                route_types,
                _arrivals(z, _trips_on_date(z, calendar.service_ids_active)),
            )
    return FeedScreen(stops=list(stops.values()), calendar=calendar)


def _load_stops(z: zipfile.ZipFile) -> dict[str, StopService]:
    stops = {
        s["stop_id"]: StopService(
            stop_id=s["stop_id"],
            name=s["stop_name"],
            lat=float(s["stop_lat"]),
            lon=float(s["stop_lon"]),
        )
        for s in _read(z, "stops.txt")
        if s.get("stop_lat")
    }
    return stops


def _is_major_stop(stop: StopService, all_stops: list[StopService]) -> bool:
    """PRC § 21064.3 transit-mode and service conditions.

    Rail is independently qualifying. A ferry terminal qualifies only when
    the same clustered stop is served by bus or rail transit. Bus stops need
    two major routes with qualifying peak service.
    """
    if stop.rail:
        return True
    cluster = [stop]
    cluster.extend(
        other
        for other in all_stops
        if other.stop_id != stop.stop_id
        and haversine_miles(
            stop.lat,
            stop.lon,
            other.lat,
            other.lon,
        )
        <= STOP_CLUSTER_MILES
    )
    if stop.ferry and any(member.rail or member.bus_routes for member in cluster):
        return True
    routes = set(stop.major_candidate_routes())
    for other in cluster[1:]:
        routes |= set(other.major_candidate_routes())
    return len(routes) >= 2


@dataclass(frozen=True)
class HQStop:
    """A stop from the Caltrans/Cal-ITP statewide High Quality Transit
    Stops dataset — the state's own PRC § 21064.3 / § 21155 analysis,
    covering every agency (including rail and ferry the local feed may
    lack). Used alongside, not instead of, live-feed headway analysis:
    the two sources cross-check each other."""

    lat: float
    lon: float
    hqta_type: str  # major_stop_rail | major_stop_brt | major_stop_ferry
    # | major_stop_bus | hq_corridor_bus
    details: str
    agency: str

    @property
    def is_major(self) -> bool:
        """The dataset's own type classification, planned rows included."""
        return self.hqta_type.startswith("major_stop")

    @property
    def is_planned(self) -> bool:
        """An MPO submitted this location as planned in its adopted regional
        transportation plan. Caltrans records it as future service and states
        it does not validate these rows."""
        return self.details == PLANNED_MAJOR_STOP_DETAIL

    @property
    def is_existing_major(self) -> bool:
        """A major-stop row Caltrans derived from published service."""
        return self.is_major and not self.is_planned


def load_hq_stops(path: Path) -> list[HQStop]:
    data: dict[str, list[list[object]]] = json.loads(Path(path).read_text())
    seen: set[tuple[float, float, str, str]] = set()
    out: list[HQStop] = []
    for lat, lon, hqta_type, details, agency, _peak in data["stops"]:
        # `details` decides whether a row can support a candidate, so an
        # unreadable value fails the load rather than defaulting to a value
        # that would read as an operating facility.
        if not (
            isinstance(lat, (int, float))
            and isinstance(lon, (int, float))
            and isinstance(hqta_type, str)
            and isinstance(details, str)
        ):
            raise ValueError("invalid high-quality transit stop record")
        key = (lat, lon, hqta_type, details)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            HQStop(
                lat=float(lat),
                lon=float(lon),
                hqta_type=hqta_type,
                details=details,
                agency=agency if isinstance(agency, str) else "",
            )
        )
    return out


@dataclass(frozen=True)
class Determination:
    nearest_stop: StopService | None
    nearest_miles: float | None
    parking_exemption: str  # "candidate" | "no" | "unknown"
    height_18ft: str  # "candidate" | "no" | "unknown"
    qualifying_stops: list[tuple[StopService, float, str]]
    # Dataset rows inside the radius that were withheld because an MPO
    # submitted them as planned. Reported, never counted.
    planned_major_stops: list[tuple[HQStop, float]] = field(default_factory=list)
    #: The service-date state the feed stops were read under, when a feed was
    #: supplied. ``None`` means no feed calendar constrained this screen.
    calendar: ServiceCalendar | None = None

    def summary(self) -> str:
        lines = []
        if self.calendar is not None:
            lines.append(self.calendar.reason())
        if self.nearest_stop:
            # Qualified when the calendar did not resolve, so a stop's distance
            # is never read as service the feed did not confirm for the date.
            unread = (
                ""
                if self.calendar is None or self.calendar.headways_measurable
                else " — a location in the feed; this run does not state whether"
                " it is served on the requested date"
            )
            lines.append(
                f"Nearest transit stop: {self.nearest_stop.name} "
                f"({self.nearest_miles:.2f} mi straight-line){unread}"
            )
        if self.parking_exemption == "candidate":
            lines.append(
                "Parking exemption (Gov. Code § 66322(a)(1)): CANDIDATE — public "
                "transit within a half mile straight-line; confirm walking distance."
            )
        elif self.parking_exemption == "unknown":
            lines.append(
                "Parking exemption (Gov. Code § 66322(a)(1)): UNKNOWN — the "
                "supplied feed does not state what runs on the requested date, so "
                "no stop in it was read either way. This is not a finding that "
                "there is no transit near the site."
            )
        else:
            lines.append(
                "Parking exemption (Gov. Code § 66322(a)(1)): NO CANDIDATE FOUND "
                "IN SUPPLIED DATA — confirm operator/feed coverage before relying "
                "on this result."
            )
        if self.height_18ft == "candidate":
            stop, miles, reason = self.qualifying_stops[0]
            lines.append(
                f"18-ft height allowance (Gov. Code § 66321(b)(4)(B)): CANDIDATE — "
                f"{stop.name} ({miles:.2f} mi) is a {reason}; confirm walking distance."
            )
        elif self.height_18ft == "unknown":
            lines.append(
                "18-ft height allowance (Gov. Code § 66321(b)(4)(B)): UNKNOWN — "
                "peak headways decide this screen and none were measured, so the "
                "supplied feed neither establishes nor rules out a qualifying stop."
            )
        else:
            lines.append(
                "18-ft height allowance (Gov. Code § 66321(b)(4)(B)): NO CANDIDATE "
                "FOUND IN SUPPLIED DATA — no encoded qualifying stop was found "
                "within a half mile; confirm source coverage."
            )
        if self.planned_major_stops:
            lines.append(self._planned_stop_note())
        lines.append(
            "Screening result from GTFS peak headways on the stated service date; "
            "straight-line distance can eliminate a supplied stop but cannot prove "
            "dataset completeness. Not a legal determination."
        )
        return "\n".join(lines)

    def _planned_stop_note(self) -> str:
        """Say what was withheld, why, and who can settle it."""
        count = len(self.planned_major_stops)
        nearest, miles = self.planned_major_stops[0]
        noun = "stop" if count == 1 else "stops"
        return (
            f"Planned major transit {noun} (found, not counted): the Caltrans "
            f"HQ Transit Stops dataset lists {count} {noun} within a half mile "
            f"that an MPO submitted as planned in its adopted regional "
            f"transportation plan, nearest {nearest.agency} "
            f"({nearest.hqta_type}, {miles:.2f} mi). Caltrans records these as "
            f"future service and states it does not validate them. "
            f"PRC § 21155 counts a planned regional transportation plan stop "
            f"for its own section; PRC § 21064.3 does not. This screen counts "
            f"none of them; ask the transit agency or planning staff whether "
            f"the facility is in service."
        )


def _verdict(found: bool, feed_answers: bool) -> str:
    """A screen's verdict: a candidate, a supported negative, or `unknown`.

    A negative is only reportable when the sources could actually answer. The
    third branch is the whole point: an unreadable feed used to arrive here as
    "no".
    """
    if found:
        return "candidate"
    return "no" if feed_answers else "unknown"


def determine(
    lat: float,
    lon: float,
    stops: list[StopService],
    hq_stops: list[HQStop] | None = None,
    *,
    calendar: ServiceCalendar | None = None,
) -> Determination:
    """Screen a point against feed stops and the statewide dataset.

    ``calendar`` is the state the feed stops were loaded under. When it says
    the feed could not be read for the requested date, feed stops are neither
    counted nor read as an absence: the feed-derived screens report ``unknown``
    unless the statewide dataset — which carries its own currency and is not
    scoped by this feed's calendar — independently supplies a candidate.
    """
    # `calendar is None` means no feed calendar constrains this screen, which
    # is the shape of a statewide-dataset-only call.
    feed_counts = calendar is None or calendar.headways_measurable
    feed_answers = calendar is None or calendar.supports_a_negative
    if not stops and not hq_stops:
        verdict = _verdict(False, feed_answers)
        return Determination(None, None, verdict, verdict, [], calendar=calendar)
    with_dist = sorted(
        ((s, haversine_miles(lat, lon, s.lat, s.lon)) for s in stops),
        key=lambda x: x[1],
    )
    nearest, nearest_miles = with_dist[0] if with_dist else (None, None)

    qualifying: list[tuple[StopService, float, str]] = []
    for stop, miles in with_dist if feed_counts else []:
        if miles > HALF_MILE:
            break
        if _is_major_stop(stop, stops):
            qualifying.append(
                (stop, miles, "major transit stop (PRC § 21064.3, from feed headways)")
            )
        elif stop.hqtc_routes():
            qualifying.append(
                (
                    stop,
                    miles,
                    "high-quality transit corridor stop (PRC § 21155(b), from feed headways)",
                )
            )

    hq_within = []
    planned_within: list[tuple[HQStop, float]] = []
    for hq in hq_stops or []:
        miles = haversine_miles(lat, lon, hq.lat, hq.lon)
        if miles > HALF_MILE:
            continue
        if hq.is_planned:
            # Future service an MPO submitted and Caltrans does not validate.
            # It cannot show that transit is near the site today, so it
            # establishes neither standard; it is reported instead.
            planned_within.append((hq, miles))
            continue
        hq_within.append((hq, miles))
        label = (
            "major transit stop"
            if hq.is_major
            else "high-quality transit corridor stop"
        )
        stop_view = StopService(
            stop_id=f"hq:{hq.hqta_type}",
            name=f"{hq.agency} ({hq.hqta_type})",
            lat=hq.lat,
            lon=hq.lon,
        )
        qualifying.append(
            (
                stop_view,
                miles,
                f"{label} (Caltrans HQ Transit Stops dataset: "
                f"{hq.hqta_type}, {hq.details})",
            )
        )

    qualifying.sort(key=lambda x: x[1])
    planned_within.sort(key=lambda x: x[1])
    feed_stop_in_range = (
        feed_counts and nearest_miles is not None and nearest_miles <= HALF_MILE
    )
    parking = _verdict(bool(hq_within or feed_stop_in_range), feed_answers)
    height = _verdict(bool(qualifying), feed_answers)
    return Determination(
        nearest_stop=nearest,
        nearest_miles=nearest_miles,
        parking_exemption=parking,
        height_18ft=height,
        qualifying_stops=qualifying,
        planned_major_stops=planned_within,
        calendar=calendar,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="permit_pathways.transit",
        description="Transit-proximity screening for ADU parking/height standards.",
    )
    parser.add_argument("--gtfs", type=Path, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Service date to screen. Required for any headway conclusion; "
            "there is deliberately no default to today, so a recorded run "
            "stays reproducible."
        ),
    )
    default_hq = (
        Path(__file__).resolve().parents[2]
        / "corpus"
        / "transit"
        / "ca-hq-transit-stops.json"
    )
    parser.add_argument(
        "--hq-stops", type=Path, default=default_hq if default_hq.exists() else None
    )
    args = parser.parse_args()

    screen = load_feed(args.gtfs, as_of=args.as_of)
    stops = screen.stops
    hq = load_hq_stops(args.hq_stops) if args.hq_stops else []
    if screen.calendar.headways_measurable:
        headway_note = (
            f"{sum(1 for s in stops if s.hqtc_routes())} with ≤15-min peak routes, "
            f"{sum(1 for s in stops if len(s.major_candidate_routes()) >= 1)} with "
            f"≤20-min peak routes on "
            f"{screen.calendar.service_date} (service "
            f"{', '.join(screen.calendar.service_ids_active)})"
        )
    else:
        headway_note = "no peak headways measured"
    print(
        f"Loaded {len(stops)} feed stops; {headway_note}; "
        f"{len(hq)} Caltrans HQ dataset stops, of "
        f"which {sum(1 for s in hq if s.is_planned)} are MPO-submitted "
        f"planned stops this screen does not count.\n"
    )
    print(
        determine(
            args.lat, args.lon, stops, hq_stops=hq, calendar=screen.calendar
        ).summary()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
