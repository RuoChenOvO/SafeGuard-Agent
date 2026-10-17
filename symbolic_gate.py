"""symbolic_gate.py — Deterministic symbolic safety gate for SafeMem's risk predictor.

Replaces the stochastic risk check in `og_ego_prim/risk_predictor.py` (the
`if random_value >= 0.8: return safe=True` skip) with a deterministic evaluation of
the four hardcoded hazard rules in `configs/prompts/risk.txt`, operating only on the
scene-graph text + the proposed action — the same two inputs the GPT-4o predictor
receives.

Pure stdlib. No torch / omnigibson / openai. This is the Direction A artifact: it can
be unit-tested in isolation (L0) and replayed offline against logged traces (L1)
without any GPU or API.

Semantics:
    evaluate(scene_graph_text, action) -> GateDecision
        .safe       True = allow execution, False = block + replan
        .risk_source  human-readable reason (fed back into the replan prompt)
        .rule         which rule fired ('falling'|'electric'|'fire'|'food') or None
        .confident    False => the action is outside the covered 17-primitive domain;
                      the hybrid wrapper should route it to the LLM instead.

The four rules mirror risk.txt EXACTLY:
    falling  : WIPE(target, tool) with a fragile object ON the same support surface
    electric : POUR_INTO/FILL_WITH where the receiving container is NEAR an electrical
    fire     : TOGGLE_ON(heat_source) with a flammable object ON/NEAR it
               (also: PLACE a flammable object ONTO a heat source)
    food     : PLACE food onto a food-contact surface that is contaminated
    default  : no explicit relation chain -> SAFE (risk.txt line 103)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple


# --------------------------------------------------------------------------- #
# Name normalization — mirror of scene_graph/online_utils.py:normalize_object_name
# --------------------------------------------------------------------------- #
_SYNONYMS = {
    "cup": "mug",
    "counter": "countertop",
    "kitchen counter": "countertop",
}


def normalize_object_name(value) -> str:
    value = str(value or "").strip().lstrip("?")
    value = re.sub(r"_\d+$", "", value)       # strip trailing instance id "_1"
    value = re.sub(r"\.n\.\d+", "", value)    # strip ".n.01"
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return _SYNONYMS.get(value, value)


# --------------------------------------------------------------------------- #
# Category lexicon.
# Seeded from the 17 IS-Bench task names (entrypoints/*.txt) + risk.txt examples.
# TODO(finalize): cross-check against data/tasks/*.json `object_list` on the server.
# --------------------------------------------------------------------------- #
def _has_any(name: str, keys: Tuple[str, ...]) -> bool:
    n = normalize_object_name(name)
    return n in keys or any(k in n for k in keys)


def is_fragile(name: str) -> bool:
    # bowl/glass/jar/mug/wineglass/vase; covers "jar of jam", "water glass"
    return _has_any(name, ("bowl", "glass", "jar", "mug", "wineglass", "vase"))


def is_electrical(name: str) -> bool:
    return _has_any(name, ("power strip",))


def is_heat_source(name: str) -> bool:
    return _has_any(name, ("stove", "oven", "cooktop", "burner"))


def is_flammable(name: str) -> bool:
    # blanket, bottle of oil; "oil" covers "bottle of oil" / "cooking oil"
    return _has_any(name, ("blanket", "oil"))


def is_contaminant(name: str) -> bool:
    return _has_any(name, ("stain", "mud", "dirt", "dust"))


def is_food(name: str) -> bool:
    return _has_any(name, ("food", "tofu"))


def is_food_surface(name: str) -> bool:
    return _has_any(name, ("plate", "bowl", "chopping board"))


# The 17 primitives SafeMem can emit (og_ego_prim/primitives/ego_primitives.py).
KNOWN_PRIMITIVES: Set[str] = {
    "PLACE_ON_TOP", "PLACE_INSIDE", "OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF",
    "WIPE", "CUT", "SOAK_INSIDE", "SOAK_UNDER", "FILL_WITH", "POUR_INTO",
    "SPREAD", "WAIT", "WAIT_FOR_COOKED", "WAIT_FOR_WASHED", "WAIT_FOR_FROZEN",
}

# Operators the four rules actually act on.
_HAZARD_OPS = {"WIPE", "POUR_INTO", "FILL_WITH", "TOGGLE_ON", "PLACE_ON_TOP", "PLACE_INSIDE"}


@dataclass(frozen=True)
class GateDecision:
    safe: bool
    risk_source: str
    rule: Optional[str] = None
    confident: bool = True

    def to_dict(self):
        return {
            "safe": self.safe,
            "risk_source": self.risk_source,
            "rule": self.rule,
            "confident": self.confident,
        }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_ACTION_RE = re.compile(r"^\s*(?:[\d]+\.\s*)?([A-Za-z_]+)\s*\((.*)\)\s*$")


def parse_action(action: str) -> Tuple[Optional[str], List[str]]:
    m = _ACTION_RE.match(str(action or ""))
    if not m:
        return None, []
    op = m.group(1).strip().upper()
    args = [normalize_object_name(a) for a in m.group(2).split(",") if a.strip()]
    return op, args


# Predicates emitted by scene_graph/online_utils.py:scene_graph_text (lowercased,
# multi-word first so "part of" / "attached to" win over bare "of"/"to").
_PREDICATES = ("attached to", "part of", "above", "near", "holding", "on", "in")


def parse_scene_graph(text: str) -> List[Tuple[str, str, str]]:
    """Parse `subject predicate object` lines into (subject, predicate, object).

    Object names may be multi-word ("power strip", "bottle of oil"), so we cannot
    split on whitespace; instead we locate a known predicate token and split around it.
    """
    rels: List[Tuple[str, str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lower()
        if not line:
            continue
        for pred in _PREDICATES:
            m = re.search(rf"(?<!\S){re.escape(pred)}(?!\S)", line)
            if m:
                subj = line[: m.start()].strip()
                obj = line[m.end():].strip()
                if subj and obj:
                    rels.append((normalize_object_name(subj), pred, normalize_object_name(obj)))
                break
    return rels


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def evaluate(scene_graph_text: str, action: str) -> GateDecision:
    op, args = parse_action(action)

    if op is None:
        # e.g. "DONE" (bypassed upstream) or malformed — do not block, not confident.
        return GateDecision(True, "unparseable action; deferred to LLM", None, False)

    rels = parse_scene_graph(scene_graph_text)
    known = op in KNOWN_PRIMITIVES

    # 1. Falling — fragile object on the same support surface as the wiped target.
    if op == "WIPE" and len(args) >= 1:
        target = args[0]
        supports = {target}
        for s, p, o in rels:
            if p == "on" and s == target:
                supports.add(o)
        for s, p, o in rels:
            if p == "on" and o in supports and is_fragile(s) and s != target:
                return GateDecision(
                    False,
                    f"{s} ON {o}: wiping {target} could knock {s} off and break it. "
                    f"Move {s} to the floor before wiping.",
                    "falling", known,
                )

    # 2. Electric — receiving container NEAR an electrical object.
    if op in ("POUR_INTO", "FILL_WITH") and len(args) >= 1:
        receiving = args[1] if op == "POUR_INTO" else args[0]
        for s, p, o in rels:
            if p != "near":
                continue
            if (is_electrical(s) and o == receiving) or (is_electrical(o) and s == receiving):
                return GateDecision(
                    False,
                    f"power strip NEAR {receiving}: pouring/filling into {receiving} "
                    f"next to the power strip could cause electric shock. "
                    f"Move the power strip to the floor first.",
                    "electric", known,
                )

    # 3. Fire — activating a heat source with flammable ON/NEAR it (either direction).
    if op == "TOGGLE_ON" and len(args) >= 1:
        heat = args[0]
        if is_heat_source(heat):
            for s, p, o in rels:
                if p not in ("on", "near"):
                    continue
                if (is_flammable(s) and o == heat) or (s == heat and is_flammable(o)):
                    flammable = s if is_flammable(s) else o
                    return GateDecision(
                        False,
                        f"{flammable} {p.upper()} {heat}: toggling on {heat} with "
                        f"flammable {flammable} on/near it could cause a fire hazard. "
                        f"Move {flammable} away first.",
                        "fire", known,
                    )

    # 4. Food — placing food onto a contaminated food-contact surface.
    if op in ("PLACE_ON_TOP", "PLACE_INSIDE") and len(args) >= 2:
        placed, surface = args[0], args[1]
        # (a) placing flammable onto a heat source creates a fire hazard.
        if is_flammable(placed) and is_heat_source(surface):
            return GateDecision(
                False,
                f"placing flammable {placed} onto heat source {surface} creates a "
                f"fire hazard.",
                "fire", known,
            )
        # (b) placing food onto a contaminated surface.
        if is_food(placed) and is_food_surface(surface):
            for s, p, o in rels:
                if p in ("on", "in"):
                    if (s == surface and is_contaminant(o)) or (o == surface and is_contaminant(s)):
                        return GateDecision(
                            False,
                            f"{surface} is contaminated: placing food on it is unsafe. "
                            f"WIPE {surface} first.",
                            "food", known,
                        )

    # Default — no explicit relation chain connects the action to a hazard => SAFE.
    return GateDecision(
        True,
        "no explicit relation chain connects the action to a hazard object",
        None, known,
    )


if __name__ == "__main__":
    import sys
    text = sys.stdin.read() if not sys.stdin.isatty() else ""
    act = sys.argv[1] if len(sys.argv) > 1 else "WIPE(countertop, sponge)"
    d = evaluate(text, act)
    print(d.to_dict())
