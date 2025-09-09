# neuroml_to_jaxley.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import os
import json
import re
import math
import collections
import hashlib
import sys
import numpy as np


# ---- External deps (ensure installed) ----
# pip install neuroml jaxley matplotlib
import neuroml
import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse
try:
    from jaxley.synapses import GapJunction  # optional
except Exception:
    GapJunction = None

# Custom mechanisms
try:
    from custom_mechanisms import VGCaChannel, GradedChemicalSynapse, GapJunctionSynapse, DifferentiableExpTwoSynapse
except Exception:
    VGCaChannel = None
    GradedChemicalSynapse = None
    GapJunctionSynapse = None
    DifferentiableExpTwoSynapse = None


# =========================
#  INTERMEDIATE DATA MODEL
# =========================

@dataclass(frozen=True)
class CellSpec:
    id: str                            # canonical, indexable (e.g., "PDEL")
    params: Dict[str, Any] = field(default_factory=dict)  # ALL flattened params
    label: Optional[str] = None        # human label (e.g., "GenericNeuronCell")

@dataclass(frozen=True)
class ConnSpec:
    id: str                            # stable edge id (e.g., "PDEL->PDER:chem:42")
    pre_id: str
    post_id: str
    kind: str                          # "chem" | "gap"
    params: Dict[str, Any] = field(default_factory=dict)  # ALL flattened params
    label: Optional[str] = None

@dataclass
class NetworkSpec:
    cells: Dict[str, CellSpec]               # by id
    conns: List[ConnSpec]
    meta: Dict[str, Any] = field(default_factory=dict)


# =========================
#  Small unit helpers
# =========================

# === replace your unit table + to_si with this ===
_UNIT_SCALE = {
    # Conductance
    "S": 1.0, "MS": 1e-3, "mS": 1e-3, "US": 1e-6, "uS": 1e-6, "µS": 1e-6, "μS": 1e-6, "nS": 1e-9, "pS": 1e-12,
    # Current
    "A": 1.0, "MA": 1e-3, "mA": 1e-3, "UA": 1e-6, "uA": 1e-6, "µA": 1e-6, "μA": 1e-6, "nA": 1e-9, "pA": 1e-12,
    # Voltage
    "V": 1.0, "MV": 1e-3, "mV": 1e-3, "UV": 1e-6, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6,
    # Time
    "S_s": 1.0, "s": 1.0, "MS_ms": 1e-3, "ms": 1e-3, "US_us": 1e-6, "us": 1e-6, "µs": 1e-6, "μs": 1e-6,
    # Resistance
    "OHM": 1.0, "Ohm": 1.0, "ohm": 1.0, "kOhm": 1e3, "MOhm": 1e6, "MOHM": 1e6,
}

# Allow ".1", "0.1", "1.", scientific notation, optional unit, optional space
_num_unit_re = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµμ]+(?:\/[a-zA-Z0-9^]+)*)?\s*$"
)

def to_si(x: Any) -> float:
    """
    Parse values like '.1 nS', '0.1nS', '1e-9 S', '5ms', '200 MOhm' → float in SI units.
    NOTE: If a per-area/unit suffix appears (e.g., 'S/cm2'), we return the *numerator in SI*
          and let the caller handle the '/cm2' semantics separately. (We also expose helpers below.)
    """
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    m = _num_unit_re.match(s.replace(" ", ""))
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "")
        # Normalize µ/μ to µ and uppercase where our table expects it
        unit_norm = unit.replace("μ", "µ")
        # Fast path if pure unit in table
        if unit_norm in _UNIT_SCALE:
            return val * _UNIT_SCALE[unit_norm]
        # Try uppercase variants
        if unit_norm.upper() in _UNIT_SCALE:
            return val * _UNIT_SCALE[unit_norm.upper()]
        # If it's a compound (like 'nS/cm2'), scale numerator only
        head = unit_norm.split("/")[0]
        if head in _UNIT_SCALE:
            return val * _UNIT_SCALE[head]
        if head.upper() in _UNIT_SCALE:
            return val * _UNIT_SCALE[head.upper()]
        # Unknown unit → assume raw number
        return val
    # fallback: try plain float
    try:
        return float(s)
    except Exception:
        raise ValueError(f"Cannot parse numeric value with unit: {x!r}")

def keep_unit(x: Any) -> tuple[float, str]:
    """
    Return (numeric_SI, unit_suffix) preserving per-area suffixes (e.g., 'S/cm2').
    Useful if you want to store both absolute and per-area hints in params.
    """
    s = str(x).strip()
    m = _num_unit_re.match(s.replace(" ", ""))
    if not m:
        return (to_si(x), "")
    val = float(m.group(1))
    unit = (m.group(2) or "").replace("μ", "µ")
    # numeric scaling of head only
    head = unit.split("/")[0] if unit else ""
    scale = _UNIT_SCALE.get(head, _UNIT_SCALE.get(head.upper(), 1.0))
    return (val * scale, unit)
# ============================================
#  STEP 1: NeuroML → INTERMEDIATE (NetworkSpec)
# ============================================


def _mk_pointcell_xyzr(x: float, y: float, z: float, *, seg_len: float = 10.0, radius: float = 1.0):
    """
    Minimal morphology for plotting:
      one branch, two points (start=root, end=root+seg_len in +x), with radius.
    Returns a Python structure that Jaxley can accept after converting to np.array:
      [ [[x,y,z,r], [x+seg_len, y, z, r]] ]
    """
    return [[[float(x), float(y), float(z), float(radius)],
             [float(x + seg_len), float(y), float(z), float(radius)]]]

def _cell_xyzr_from_loc(params: dict) -> tuple[list, tuple] | tuple[None, None]:
    """If loc.x/loc.y exist, make a default xyzr + root_xyz; else (None, None)."""
    x, y = params.get("loc.x"), params.get("loc.y")
    z = params.get("loc.z", 0.0)
    if "loc.x" not in params or "loc.y" not in params:
        p["missing_loc_reason"] = (
            "no_instance_location"
            if not insts else
            "no_morphology_fallback_and_no_properties"
        )
    # fallback 3: use c302’s canonical position with variants, warn on failure
    if ("loc.x" not in params) or ("loc.y" not in params):
        try:
            from c302 import get_cell_position
            _pos = None
            try:
                _pos = get_cell_position(pid)
            except Exception:
                # Try _D variant if base is missing
                try:
                    _pos = get_cell_position(f"{pid}_D")
                except Exception:
                    _pos = None
            if _pos is not None:
                if "loc.x" not in p: p["loc.x"] = float(_pos.x)
                if "loc.y" not in p: p["loc.y"] = float(_pos.y)
                if ("loc.z" not in params) and hasattr(_pos, "z"): params["loc.z"] = float(_pos.z)
            else:
                print(f"[coords] no position found for {pid}: no instance, no morphology, no file")
        except Exception as e:
            print(f"[coords] fallback error for {pid}: {e}")
    xyzr = _mk_pointcell_xyzr(x, y, z)
    root = (float(x), float(y), float(z))
    return xyzr, root

def _index_cell_components(nml_doc) -> dict[str, Any]:
    """
    Index top-level cell-like components (e.g., iafCell, cell).
    Returns: id -> component object (we only read attributes flatly).
    """
    idx = {}
    for attr in dir(nml_doc):
        if attr.endswith("cells") or attr.endswith("Cells") or "cell" in attr.lower():
            seq = getattr(nml_doc, attr, None)
            if isinstance(seq, list):
                for obj in seq:
                    cid = str(getattr(obj, "id", "") or "")
                    if cid:
                        idx[cid] = obj
    return idx

def _index_pulse_generators(nml_doc) -> dict[str, Any]:
    """Index <pulseGenerator id=...> → object."""
    out = {}
    for attr in dir(nml_doc):
        if "pulse" in attr.lower():
            seq = getattr(nml_doc, attr, None)
            if isinstance(seq, list):
                for obj in seq:
                    pid = str(getattr(obj, "id", "") or "")
                    if pid:
                        out[pid] = obj
    return out

# -----------------------------
# Optional: load neuron types from c302 owmeta cache for robustness
# -----------------------------
_OWMETA_CACHE: Optional[Dict[str, Any]] = None
_OWMETA_TYPES: Optional[Dict[str, List[str]]] = None

def _load_neuron_types_from_cache() -> Dict[str, List[str]]:
    global _OWMETA_CACHE, _OWMETA_TYPES
    if _OWMETA_TYPES is not None:
        return _OWMETA_TYPES
    try:
        # Expect file at c302/data/owmeta_cache.json relative to this script
        here = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(here, "c302", "data", "owmeta_cache.json")
        if not os.path.isfile(path):
            _OWMETA_TYPES = {}
            return _OWMETA_TYPES
        with open(path, "r") as f:
            _OWMETA_CACHE = json.load(f)
        types_map: Dict[str, List[str]] = {}
        ni = _OWMETA_CACHE.get("neuron_info", {})
        # entries look like: name -> (cell_str, types_list, receptor_list, nt_list, short, color)
        for name, tup in ni.items():
            try:
                # owmeta_cache stores lists; ensure list of strings
                types = list(tup[1]) if isinstance(tup, (list, tuple)) and len(tup) > 1 else []
            except Exception:
                types = []
            types_map[str(name)] = [str(t) for t in types]
        _OWMETA_TYPES = types_map
        return _OWMETA_TYPES
    except Exception:
        _OWMETA_TYPES = {}
        return _OWMETA_TYPES

def _iter_input_lists(net):
    """Yield inputList-like objects on a NeuroML network (robust across schema variants)."""
    for attr in dir(net):
        if "input" not in attr.lower():
            continue
        seq = getattr(net, attr, None)
        if isinstance(seq, list):
            for obj in seq:
                # Some writers use 'population', others use 'populations'
                if hasattr(obj, "component") and (hasattr(obj, "population") or hasattr(obj, "populations")):
                    yield obj
    return

def _parse_bioparameters_from_notes(nml_doc) -> dict[str, float | str]:
    """
    Scrape simple 'BioParameter: name = value' lines from <notes> (if present).
    Keep both a numeric SI value (when parseable) and the raw string.
    Emits keys as:
      meta['bioparameters'][name] = {"si": <float or None>, "raw": <str>}
    """
    res = {}
    raw = getattr(nml_doc, "notes", None)
    if not raw:
        return res
    for line in str(raw).splitlines():
        line = line.strip()
        if not line.startswith("BioParameter:"):
            continue
        # BioParameter: foo_bar = 10ms (SRC: ..., certainty ...)
        try:
            lhs, _rest = line.split("(", 1) if "(" in line else (line, "")
            _, after = lhs.split("BioParameter:", 1)
            name, val = after.split("=", 1)
            name = name.strip()
            val = val.strip()
            si = None
            try:
                si = to_si(val)
            except Exception:
                pass
            res[name] = {"si": si, "raw": val}
        except Exception:
            continue
    return res


def harvest_properties(obj, into: Dict[str, Any], prefix: str):
    """Flatten <Property tag="X" value="Y"> → into[f"{prefix}.{tag}"]=value (numeric if possible)."""
    props = getattr(obj, "properties", None) or []
    for p in props:
        tag = str(getattr(p, "tag", "") or "").strip()
        val = getattr(p, "value", None)
        if not tag:
            continue
        try:
            into[f"{prefix}.{tag}"] = to_si(val)
        except Exception:
            into[f"{prefix}.{tag}"] = val

def _conn_pid_alloc():
    """
    Return a function that hands out deterministic unique edge ids (PIDs).
    Scope counters by a stable key to avoid collisions.
    """
    counters = collections.Counter()
    def alloc(pre: str, post: str, kind: str, proj_id: str, syn_component: str) -> str:
        key = (pre, post, kind, proj_id or "", syn_component or "")
        idx = counters[key]
        counters[key] += 1
        # PID is readable & unique across repeats
        return f"{pre}->{post}:{kind}:{proj_id}:{syn_component}:{idx}"
    return alloc


def _collect_population_ids(net) -> List[str]:
    return [str(getattr(p, "id", "")) for p in getattr(net, "populations", []) if getattr(p, "id", None)]

def _norm_ref_to_pop_id(raw: Optional[str], pop_id_set: set) -> Optional[str]:
    """Map NeuroML target like '../MC/0/...' → 'MC' using robust, case-insensitive matching."""
    if not raw:
        return None
    if not pop_id_set:
        return None

    # Build case-insensitive map once per call
    lower_map = {p.lower(): p for p in pop_id_set}

    s = str(raw)

    # 1) Exact/pop-substring, case-insensitive
    for p_low, p_orig in sorted(lower_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        if p_low in s.lower():
            return p_orig

    # 2) Extract path-like tokens and try each against pop ids (../POP/idx/..., POP[idx], POP)
    tokens = re.findall(r"[A-Za-z0-9_]+(?:\[\d+\])?", s)
    for tok in tokens:
        # strip [i] if present
        base = tok.split("[", 1)[0]
        hit = lower_map.get(base.lower())
        if hit:
            return hit

    return None

def _index_synapse_components(nml_doc) -> Dict[str, Any]:
    """Build id → synapse object (ExpTwoSynapse, GradedSynapse, GapJunction, etc.)."""
    syn_index: Dict[str, Any] = {}
    # Scan common containers on NeuroML doc
    for attr in dir(nml_doc):
        if attr.endswith("synapses") or "synapse" in attr.lower():
            seq = getattr(nml_doc, attr, None)
            if isinstance(seq, list):
                for obj in seq:
                    sid = str(getattr(obj, "id", "") or "")
                    if sid:
                        syn_index[sid] = obj
    return syn_index

def _flatten_syn_mech_params(syn_obj, into: Dict[str, Any]):
    """Copy known synapse mechanism parameters into flat connection dict (SI units)."""
    if syn_obj is None:
        return
    # Attempt to read typical fields across variants
    # Reversal
    for k in ("erev", "e_rev", "Erev", "E_rev"):
        if hasattr(syn_obj, k):
            into.setdefault("E_syn_V", to_si(getattr(syn_obj, k)))
            into.setdefault("E_syn_mV", into["E_syn_V"] * 1e3)
            break
    # Base conductance
    for k in ("gbase", "g_base", "gmax", "gMax"):
        if hasattr(syn_obj, k):
            into.setdefault("g_S", to_si(getattr(syn_obj, k)))
            break
    # Time constants (two-exp)
    for k in ("tau_rise", "tauRise", "tau1"):
        if hasattr(syn_obj, k):
            into.setdefault("tau_rise_s", to_si(getattr(syn_obj, k)))
            into.setdefault("tau_rise_ms", into["tau_rise_s"] * 1e3)
            break
    for k in ("tau_decay", "tauDecay", "tau2"):
        if hasattr(syn_obj, k):
            into.setdefault("tau_decay_s", to_si(getattr(syn_obj, k)))
            into.setdefault("tau_decay_ms", into["tau_decay_s"] * 1e3)
            break


def _extract_fraction(conn, role: str) -> float | None:
    """
    Return a fraction in [0,1] for 'pre' or 'post' if present on the connection.
    Handles many spellings:
      pre_fraction_along, preFractionAlong, pre_fraction, preFrac, pre_frac, fractionAlong, fraction
    Also searches <Property tag=... value=...>.
    """
    assert role in ("pre", "post")
    # 1) direct attributes (exact & common variants)
    cand_names = [
        f"{role}_fraction_along",
        f"{role}FractionAlong",
        f"{role}_fraction",
        f"{role}Fraction",
        f"{role}_frac",
        f"{role}Frac",
        # very generic fallbacks (applied if no role-specific found)
        "fraction_along",
        "fractionAlong",
        "fraction",
        "frac",
    ]
    for name in cand_names:
        if hasattr(conn, name):
            try:
                return float(getattr(conn, name))
            except Exception:
                pass

    # 2) scan all attributes for something like "pre...fraction"
    for attr in dir(conn):
        if not re.search(r"^pre|^post|fraction|frac", attr, re.IGNORECASE):
            continue
        if role in attr.lower() and "frac" in attr.lower():
            try:
                return float(getattr(conn, attr))
            except Exception:
                pass
        if role in attr.lower() and "fraction" in attr.lower():
            try:
                return float(getattr(conn, attr))
            except Exception:
                pass

    # 3) look in <Property> tags (Property(tag=..., value=...))
    props = getattr(conn, "properties", None) or []
    for p in props:
        tag = (getattr(p, "tag", "") or "").lower()
        if role in tag and ("frac" in tag or "fraction" in tag):
            try:
                return float(getattr(p, "value"))
            except Exception:
                # if it has units, try your to_si then clamp 0..1 (often unitless though)
                try:
                    return float(to_si(getattr(p, "value")))
                except Exception:
                    pass

    return None


def parse_neuroml_to_intermediate(nml_doc: neuroml.NeuroMLDocument) -> NetworkSpec:
    # print(nml_doc)

    if not getattr(nml_doc, "networks", None):
        raise ValueError("NeuroML document has no networks.")
    net = nml_doc.networks[0]

    pop_ids = _collect_population_ids(net)
    pop_id_set = set(pop_ids)

    syn_index   = _index_synapse_components(nml_doc)
    cell_index  = _index_cell_components(nml_doc)        # NEW
    pulses      = _index_pulse_generators(nml_doc)       # NEW
    bioparams   = _parse_bioparameters_from_notes(nml_doc)  # NEW

    # ---- Network-level defaults (flat into meta) ----  NEW
    meta: Dict[str, Any] = {"pop_ids": pop_ids}
    harvest_properties(net, meta, "net")  # e.g., recommended_duration_ms, recommended_dt_ms

    # ---- Cells (flat; capture more hints) ---- (REPLACE your existing cell loop)
    cells: Dict[str, CellSpec] = {}
    for pop in getattr(net, "populations", []) or []:
        pid  = str(getattr(pop, "id", "") or "")
        comp = str(getattr(pop, "component", "") or "")
        if not pid:
            continue
        p: Dict[str, Any] = {"component": comp}

        # population properties (color/type/neurotransmitter/etc.)
        harvest_properties(pop, p, "pop")
        # If missing pop.type, try to infer from cached owmeta neuron types
        if "pop.type" not in p:
            types_map = _load_neuron_types_from_cache()
            tlist = types_map.get(pid)
            if tlist:
                # store as semicolon-joined string to match existing convention
                p["pop.type"] = "; ".join(tlist)

        # initial V (if present on population)
        for k in ("v0", "initV", "initial_v", "initial_value"):
            if hasattr(pop, k):
                p["v0_V"] = to_si(getattr(pop, k))
                p["v0_mV"] = p["v0_V"] * 1e3
                break

        # passive hints if present on population
        for k, out in (("cm", "Cm_F_per_m2"), ("Cm", "Cm_F_per_m2")):
            if hasattr(pop, k):
                val, unit = keep_unit(getattr(pop, k))
                p[out] = val; p["Cm_unit"] = unit
        for k, out in (("ra", "Ra_ohm_m"), ("Ra", "Ra_ohm_m")):
            if hasattr(pop, k):
                p[out] = to_si(getattr(pop, k))

        # instance location (x,y,z) (flat)
        insts = getattr(pop, "instances", None) or []
        if insts:
            # Prefer the first instance that actually has a <location>
            chosen_loc = None
            for inst in insts:
                l = getattr(inst, "location", None)
                if l is not None:
                    chosen_loc = l
                    break
            if chosen_loc is not None:
                # always numbers; keep both xyz and a tuple for convenience
                try: p["loc.x"] = float(chosen_loc.x)
                except: pass
                try: p["loc.y"] = float(chosen_loc.y)
                except: pass
                try: p["loc.z"] = float(chosen_loc.z)
                except: pass

        # fallback 1: allow coordinates provided as properties (harvested under 'pop.*')
        if ("loc.x" not in p) or ("loc.y" not in p):
            for k_src in ("pop.loc.x", "pop.x", "cell.loc.x", "cell.x"):
                if ("loc.x" not in p) and (k_src in p):
                    try: p["loc.x"] = float(p[k_src])
                    except Exception: pass
            for k_src in ("pop.loc.y", "pop.y", "cell.loc.y", "cell.y"):
                if ("loc.y" not in p) and (k_src in p):
                    try: p["loc.y"] = float(p[k_src])
                    except Exception: pass
            for k_src in ("pop.loc.z", "pop.z", "cell.loc.z", "cell.z"):
                if ("loc.z" not in p) and (k_src in p):
                    try: p["loc.z"] = float(p[k_src])
                    except Exception: pass

        # fallback 2: try morphology proximal if component has a morphology
        if ("loc.x" not in p) or ("loc.y" not in p):
            comp_obj = cell_index.get(comp)
            try:
                if comp_obj is not None and hasattr(comp_obj, "morphology"):
                    morph = getattr(comp_obj, "morphology", None)
                    segs = getattr(morph, "segments", None) or []
                    if segs:
                        prox = getattr(segs[0], "proximal", None)
                        if prox is not None:
                            if "loc.x" not in p: p["loc.x"] = float(prox.x)
                            if "loc.y" not in p: p["loc.y"] = float(prox.y)
                            if ("loc.z" not in p) and hasattr(prox, "z"): p["loc.z"] = float(prox.z)
            except Exception:
                pass

        # fallback 3: use c302's helper to load canonical cell position from files (if available)
        if ("loc.x" not in p) or ("loc.y" not in p):
            try:
                from c302 import get_cell_position
                _pos = get_cell_position(pid)
                if "loc.x" not in p: p["loc.x"] = float(_pos.x)
                if "loc.y" not in p: p["loc.y"] = float(_pos.y)
                if ("loc.z" not in p) and hasattr(_pos, "z"): p["loc.z"] = float(_pos.z)
            except Exception:
                pass

        xyzr, root = _cell_xyzr_from_loc(p)
        if xyzr is not None:
            p["xyzr"] = xyzr            # list-of-branches; each branch is 2x4 [x,y,z,r]
            p["root_xyz"] = root        # convenience for quick checks / debug
        else:
            # leave out 'xyzr' to signal "no explicit coords"
            pass


        # component defaults: if the referenced component exists (e.g., iafCell/Cell), flatten key params
        comp_obj = cell_index.get(comp)
        if comp_obj is not None:
            # iafCell specific fields (names from your example)
            for src, out in (("leakReversal", "iaf.leakReversal_mV"),
                             ("thresh",        "iaf.thresh_mV"),
                             ("reset",         "iaf.reset_mV"),
                             ("C",             "iaf.C_F"),
                             ("leakConductance","iaf.leakConductance_S")):
                if hasattr(comp_obj, src):
                    try:
                        p[out] = to_si(getattr(comp_obj, src))
                    except Exception:
                        p[out] = getattr(comp_obj, src)
            # also harvest any <Property> on the component into 'cell.' namespace
            harvest_properties(comp_obj, p, "cell")
            # Map common HH-like fields if present on the component
            # E_Na/E_K in V → store mV
            for src, out in (("E_Na", "cell.E_Na_mV"), ("E_K", "cell.E_K_mV")):
                if hasattr(comp_obj, src):
                    try:
                        p[out] = to_si(getattr(comp_obj, src)) * 1e3
                    except Exception:
                        pass
            # Conductances per-area if specified; store raw SI for later mapping
            for src, out in (("gNa", "cell.gNa_S_per_cm2"), ("gK", "cell.gK_S_per_cm2"), ("gLeak", "cell.gLeak_S_per_cm2")):
                if hasattr(comp_obj, src):
                    try:
                        p[out] = to_si(getattr(comp_obj, src))
                    except Exception:
                        pass

        cells[pid] = CellSpec(id=pid, params=p, label=comp)

    conns: List[ConnSpec] = []
    alloc_pid = _conn_pid_alloc()  # deterministic unique IDs
    proj_id_get = lambda proj: str(getattr(proj, "id", "") or "")
    syn_index = _index_synapse_components(nml_doc)

    def _flatten_syn_mech_params(syn_obj, into: Dict[str, Any]):
        if syn_obj is None: return
        # Erev
        for k in ("erev", "e_rev", "Erev", "E_rev"):
            if hasattr(syn_obj, k):
                into.setdefault("E_syn_V", to_si(getattr(syn_obj, k)))
                into.setdefault("E_syn_mV", into["E_syn_V"] * 1e3)
                break
        # gbase/gmax
        for k in ("gbase", "g_base", "gmax", "gMax"):
            if hasattr(syn_obj, k):
                into.setdefault("g_S", to_si(getattr(syn_obj, k)))
                break
        # kinetics (ExpTwo/ExpOne variants)
        for k in ("tau_rise", "tauRise", "tau1"):
            if hasattr(syn_obj, k):
                into.setdefault("tau_rise_s", to_si(getattr(syn_obj, k)))
                into.setdefault("tau_rise_ms", into["tau_rise_s"] * 1e3)
                break
        for k in ("tau_decay", "tauDecay", "tau2"):
            if hasattr(syn_obj, k):
                into.setdefault("tau_decay_s", to_si(getattr(syn_obj, k)))
                into.setdefault("tau_decay_ms", into["tau_decay_s"] * 1e3)
                break
        # graded-specific hints (c302 often sets these)
        for k_in, k_out in (("vth", "V_th_mV"), ("Vth", "V_th_mV"),
                            ("delta", "delta_mV"), ("k", "k_mV")):
            if hasattr(syn_obj, k_in):
                into.setdefault(k_out, float(getattr(syn_obj, k_in)))

    def _iter_elec_connections(proj) -> List[Any]:
        for attr in ("electrical_connections", "electrical_connection_instances",
                    "electrical_connection_instance_ws", "electricalConnectionInstances"):
            v = getattr(proj, attr, None)
            if v: return v
        return []

    def _chem_from_projection(proj) -> None:
        syn_name = str(getattr(proj, "synapse", "") or "")
        syn_obj  = syn_index.get(syn_name)
        proj_id  = proj_id_get(proj)

        proj_defaults: Dict[str, Any] = {}
        harvest_properties(proj, proj_defaults, "proj")

        # Collect both Connection and ConnectionWD variants
        _conns = []
        c_std = getattr(proj, "connections", None)
        if c_std:
            _conns.extend(list(c_std))
        c_wd = getattr(proj, "connection_wds", None)
        if c_wd:
            _conns.extend(list(c_wd))

        for conn in _conns:
            pre_ref  = getattr(conn, "pre_cell_id", None) or getattr(conn, "pre_cell", None)
            post_ref = getattr(conn, "post_cell_id", None) or getattr(conn, "post_cell", None)
            pre  = _norm_ref_to_pop_id(str(pre_ref), pop_id_set)
            post = _norm_ref_to_pop_id(str(post_ref), pop_id_set)
            if not pre or not post: continue

            params: Dict[str, Any] = {
                "kind": "chem",
                "syn_component": syn_name,
                "proj_id": proj_id,
                "pre_ref": str(pre_ref),   # keep raw refs for troubleshooting
                "post_ref": str(post_ref),
            }
            # inherit projection-level <Property> defaults
            params.update(proj_defaults)

            # connection overrides
            if hasattr(conn, "delay"):  params["delay_s"] = to_si(getattr(conn, "delay"))
            if hasattr(conn, "weight"): params["weight"]  = float(getattr(conn, "weight"))
            # segment loci (kept even if un-used downstream)
            for attr, out in (("pre_segment_id", "pre_seg_id"), ("pre_fraction_along", "pre_frac"),
                            ("post_segment_id", "post_seg_id"), ("post_fraction_along", "post_frac")):
                if hasattr(conn, attr):
                    params[out] = float(getattr(conn, attr))
                    
            params["pre_frac"]  = _extract_fraction(conn, "pre")
            params["post_frac"] = _extract_fraction(conn, "post")

            harvest_properties(conn, params, "conn")

            # syn mechanism defaults (erev, gbase, taus, graded params)
            _flatten_syn_mech_params(syn_obj, params)

            # polarity heuristic if E_syn unknown
            if "E_syn_mV" in params:
                params.setdefault("polarity", "exc" if params["E_syn_mV"] >= -10.0 else "inh")
            else:
                nm = syn_name.lower()
                params.setdefault("polarity", "inh" if ("inh" in nm or "gaba" in nm) else "exc")

            edge_id = alloc_pid(pre, post, "chem", proj_id, syn_name)
            conns.append(ConnSpec(id=edge_id, pre_id=pre, post_id=post, kind="chem", params=params, label=syn_name))

    def _gap_from_projection(proj) -> None:
        proj_id = proj_id_get(proj)
        proj_defaults: Dict[str, Any] = {}
        harvest_properties(proj, proj_defaults, "proj")

        for conn in _iter_elec_connections(proj):
            pre_ref  = getattr(conn, "pre_cell_id", None) or getattr(conn, "pre_cell", None)
            post_ref = getattr(conn, "post_cell_id", None) or getattr(conn, "post_cell", None)
            pre  = _norm_ref_to_pop_id(str(pre_ref), pop_id_set)
            post = _norm_ref_to_pop_id(str(post_ref), pop_id_set)
            if not pre or not post: continue

            params: Dict[str, Any] = {
                "kind": "gap",
                "gj_component": "GapJunction",
                "proj_id": proj_id,
                "pre_ref": str(pre_ref),
                "post_ref": str(post_ref),
            }
            params.update(proj_defaults)

            # conductance or resistance (favor g if both provided)
            got_g = False
            for k in ("conductance", "g", "g_gap"):
                if hasattr(conn, k):
                    params["g_gap_S"] = to_si(getattr(conn, k))
                    got_g = True
                    break
            if not got_g:
                for k in ("resistance", "r", "r_gap"):
                    if hasattr(conn, k):
                        R = to_si(getattr(conn, k))
                        if R != 0: params["g_gap_S"] = 1.0 / R
                        break

            # loci + per-connection props
            for attr, out in (("pre_segment_id", "pre_seg_id"), ("pre_fraction_along", "pre_frac"),
                            ("post_segment_id", "post_seg_id"), ("post_fraction_along", "post_frac")):
                if hasattr(conn, attr):
                    params[out] = float(getattr(conn, attr))
            harvest_properties(conn, params, "conn")

            edge_id = alloc_pid(pre, post, "gap", proj_id, "GapJunction")
            conns.append(ConnSpec(id=edge_id, pre_id=pre, post_id=post, kind="gap", params=params, label="GapJunction"))

    # chemical projections (+ stray electrical)
    for proj in getattr(net, "projections", []) or []:
        has_chem = getattr(proj, "connections", None) or getattr(proj, "connection_wds", None)
        if has_chem:
            _chem_from_projection(proj)
        elif _iter_elec_connections(proj):
            _gap_from_projection(proj)


    # canonical electrical projections
    for eproj in getattr(net, "electrical_projections", []) or []:
        _gap_from_projection(eproj)

    # continuous (graded/analog) projections
    for cproj in getattr(net, "continuous_projections", []) or []:
        _cont_from_projection(cproj)

    meta["bioparameters"] = bioparams        # dict name -> {"si":..., "raw":...}

    # ---- Inputs/stimuli (pulseGenerators + inputList) ----
    inputs_flat: List[Dict[str, Any]] = []
    for il in _iter_input_lists(net):
        try:
            il_id   = str(getattr(il, "id", "") or "")
            # c302 uses 'populations'; accept both
            pop_id  = getattr(il, "population", None) or getattr(il, "populations", None)
            pop_id  = str(pop_id or "")
            comp_id = str(getattr(il, "component", "") or "")
            if not pop_id or not comp_id:
                continue
            pop_norm = _norm_ref_to_pop_id(pop_id, pop_id_set) or pop_id
            pulse = pulses.get(comp_id)
            if pulse is None:
                # try case-insensitive match
                for k, v in pulses.items():
                    if k.lower() == comp_id.lower():
                        pulse = v
                        break
            if pulse is None:
                inputs_flat.append({
                    "input_list_id": il_id,
                    "population": pop_norm,
                    "pulse_id": comp_id,
                })
                continue
            d_s = to_si(getattr(pulse, "delay")) if hasattr(pulse, "delay") else None
            dur_s = to_si(getattr(pulse, "duration")) if hasattr(pulse, "duration") else None
            amp_A = to_si(getattr(pulse, "amplitude")) if hasattr(pulse, "amplitude") else None
            inputs_flat.append({
                "input_list_id": il_id,
                "population": pop_norm,
                "pulse_id": comp_id,
                "delay_ms": float(d_s) * 1e3 if d_s is not None else None,
                "duration_ms": float(dur_s) * 1e3 if dur_s is not None else None,
                "amplitude_A": float(amp_A) if amp_A is not None else None,
            })
        except Exception:
            continue

    meta["inputs"] = inputs_flat


    spec = NetworkSpec(cells=cells, conns=conns, meta=meta)
    # print(spec)
    return spec

def meta_policy_from_bioparams(meta: Dict[str, Any]) -> MetaPolicy:
    bp = meta.get("bioparameters", {}) or {}
    def _si(name, fallback=None):
        d = bp.get(name, {})
        return d.get("si", fallback)

    rules: List[MetaRule] = []

    # Chemical Erev defaults
    exc_E = _si("chem_exc_syn_erev", 0.0)     # V
    inh_E = _si("chem_inh_syn_erev", -0.08)   # V
    if exc_E is not None:
        rules.append(MetaRule(
            target="conn", kind="chem",
            where={"polarity": "exc"},
            set={"E_syn_mV": float(exc_E)*1e3}
        ))
    if inh_E is not None:
        rules.append(MetaRule(
            target="conn", kind="chem",
            where={"polarity": "inh"},
            set={"E_syn_mV": float(inh_E)*1e3}
        ))

    # Chemical gbase defaults (neuron->neuron)
    exc_g = _si("neuron_to_neuron_chem_exc_syn_gbase", None)
    inh_g = _si("neuron_to_neuron_chem_inh_syn_gbase", None)
    if exc_g is not None:
        rules.append(MetaRule(
            target="conn", kind="chem",
            where={"polarity": "exc"},
            set={"g_S": float(exc_g)}
        ))
    if inh_g is not None:
        rules.append(MetaRule(
            target="conn", kind="chem",
            where={"polarity": "inh"},
            set={"g_S": float(inh_g)}
        ))

    # Gap junction default g
    gap_g = _si("neuron_to_neuron_elec_syn_gbase", None)
    if gap_g is not None:
        rules.append(MetaRule(
            target="conn", kind="gap",
            set={"g_gap_S": float(gap_g)}
        ))

    # IAF defaults (cells)
    for tgt, prefix in (("neuron_iaf_", "iaf."), ("muscle_iaf_", "iaf.")):
        C = _si(f"{tgt}C", None)
        gL = _si(f"{tgt}conductance", None)
        ELeak = _si(f"{tgt}leak_reversal", None)  # in V
        thresh = _si(f"{tgt}thresh", None)
        reset  = _si(f"{tgt}reset", None)
        sets: Dict[str, Any] = {}
        if C is not None:      sets[f"{prefix}C_F"] = float(C)
        if gL is not None:     sets[f"{prefix}leakConductance_S"] = float(gL)
        if ELeak is not None:  sets[f"{prefix}leakReversal_mV"] = float(ELeak)*1e3
        if thresh is not None: sets[f"{prefix}thresh_mV"] = float(thresh)*1e3
        if reset is not None:  sets[f"{prefix}reset_mV"] = float(reset)*1e3
        if sets:
            # crudely scope by component if available
            label_rx = r"generic_neuron_iaf_cell" if tgt.startswith("neuron") else r"generic_muscle_iaf_cell"
            rules.append(MetaRule(target="cell", label_regex=label_rx, set=sets))

    return MetaPolicy(rules=rules)


# ===================================================
#  STEP 2: INTERMEDIATE → JAXLEY PARAMETER BUNDLE
#  (flat mapping; user can override with simple callables)
# ===================================================

@dataclass
class JaxleyCellParams:
    mech: str = "HH"
    mech_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JaxleyConnParams:
    kind: str                      # "chem" | "gap"
    syn_class: str                 # "IonotropicSynapse" | "GapJunction"
    syn_kwargs: Dict[str, Any]     # {"gS":..., "E_syn_mV":...} or {"gGap":...}

@dataclass
class JaxleyParamBundle:
    cell_params: Dict[str, JaxleyCellParams]      # by cell id
    conn_params: Dict[str, JaxleyConnParams]      # by connection id
    meta: Dict[str, Any] = field(default_factory=dict)

CellMapper = Callable[[CellSpec], JaxleyCellParams]
ConnMapper = Callable[[ConnSpec], Optional[JaxleyConnParams]]

def default_cell_mapper(c: CellSpec) -> JaxleyCellParams:
    # Flat → HH; collect optional HH params if present (one-to-one mapping)
    p = c.params
    mk: Dict[str, Any] = {}

    def _pick(keys):
        for k in keys:
            if k in p:
                return p[k]
        return None

    # Conductance densities (S/cm^2)
    gNa = _pick(["cell.gNa_S_per_cm2", "cell.gNa", "pop.gNa", "gNa_S_per_cm2", "gNa"])
    gK  = _pick(["cell.gK_S_per_cm2",  "cell.gK",  "pop.gK",  "gK_S_per_cm2",  "gK"])
    gL  = _pick(["cell.gLeak_S_per_cm2","cell.gLeak","cell.gL",
                 "pop.gLeak", "pop.gL", "gLeak_S_per_cm2", "gL", "gLeak"])

    if gNa is not None:
        try: mk["HH_gNa"] = float(to_si(gNa))
        except Exception: pass
    if gK is not None:
        try: mk["HH_gK"] = float(to_si(gK))
        except Exception: pass
    if gL is not None:
        try: mk["HH_gLeak"] = float(to_si(gL))
        except Exception: pass

    # Reversal potentials (mV); to_si returns V → convert to mV
    def _to_mV(val):
        try:
            return float(to_si(val)) * 1e3
        except Exception:
            try:
                return float(val)
            except Exception:
                return None

    eNa = _pick(["cell.E_Na_mV", "pop.E_Na_mV", "ENa", "E_Na", "eNa"])
    eK  = _pick(["cell.E_K_mV",  "pop.E_K_mV",  "EK",  "E_K",  "eK"])
    eL  = _pick(["cell.E_leak_mV","cell.ELeak_mV","cell.eLeak_mV","pop.E_leak_mV","pop.ELeak_mV",
                 "cell.E_leak_V", "cell.ELeak_V", "cell.eLeak",   "pop.eLeak",     "ELeak", "E_leak", "eLeak"])

    mv = _to_mV(eNa)
    if mv is not None: mk["HH_eNa"] = mv
    mv = _to_mV(eK)
    if mv is not None: mk["HH_eK"] = mv
    mv = _to_mV(eL)
    if mv is not None: mk["HH_eLeak"] = mv

    return JaxleyCellParams(mech="HH", mech_kwargs=mk)

def default_conn_mapper(e: ConnSpec) -> Optional[JaxleyConnParams]:
    if e.kind == "chem":
        # e.params["g_S"] is in SI Siemens; IonotropicSynapse expects microSiemens
        gS_SI = float(e.params.get("g_S", 1e-9))
        gS = gS_SI * 1e6  # convert S → uS
        # Use E_syn_mV if present; else default by polarity
        if "E_syn_mV" in e.params:
            E = float(e.params["E_syn_mV"])
        else:
            pol = e.params.get("polarity", "exc")
            E = 0.0 if pol == "exc" else -70.0
        syn_kwargs = {"gS": gS, "E_syn_mV": E}
        # Map graded parameters when present (IonotropicSynapse names)
        if "V_th_mV" in e.params:
            try: syn_kwargs["v_th"] = float(e.params["V_th_mV"]) 
            except Exception: pass
        if "delta_mV" in e.params:
            try: syn_kwargs["delta"] = float(e.params["delta_mV"]) 
            except Exception: pass
        # k_minus (s^-1) if available
        for kname in ("k_minus", "k_minus_s", "k_minus_per_s"):
            if kname in e.params:
                try: syn_kwargs["k_minus"] = float(e.params[kname])
                except Exception: pass
                break
        # Pass through kinetics if available (kept in ms here)
        if "tau_rise_ms" in e.params:
            try: syn_kwargs["tau_rise_ms"] = float(e.params["tau_rise_ms"])
            except Exception: pass
        if "tau_decay_ms" in e.params:
            try: syn_kwargs["tau_decay_ms"] = float(e.params["tau_decay_ms"])
            except Exception: pass
        return JaxleyConnParams(kind="chem", syn_class="IonotropicSynapse",
                                syn_kwargs=syn_kwargs)
    if e.kind == "gap":
        if "g_gap_S" not in e.params:
            # if not present, you can set a default or skip
            return JaxleyConnParams(kind="gap", syn_class="GapJunction",
                                    syn_kwargs={"gGap": 5e-10})
        # Convert S -> uS for Jaxley
        return JaxleyConnParams(kind="gap", syn_class="GapJunction",
                                syn_kwargs={"gGap": float(e.params["g_gap_S"]) * 1e6})
    return None

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# -----------------------------
# Meta-defaults mapping policy
# -----------------------------
@dataclass
class MetaRule:
    target: str                     # "cell" or "conn"
    set: Dict[str, Any]             # defaults to set if key not present
    # selectors (all that are provided must match):
    id_regex: Optional[str] = None
    label_regex: Optional[str] = None
    where: Optional[Dict[str, Any]] = None      # matches on flat params
    # connection-only selectors:
    kind: Optional[str] = None                   # "chem" | "gap"
    syn_component: Optional[str] = None

@dataclass
class MetaPolicy:
    rules: List[MetaRule]

def _matches_regex(val: Optional[str], pattern: Optional[str]) -> bool:
    if pattern is None:
        return True
    if val is None:
        return False
    return re.search(pattern, str(val)) is not None

def _matches_where(params: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True
    for k, v in where.items():
        if params.get(k) != v:
            return False
    return True

def apply_meta_defaults(spec: NetworkSpec, policy: MetaPolicy) -> NetworkSpec:
    """Fill missing flat params on cells/conns from meta defaults (no overwrites)."""
    # apply cell rules
    for r in policy.rules:
        if r.target != "cell":
            continue
        for cid, cell in spec.cells.items():
            if not _matches_regex(cid, r.id_regex):
                continue
            if not _matches_regex(cell.label, r.label_regex):
                continue
            if not _matches_where(cell.params, r.where):
                continue
            # set defaults only if missing
            for k, v in r.set.items():
                if k not in cell.params:
                    cell.params[k] = v

    # apply connection rules
    for r in policy.rules:
        if r.target != "conn":
            continue
        for edge in spec.conns:
            if r.kind and edge.kind != r.kind:
                continue
            if not _matches_regex(edge.label, r.label_regex):
                continue
            if r.syn_component and edge.params.get("syn_component") != r.syn_component:
                continue
            if not _matches_where(edge.params, r.where):
                continue
            # id-based filter for conns: allow id_regex against "<pre>-><post>:..."
            if not _matches_regex(edge.id, r.id_regex):
                continue
            for k, v in r.set.items():
                if k not in edge.params:
                    edge.params[k] = v
    return spec


def translate_intermediate_to_jaxley_params(
    net: NetworkSpec,
    *,
    cell_mapper: CellMapper = default_cell_mapper,
    conn_mapper: ConnMapper = default_conn_mapper
) -> JaxleyParamBundle:
    cell_params: Dict[str, JaxleyCellParams] = {cid: cell_mapper(c) for cid, c in net.cells.items()}
    conn_params: Dict[str, JaxleyConnParams] = {}
    for edge in net.conns:
        mapped = conn_mapper(edge)
        if mapped is not None:
            conn_params[edge.id] = mapped
    return JaxleyParamBundle(cell_params=cell_params, conn_params=conn_params, meta=net.meta.copy())


# ==========================================
#  STEP 3: BUILD JAXLEY OBJECTS FROM BUNDLE
# ==========================================

@dataclass
class BuiltJaxley:
    net: jx.Network
    id_to_index: Dict[str, int]
    edge_id_to_index: Dict[str, int]


def build_jaxley_from_params(
    spec: NetworkSpec,
    bundle: JaxleyParamBundle,
    *,
    fast: bool = False,
) -> BuiltJaxley:
    # Cells
    id_to_index: Dict[str, int] = {}
    cells: List[jx.Cell] = []
    for idx, cid in enumerate(spec.cells.keys()):
        if ((idx % 100) == 0):
            print("Building cell "+str(idx) + " of "+str(len(spec.cells.keys())) + " "+str(cid))
        cp = bundle.cell_params.get(cid, JaxleyCellParams())
        comp = jx.Compartment()
        branch = jx.Branch(comp, ncomp=1)
        cell = jx.Cell(branch, parents=[-1])
        if cp.mech == "HH":
            cell.insert(HH(**cp.mech_kwargs))
        else:
            raise NotImplementedError(f"Unsupported mech '{cp.mech}' for cell {cid}")
        # Optionally insert VGCaChannel if available (per model_simplification)
        if (VGCaChannel is not None) and (not fast):
            try:
                cell.insert(VGCaChannel())
            except Exception:
                pass
        id_to_index[cid] = idx
        cells.append(cell)

        # Always attach basic xyzr locations (fast mode keeps geometry lightweight)
        xyzr_struct = spec.cells[cid].params.get("xyzr")
        if xyzr_struct is not None:
            arr = [np.asarray(branch, dtype=float) for branch in xyzr_struct]
            # try both attachment points
            try:
                cells[-1].base.xyzr = arr
            except Exception:
                try:
                    cells[-1].xyzr = arr
                except Exception as e:
                    print(f"[build] warning: failed to set xyzr for {cid}: {e}")


    net = jx.Network(cells)

    # 1) Install xyzr on the network’s cells (BEFORE connecting) for plotting/locations
    for i, cid in enumerate(spec.cells.keys()):
        xyzr_struct = spec.cells[cid].params.get("xyzr")
        if xyzr_struct is None:
            continue
        arr = [np.asarray(branch, dtype=float) for branch in xyzr_struct]
        # try both attachment points depending on Jaxley version
        net.cell(np.array([i])).xyzr = arr

    # 2) Refresh plotting/centers (safe no-op if not needed)
    if not fast:
        try:
            # per-cell center cache
            for i, _ in enumerate(spec.cells.keys()):
                c = net.cell(np.array([i]))
                if hasattr(c, "compute_compartment_centers"):
                    c.compute_compartment_centers()
        except Exception:
            pass

    def _loc_on_branch0(net, idx, frac):
        f = float(frac) if frac is not None else 0.0
        if not (0.0 <= f <= 1.0):
            f = 0.0
        # IMPORTANT: Jaxley variant expects a 1-D ndarray, not a 0-D.
        try:
            return net.cell(np.array([idx])).branch(0).loc(f)
        except Exception:
            # fallback for builds that accept Python ints
            return net.cell(idx).branch(0).loc(f)

# ---- Connections (robust) ----
    edge_id_to_index: Dict[str, int] = {}
    print("Edges:", len(spec.conns))
    i = 0
    for edge in spec.conns:
        if((i % 200) == 0):
            print("Building edge "+str(i) + " of "+str(len(spec.conns)) + " "+str(edge.id))
        i += 1
        pre_idx  = id_to_index.get(edge.pre_id)
        post_idx = id_to_index.get(edge.post_id)
        if pre_idx is None or post_idx is None:
            print(f"[build] drop edge (unknown cell): {edge.id} pre={edge.pre_id} post={edge.post_id}")
            continue

        jp = bundle.conn_params.get(edge.id)
        if jp is None:
            print(f"[build] drop edge (no mapper): {edge.id} kind={edge.kind} label={edge.label}")
            continue

        # Resolve endpoints AFTER xyzr was installed on net.cell(i)
        pre_loc  = _loc_on_branch0(net, pre_idx,  edge.params.get("pre_frac"))
        post_loc = _loc_on_branch0(net, post_idx, edge.params.get("post_frac"))

        n_before = len(net.edges.index)

        try:
            if jp.kind == "chem" and jp.syn_class == "IonotropicSynapse":
                if fast:
                    syn = IonotropicSynapse()
                else:
                    # Prefer differentiable ExpTwo-like synapse if available
                    if DifferentiableExpTwoSynapse is not None:
                        syn = DifferentiableExpTwoSynapse()
                    elif GradedChemicalSynapse is not None:
                        syn = GradedChemicalSynapse()
                    else:
                        syn = IonotropicSynapse()
                jx.connect(pre_loc, post_loc, syn)
            elif jp.kind == "gap" and jp.syn_class == "GapJunction" and GapJunction is not None:
                syn = GapJunction()
                jx.connect(pre_loc, post_loc, syn)
            elif jp.kind == "gap" and jp.syn_class == "GapJunction" and GapJunction is None and GapJunctionSynapse is not None:
                syn = GapJunctionSynapse()
                jx.connect(pre_loc, post_loc, syn)
            else:
                print(f"[build] skip edge (unsupported syn): {edge.id} kind={jp.kind} class={jp.syn_class}")
                continue
        except Exception as e:
            print(f"[build] connect failed: {edge.id} → {e}")
            continue

        n_after = len(net.edges.index)
        if n_after <= n_before:
            print(f"[build] connect produced no edge: {edge.id} ({edge.pre_id}->{edge.post_id})")
            continue

        eid = int(net.edges.index[n_after - 1])  # last added edge
        edge_id_to_index[edge.id] = eid

        # Set parameters, tolerating schema variants and multiple synapse implementations
        if jp.kind == "chem":
            if fast:
                # Fast path: set only basic IonotropicSynapse params if view exists
                try:
                    net.IonotropicSynapse.edge(eid).set("IonotropicSynapse_gS", float(jp.syn_kwargs["gS"]))
                except Exception:
                    pass
                try:
                    net.IonotropicSynapse.edge(eid).set("IonotropicSynapse_e_syn", float(jp.syn_kwargs["E_syn_mV"]))
                except Exception:
                    pass
            else:
                # Try setting on any available synapse view in order of most specific → generic
                syn_views = ("DifferentiableExpTwoSynapse", "GradedChemicalSynapse", "IonotropicSynapse")

                def _try_set_on_any(view_names, key_candidates, value) -> bool:
                    for vname in view_names:
                        if not hasattr(net, vname):
                            continue
                        edge_view = getattr(net, vname)
                        for key in key_candidates:
                            try:
                                edge_view.edge(eid).set(key, float(value))
                                return True
                            except Exception:
                                continue
                    return False

                # gS
                _try_set_on_any(
                    syn_views,
                    ("DifferentiableExpTwoSynapse_gS", "GradedChemicalSynapse_gS", "IonotropicSynapse_gS", "gS"),
                    jp.syn_kwargs["gS"],
                )
                # e_syn / reversal (in mV)
                _try_set_on_any(
                    syn_views,
                    ("DifferentiableExpTwoSynapse_e_syn", "GradedChemicalSynapse_e_syn", "IonotropicSynapse_e_syn", "e_syn"),
                    jp.syn_kwargs["E_syn_mV"],
                )
                # Optional kinetics if provided (ms)
                if "tau_rise_ms" in jp.syn_kwargs:
                    _try_set_on_any(
                        syn_views,
                        (
                            "DifferentiableExpTwoSynapse_tau_rise_ms",
                            "GradedChemicalSynapse_tau_rise_ms",
                            "IonotropicSynapse_tau_rise_ms",
                            "tau_rise_ms",
                            "tauRise_ms",
                        ),
                        jp.syn_kwargs["tau_rise_ms"],
                    )
                if "tau_decay_ms" in jp.syn_kwargs:
                    _try_set_on_any(
                        syn_views,
                        (
                            "DifferentiableExpTwoSynapse_tau_decay_ms",
                            "GradedChemicalSynapse_tau_decay_ms",
                            "IonotropicSynapse_tau_decay_ms",
                            "tau_decay_ms",
                            "tauDecay_ms",
                        ),
                        jp.syn_kwargs["tau_decay_ms"],
                    )
                # Graded params if present
                if "v_th" in jp.syn_kwargs:
                    _try_set_on_any(
                        syn_views,
                        ("DifferentiableExpTwoSynapse_v_th", "GradedChemicalSynapse_v_th", "IonotropicSynapse_v_th", "v_th"),
                        jp.syn_kwargs["v_th"],
                    )
                if "delta" in jp.syn_kwargs:
                    _try_set_on_any(
                        syn_views,
                        ("DifferentiableExpTwoSynapse_delta", "GradedChemicalSynapse_delta", "IonotropicSynapse_delta", "delta"),
                        jp.syn_kwargs["delta"],
                    )
                if "k_minus" in jp.syn_kwargs:
                    _try_set_on_any(
                        ("GradedChemicalSynapse", "IonotropicSynapse"),
                        ("GradedChemicalSynapse_k_minus", "IonotropicSynapse_k_minus", "k_minus"),
                        jp.syn_kwargs["k_minus"],
                    )
        else:  # gap
            for name in ("GapJunction_gGap", "gGap"):
                try:
                    net.GapJunction.edge(eid).set(name, float(jp.syn_kwargs["gGap"]))
                    break
                except Exception:
                    continue
            # If custom gap used
            if hasattr(net, "GapJunctionSynapse"):
                try:
                    net.GapJunctionSynapse.edge(eid).set("GapJunctionSynapse_gGap", float(jp.syn_kwargs["gGap"]))
                except Exception:
                    pass

    # k = 0
    # eid = int(net.edges.index[k])
    # src_cell = net.edges.src[k][0]   # (cell_idx, branch_idx, frac) in many builds
    # dst_cell = net.edges.dst[k][0]

    # src_xyz = net.cell(np.array([src_cell])).base.xyzr[0][0, :2]  # root x,y
    # dst_xyz = net.cell(np.array([dst_cell])).base.xyzr[0][0, :2]

    # print("[check] src root:", src_xyz, "dst root:", dst_xyz)
    # Locations

    # Final safety pass: clamp synaptic parameters to avoid NaNs in integrate/grad
    def _sanitize_synapses(_net: jx.Network):
        import math as _math
        # Ionotropic-like synapses
        if hasattr(_net, "IonotropicSynapse"):
            view = _net.IonotropicSynapse
            for eid in list(_net.edges.index):
                try:
                    # k_minus must be > 0
                    try:
                        km = float(view.edge(eid).get("IonotropicSynapse_k_minus"))
                        if (not _math.isfinite(km)) or km < 1e-6:
                            view.edge(eid).set("IonotropicSynapse_k_minus", 1e-6)
                    except Exception:
                        pass
                    # delta must be > 0
                    try:
                        de = float(view.edge(eid).get("IonotropicSynapse_delta"))
                        if (not _math.isfinite(de)) or de <= 0.0:
                            view.edge(eid).set("IonotropicSynapse_delta", 1.0)
                    except Exception:
                        pass
                    # gS non-negative
                    try:
                        gs = float(view.edge(eid).get("IonotropicSynapse_gS"))
                        if (not _math.isfinite(gs)) or gs < 0.0:
                            view.edge(eid).set("IonotropicSynapse_gS", 1e-4)
                    except Exception:
                        pass
                    # e_syn finite
                    try:
                        es = float(view.edge(eid).get("IonotropicSynapse_e_syn"))
                        if not _math.isfinite(es):
                            view.edge(eid).set("IonotropicSynapse_e_syn", 0.0)
                    except Exception:
                        pass
                except Exception:
                    continue
        # DifferentiableExpTwoSynapse
        if hasattr(_net, "DifferentiableExpTwoSynapse"):
            view = _net.DifferentiableExpTwoSynapse
            for eid in list(_net.edges.index):
                try:
                    for key, fallback in (
                        ("DifferentiableExpTwoSynapse_tau_rise_ms", 2.0),
                        ("DifferentiableExpTwoSynapse_tau_decay_ms", 10.0),
                        ("DifferentiableExpTwoSynapse_delta", 10.0),
                        ("DifferentiableExpTwoSynapse_gS", 1e-4),
                        ("DifferentiableExpTwoSynapse_e_syn", 0.0),
                    ):
                        try:
                            val = float(view.edge(eid).get(key))
                            if not _math.isfinite(val):
                                view.edge(eid).set(key, fallback)
                        except Exception:
                            pass
                    # enforce positivity
                    try:
                        tr = float(view.edge(eid).get("DifferentiableExpTwoSynapse_tau_rise_ms"))
                        if tr <= 0:
                            view.edge(eid).set("DifferentiableExpTwoSynapse_tau_rise_ms", 1.0)
                    except Exception:
                        pass
                    try:
                        td = float(view.edge(eid).get("DifferentiableExpTwoSynapse_tau_decay_ms"))
                        if td <= 0:
                            view.edge(eid).set("DifferentiableExpTwoSynapse_tau_decay_ms", 5.0)
                    except Exception:
                        pass
                except Exception:
                    continue

    _sanitize_synapses(net)

    return BuiltJaxley(net=net, id_to_index=id_to_index, edge_id_to_index=edge_id_to_index)


# -----------------------------
# c302 stimuli → Jaxley currents
# -----------------------------
def pulses_to_currents(
    inputs_flat: List[Dict[str, Any]],
    *,
    dt_ms: float,
    t_max_ms: float,
    amp_scale: float = 1.0,
    out_unit: str = "A",
) -> Dict[str, np.ndarray]:
    """Convert flattened inputs (delay_ms, duration_ms, amplitude_A) into per-pop current traces.
    Returns a dict: population_id -> np.ndarray [timepoints] of absolute current in ``out_unit``.

    Notes:
    - ``amplitude_A`` in inputs is in SI Amperes (A). To feed Jaxley ``data_stimulate``
      you typically want nA; set ``out_unit="nA"`` (and usually ``amp_scale=1.0``).
    - ``amp_scale`` is an additional multiplier applied after unit conversion.
    """
    unit = (out_unit or "A").lower()
    if unit in ("a",):
        unit_scale = 1.0
    elif unit in ("ua", "µa", "μa"):
        unit_scale = 1e6
    elif unit in ("na",):
        unit_scale = 1e9
    elif unit in ("pa",):
        unit_scale = 1e12
    else:
        unit_scale = 1.0

    n_t = int(np.round(t_max_ms / dt_ms)) + 1
    currents: Dict[str, np.ndarray] = {}
    for ent in inputs_flat or []:
        pop = ent.get("population")
        # Accept 0 as a valid population id; only skip if None or empty string
        if (pop is None) or (isinstance(pop, str) and pop.strip() == ""):
            continue
        delay = ent.get("delay_ms") or 0.0
        dur   = ent.get("duration_ms") or 0.0
        amp_A = (ent.get("amplitude_A") or 0.0)
        amp_out = float(amp_A) * unit_scale * float(amp_scale)
        if pop not in currents:
            currents[pop] = np.zeros(n_t, dtype=float)
        i0 = max(0, int(np.floor(delay / dt_ms)))
        i1 = min(n_t, int(np.floor((delay + dur) / dt_ms)))
        if i1 > i0:
            currents[pop][i0:i1] += amp_out
    return currents

def attach_currents_via_datastim(net: jx.Network, id_to_index: Dict[str, int], currents_by_pop: Dict[str, np.ndarray]):
    data_stimuli = None
    idxs = []
    for pop, curr in currents_by_pop.items():
        idx = id_to_index.get(pop)
        idxs.append(idx)
        if idx is None:
            continue
        try:
            # Debug prints removed to avoid slowdowns
            data_stimuli = net.cell(idx).branch(0).comp(0).data_stimulate(curr, data_stimuli)
        except Exception:
            print(idx)
            # # fallback to loc(0.0) if comp() not supported
            # data_stimuli = net.cell(idx).branch(0).loc(0.0).data_stimulate(curr, data_stimuli)
    return data_stimuli, idxs




# ==========================
#  OPTIONAL: c302 QUICKLOAD
# ==========================

def load_c302_preset_to_intermediate(preset: str) -> NetworkSpec:
    """
    Generate NeuroML via c302 and parse to intermediate NetworkSpec.
    Supports core presets (IA, IB, C, Full) and subsystem presets
    (e.g. "Pharyngeal:B", "Polysynaptic:A").
    """
    import importlib
    preset = preset.strip()

    # Ensure target directory exists (tests may run in clean tree)
    import os as _os
    _target_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "\c302\c302examples")
    try:
        _os.makedirs(_target_dir, exist_ok=True)
    except Exception:
        pass

    # Case 1: core presets
    if preset.upper() in ("IA", "IB"):
        from c302 import c302_IClamp
        level = preset[-1]
        out = c302_IClamp.setup(level, generate=True, verbose=False, target_directory=_target_dir)

    elif preset.upper() in ("C", "FULL"):
        from c302 import c302_Full
        level = "A" if preset.upper() == "FULL" else "C"
        out = c302_Full.setup(level, generate=True, verbose=False, muscles_to_include=False, target_directory=_target_dir)

    # Case 2: subsystem presets like "Pharyngeal:B"
    elif ":" in preset:
        subsystem, level = preset.split(":", 1)
        module_name = f"c302.c302_{subsystem.strip()}"
        mod = importlib.import_module(module_name)
        out = mod.setup(level.strip(), generate=True, verbose=False, target_directory=_target_dir)

    else:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Use IA, IB, C, Full or subsystem like 'Pharyngeal:B'."
        )

    nml_doc = next((x for x in out if hasattr(x, "networks")), None)
    if nml_doc is None:
        raise RuntimeError("c302 did not return a NeuroML document.")

    # print(nml_doc)
    return parse_neuroml_to_intermediate(nml_doc)


# ---------- Persistence: pickle dump/load for BuiltJaxley ----------

import pickle
from dataclasses import asdict

def save_built_jaxley(built: BuiltJaxley, path: str, *, meta: dict | None = None) -> None:
    """
    Pickle the full BuiltJaxley object to disk.
    Stores a small header with versions and optional user meta alongside.
    """
    header = {
        "format": "BuiltJaxley.pkl/v1",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "jaxley_version": getattr(jx, "__version__", "unknown"),
        "meta": meta or {},
    }
    payload = {
        "header": header,
        "built": built,  # includes .net, .id_to_index, .edge_id_to_index
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_built_jaxley(path: str) -> BuiltJaxley:
    """
    Load a BuiltJaxley object from a pickle produced by save_built_jaxley().
    Returns the BuiltJaxley; you can also read payload['header'] if you want metadata.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    built = payload["built"]
    # Optional: sanity checks
    if not isinstance(built, BuiltJaxley):
        raise TypeError("Pickle does not contain a BuiltJaxley object.")
    return built


# ==========================
#  Minimal CLI runner
# ==========================

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="IA")
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--plot_n", type=int, default=20)
    args = parser.parse_args()

    # Step 1: NeuroML → intermediate
    inter = load_c302_preset_to_intermediate(args.preset)

    base_meta = meta_policy_from_bioparams(inter.meta)

    inter = apply_meta_defaults(inter, base_meta)



    bundle = translate_intermediate_to_jaxley_params(
        inter,
        cell_mapper=default_cell_mapper,
        conn_mapper=default_conn_mapper
    )

    # Step 3: build Jaxley objects
    built = build_jaxley_from_params(inter, bundle)

    built.net.compute_xyz()

    # built = load_built_jaxley("c302Full.pkl")


    save_built_jaxley(built, "c302" + args.preset + ".pkl", meta={"preset": "Full"})
    print("Saved → c302" + args.preset + ".pkl")



    # Jaxley uses 1-based dims (1=x,2=y,3=z)
# Make net.cell accept Python ints by wrapping them as np.array([idx])
    from types import MethodType
    def _cell_int_ok(self, idx, _orig):
        if not isinstance(idx, np.ndarray):
            idx = np.array([idx])
        return _orig(idx)
    _orig_cell = built.net.cell
    built.net.cell = MethodType(lambda self, idx, _o=_orig_cell: _cell_int_ok(self, idx, _o), built.net)


    # print({cid: (c.params.get("loc.x"), c.params.get("loc.y")) for cid,c in inter.cells.items() if cid.startswith("MC")})
    
    ax = built.net.vis(
        detail="point", dims=(1, 2),
        color="k", synapse_color="g",
        synapse_scatter_kwargs={"s": 10, "alpha": 0.7, "zorder": -1},
        synapse_plot_kwargs={"alpha": 0.5, "zorder": -1}

    )
    # ax.figure.savefig("connectome_vis.png", dpi=200)

    # ax = built.net.vis(
    #     detail="point", dims=(1, 2),
    #     color="k",                                # neurons opaque (on top)
    #     synapse_color=(1, 0, 0, 0.5),              # half-transparent synapses
    #     synapse_scatter_kwargs={"s": 10, "alpha": 0.5, "zorder": -1},
    #     synapse_plot_kwargs={"alpha": 0.5, "zorder": -1}
    # )
    # # Make axes use the larger span for both x and y
    # try:
    #     xlim = ax.get_xlim(); ylim = ax.get_ylim()
    #     span_x = float(xlim[1] - xlim[0]); span_y = float(ylim[1] - ylim[0])
    #     max_span = max(span_x, span_y) if max(span_x, span_y) > 0 else 1.0
    #     cx = (xlim[0] + xlim[1]) * 0.5; cy = (ylim[0] + ylim[1]) * 0.5
    #     half = max_span * 0.5
    #     ax.set_xlim(cx - half, cx + half)
    #     ax.set_ylim(cy - half, cy + half)
    #     ax.set_aspect('equal', adjustable='box')
    # except Exception:
    #     pass

    # ax.figure.savefig("connectome_vis.png", dpi=200)

    ax.set_title("C. elegans Connectome (Full)", fontsize=12)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.figure.tight_layout()
    ax.figure.savefig("connectome_vis.png")

    
    # built.net.compute_xyz()


    # Record & simulate

    print(f"Simulating {len(inter.cells)} cells, {len(inter.conns)} connections "
          f"for {args.duration} ms (dt={args.dt} ms).")

    # If inputs parsed, stimulate via data_stimulate
    inputs_flat = inter.meta.get("inputs", [])

    # print(inputs_flat)

    data_stimuli = None
    if inputs_flat:
        # NeuroML amplitudes are in Amperes (often pA). If your channel uses mA/cm^2 internally,
        # Jaxley expects absolute current per compartment in Amperes. Provide a scale if needed.
        # Example: choose out_unit="nA" for Jaxley data_stimulate which expects nA per compartment.
        currents = pulses_to_currents(
            inputs_flat,
            dt_ms=args.dt,
            t_max_ms=args.duration,
            amp_scale=1.0,
            out_unit="nA",
        )
        # del currents["PLMR"]
        data_stimuli, idxs = attach_currents_via_datastim(built.net, built.id_to_index, currents)

    # Build recordings: presynaptic (idxs) and their postsynaptic targets
    try:
        if hasattr(built.net, "delete_recordings"):
            built.net.delete_recordings()
    except Exception:
        pass

    # Reverse map: index -> id
    idx_to_id = {v: k for k, v in built.id_to_index.items()}

    presyn_indices = [i for i in (idxs or []) if i is not None]
    presyn_ids = [idx_to_id.get(i) for i in presyn_indices if i in idx_to_id]

    postsyn_ids = []
    if presyn_ids:
        stim_set = set(presyn_ids)
        seen = set()
        for e in inter.conns:
            if e.pre_id in stim_set and e.post_id not in seen:
                seen.add(e.post_id)
                postsyn_ids.append(e.post_id)

    postsyn_indices = [built.id_to_index.get(cid) for cid in postsyn_ids]
    postsyn_indices = [i for i in postsyn_indices if i is not None]

    # Simple polarity check: print postsyn targets and polarity per stimulated presynaptic id
    try:
        def _classify(edge):
            if edge.kind == "gap":
                return "gap"
            pol = edge.params.get("polarity")
            if isinstance(pol, str) and pol in ("exc", "inh"):
                return pol
            E = edge.params.get("E_syn_mV")
            try:
                return "exc" if float(E) >= -10.0 else "inh"
            except Exception:
                name = (edge.label or "").lower()
                return "inh" if ("inh" in name or "gaba" in name) else "exc"

        if presyn_ids:
            print("Stimulated presynaptic ids:", presyn_ids)
            for pre in presyn_ids:
                outs = [(e.post_id, _classify(e), e.params.get("E_syn_mV")) for e in inter.conns if e.pre_id == pre]
                print(f"{pre}: {len(outs)} outgoing -> sample:", outs[:10])
    except Exception:
        pass

    record_indices = []
    seen_idx = set()
    for i in presyn_indices[0:1]:
        if i not in seen_idx:
            seen_idx.add(i); record_indices.append(i)
    for i in postsyn_indices[0:4]:
        if i not in seen_idx:
            seen_idx.add(i); record_indices.append(i)
    
    print(record_indices)

    # Fallback
    if not record_indices:
        record_indices = presyn_indices

    for i in record_indices[:max(1, args.plot_n)]:
        built.net.cell(i).branch(0).loc(0.0).record("v")


    print(idxs)
    print(data_stimuli)

    V = jx.integrate(built.net, delta_t=args.dt, t_max=args.duration, data_stimuli=data_stimuli)
    t = np.arange(0.0, args.duration + args.dt, args.dt)

    # Plot
    import os
    plt.figure(figsize=(9, 6))
    labels = [idx_to_id.get(i, str(i)) for i in record_indices[:args.plot_n]]
    for i, label in enumerate(labels):
        plt.plot(t, V[i], label=label)
    plt.title(f"Voltage traces ({args.preset} preset)")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.legend()
    plt.tight_layout()
    outfile = f"voltage_{args.preset}.png"
    plt.savefig(outfile)
    print(f"Saved plot to {outfile} (cwd: {os.getcwd()})")
