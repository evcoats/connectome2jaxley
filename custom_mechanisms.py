"""
Custom Jaxley mechanisms per porting/simplification notes:
  1) Voltage-gated Ca2+ channel (minimal e+f; add Ca handling later)
  2) Graded chemical synapse (excitatory or inhibitory)
  3) Gap junction (electrical) synapse

Conventions (matching Jaxley built-ins):
  - Voltages in mV
  - Conductances in microSiemens (uS) for synapses; channels often per-area S/cm^2
  - Time in ms
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.solver_gate import save_exp, solve_inf_gate_exponential
from jaxley.channels.channel import Channel
from jaxley.synapses.synapse import Synapse


class VGCaChannel(Channel):
    """Voltage-gated Ca2+ channel consistent with c302 (Boyle-like) and Jaxley.

    - Logistic gates e,f with time constants; current i_Ca = g * e^2 * f * (v - eCa)
    - Uses prefixed param/state names to avoid collisions across instances
    - Exposes current as "i_Ca"; prefers state "eCa" if present, else falls back to param E_Ca_mV
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name if name is not None else "VGCaChannel")
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCa": 1e-6,         # S/cm^2
            f"{prefix}_E_Ca_mV": 120.0,    # fallback if no eCa state present
            f"{prefix}_e_mid": -20.0,
            f"{prefix}_e_slope": 6.0,
            f"{prefix}_e_tau": 5.0,        # ms
            f"{prefix}_f_mid": -25.0,
            f"{prefix}_f_slope": -6.0,
            f"{prefix}_f_tau": 40.0,       # ms
        }
        self.channel_states = {
            f"{prefix}_e": 0.1,
            f"{prefix}_f": 0.8,
            "eCa": 120.0,
        }
        self.current_name = "i_Ca"

    @staticmethod
    def _sigmoid(v, mid, slope):
        return 1.0 / (1.0 + save_exp((mid - v) / slope))

    def update_states(self, states: Dict[str, jnp.ndarray], dt: float, v: float, params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        prefix = self._name
        e_inf = self._sigmoid(v, params[f"{prefix}_e_mid"], params[f"{prefix}_e_slope"])
        f_inf = self._sigmoid(v, params[f"{prefix}_f_mid"], params[f"{prefix}_f_slope"])
        e_tau = jnp.maximum(1e-6, params[f"{prefix}_e_tau"])
        f_tau = jnp.maximum(1e-6, params[f"{prefix}_f_tau"])
        new_e = solve_inf_gate_exponential(states[f"{prefix}_e"], dt, e_inf, e_tau)
        new_f = solve_inf_gate_exponential(states[f"{prefix}_f"], dt, f_inf, f_tau)
        return {f"{prefix}_e": new_e, f"{prefix}_f": new_f, "eCa": states.get("eCa", 120.0)}

    def init_state(self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray], delta_t: float) -> Dict[str, jnp.ndarray]:
        prefix = self._name
        e_inf = self._sigmoid(v, params[f"{prefix}_e_mid"], params[f"{prefix}_e_slope"])
        f_inf = self._sigmoid(v, params[f"{prefix}_f_mid"], params[f"{prefix}_f_slope"])
        return {f"{prefix}_e": jnp.asarray(e_inf), f"{prefix}_f": jnp.asarray(f_inf), "eCa": states.get("eCa", 120.0)}

    def compute_current(self, states: Dict[str, jnp.ndarray], v: float, params: Dict[str, jnp.ndarray]) -> float:
        prefix = self._name
        e = states[f"{prefix}_e"]
        f = states[f"{prefix}_f"]
        g = params[f"{prefix}_gCa"]
        eCa = states.get("eCa", params.get(f"{prefix}_E_Ca_mV", 120.0))
        return g * (e ** 2) * f * (v - eCa)


class GradedChemicalSynapse(Synapse):
    """Graded chemical synapse with logistic activation on presynaptic voltage.

    Params (synapse_params):
      - gS (uS): maximal conductance across postsynaptic membrane
      - e_syn (mV): synaptic reversal potential
      - v_th (mV): threshold for presynaptic voltage
      - delta (mV): slope for presynaptic activation
      - k_minus (s^-1): unbinding rate constant

    State (synapse_states):
      - s: open probability [0,1]
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name if name is not None else "GradedChemicalSynapse")
        prefix = self._name
        # Provide both own-namespace keys and IonotropicSynapse_* aliases so the
        # NeuroML translator can set either without caring which synapse class is used.
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,       # uS
            f"{prefix}_e_syn": 0.0,     # mV
            f"{prefix}_k_minus": 0.025, # s^-1
            f"{prefix}_v_th": -35.0,    # mV
            f"{prefix}_delta": 10.0,    # mV
            # Aliases matching IonotropicSynapse keys used by translator
            "IonotropicSynapse_gS": 1e-4,
            "IonotropicSynapse_e_syn": 0.0,
            "IonotropicSynapse_k_minus": 0.025,
            "IonotropicSynapse_v_th": -35.0,
            "IonotropicSynapse_delta": 10.0,
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(self, states: Dict, delta_t: float, pre_voltage: float, post_voltage: float, params: Dict) -> Dict:
        prefix = self._name
        # Resolve parameters from own keys or IonotropicSynapse_* aliases
        def P(key: str):
            return params.get(f"{prefix}_" + key, params.get("IonotropicSynapse_" + key))

        v_th = P("v_th")
        delta = jnp.maximum(1e-9, P("delta"))
        k_minus = jnp.maximum(1e-12, P("k_minus"))
        s_inf = 1.0 / (1.0 + save_exp((v_th - pre_voltage) / delta))
        tau_s = (1.0 - s_inf) / k_minus
        slope = -1.0 / jnp.maximum(1e-9, tau_s)
        exp_term = save_exp(slope * delta_t)
        new_s = states[f"{prefix}_s"] * exp_term + s_inf * (1.0 - exp_term)
        return {f"{prefix}_s": new_s}

    def compute_current(self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict) -> float:
        prefix = self._name
        def P(key: str):
            return params.get(f"{prefix}_" + key, params.get("IonotropicSynapse_" + key))
        gS = P("gS")
        e_syn = P("e_syn")
        g_syn = gS * states[f"{prefix}_s"]   # uS
        # Matches Jaxley sign convention: uS * mV → nA
        return g_syn * (post_voltage - e_syn)


class GapJunctionSynapse(Synapse):
    """Electrical coupling (gap junction).

    Params:
      - gGap (uS): coupling conductance
    No internal state.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name if name is not None else "GapJunctionSynapse")
        prefix = self._name
        self.synapse_params = {f"{prefix}_gGap": 1e-4}  # uS
        self.synapse_states = {}

    def update_states(self, states: Dict, delta_t: float, pre_voltage: float, post_voltage: float, params: Dict) -> Dict:
        return states

    def compute_current(self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict) -> float:
        prefix = self._name
        # Accept multiple key variants: custom, built-in, or bare "gGap"
        g = (
            params.get(f"{prefix}_gGap")
            or params.get("GapJunctionSynapse_gGap")
            or params.get("GapJunction_gGap")
            or params.get("gGap")
        )
        # Current into postsynaptic compartment (Ohmic coupling)
        return g * (pre_voltage - post_voltage)


class DifferentiableExpTwoSynapse(Synapse):
    """Differentiable two-exponential synapse driven by smooth presynaptic trigger.

    States:
      - x: rising state
      - s: conductance state

    Params (synapse_params):
      - gS (uS), e_syn (mV)
      - tau_rise_ms, tau_decay_ms
      - v_th (mV), delta (mV) for smooth spike surrogate u = sigmoid((Vpre - v_th)/delta)
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name if name is not None else "DifferentiableExpTwoSynapse")
        prefix = self._name
        # Primary params (own namespace)
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,                 # uS
            f"{prefix}_e_syn": 0.0,              # mV
            f"{prefix}_tau_rise_ms": 2.0,        # ms
            f"{prefix}_tau_decay_ms": 10.0,      # ms
            f"{prefix}_v_th": -35.0,             # mV
            f"{prefix}_delta": 10.0,             # mV
            # Aliases to support tutorial-style customization via IonotropicSynapse_* columns
            "IonotropicSynapse_gS": 1e-4,
            "IonotropicSynapse_e_syn": 0.0,
            "IonotropicSynapse_v_th": -35.0,
            "IonotropicSynapse_delta": 10.0,
            # Optional aliases for kinetics if users set them this way
            "IonotropicSynapse_tau_rise_ms": 2.0,
            "IonotropicSynapse_tau_decay_ms": 10.0,
        }
        self.synapse_states = {f"{prefix}_x": 0.0, f"{prefix}_s": 0.0}

    def update_states(self, states: Dict, delta_t: float, pre_voltage: float, post_voltage: float, params: Dict) -> Dict:
        prefix = self._name
        # Resolve params with IonotropicSynapse_* aliases when present
        def P(key: str, alt: str | None = None):
            return params.get(f"{prefix}_{key}", params.get(alt if alt is not None else f"IonotropicSynapse_{key}"))

        v_th = P("v_th")
        delta = jnp.maximum(1e-9, P("delta"))
        tau_r = jnp.maximum(1e-6, P("tau_rise_ms"))
        tau_d = jnp.maximum(1e-6, P("tau_decay_ms"))

        # Smooth presynaptic trigger in [0,1]
        u = 1.0 / (1.0 + save_exp((v_th - pre_voltage) / delta))
        # Exact one-step updates for first-order filters
        x_old = states[f"{prefix}_x"]
        s_old = states[f"{prefix}_s"]
        ax = save_exp(-delta_t / tau_r)
        x_new = x_old * ax + u * (1.0 - ax)
        as_ = save_exp(-delta_t / tau_d)
        s_new = s_old * as_ + x_new * (1.0 - as_)
        return {f"{prefix}_x": x_new, f"{prefix}_s": s_new}

    def compute_current(self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict) -> float:
        prefix = self._name
        # Resolve gS/e_syn from own keys or IonotropicSynapse_* aliases
        gS = params.get(f"{prefix}_gS", params.get("IonotropicSynapse_gS"))
        e_syn = params.get(f"{prefix}_e_syn", params.get("IonotropicSynapse_e_syn"))
        g_syn = gS * states[f"{prefix}_s"]
        return g_syn * (post_voltage - e_syn)


