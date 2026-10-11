#!/usr/bin/env python3
"""
Panasonic Comfort Cloud client: read status and control your AC
(e.g. CS-XU18BKH-1) over the internet using your Comfort Cloud account.

This talks to Panasonic's cloud (the same backend the phone app uses), so it
works from anywhere, not just your home Wi-Fi. Your AC must already be
REGISTERED in the Comfort Cloud app (tap 注册 / Register in the app first).

-------------------------------------------------------------------------
IMPORTANT — you need a small helper library
-------------------------------------------------------------------------
Panasonic's login flow changes often and requires an app-version token, so
rather than re-implement a brittle login here, this script uses the
community-maintained `pcomfortcloud` package, which keeps up with those
changes.

Install it once:

    pip install pcomfortcloud

(If you prefer zero dependencies, tell me and I'll write a raw-HTTP version,
but it tends to break whenever Panasonic bumps the required app version.)

-------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------
Set your credentials as environment variables (safer than hard-coding):

    export PANASONIC_USER="you@example.com"
    export PANASONIC_PASS="your-password"

Then:

    python3 comfort_cloud.py list              # list devices + online status
    python3 comfort_cloud.py status            # show status of all devices
    python3 comfort_cloud.py on                # turn first device ON
    python3 comfort_cloud.py off               # turn first device OFF
    python3 comfort_cloud.py temp 24           # set target temp to 24°C

You can target a specific device by index: add e.g. `--device 1`.
"""

import argparse
import os
import sys


def get_session():
    try:
        import pcomfortcloud
    except ImportError:
        sys.exit(
            "Missing dependency. Install it with:\n\n"
            "    pip install pcomfortcloud\n"
        )

    user = os.environ.get("PANASONIC_USER")
    password = os.environ.get("PANASONIC_PASS")
    if not user or not password:
        sys.exit(
            "Set your credentials first:\n\n"
            '    export PANASONIC_USER="you@example.com"\n'
            '    export PANASONIC_PASS="your-password"\n'
        )

    session = pcomfortcloud.Session(user, password, verifySsl=True)
    session.login()
    return session


def cmd_list(session, _args):
    devices = session.get_devices()
    if not devices:
        print("No registered devices found on this account.")
        return
    print(f"Found {len(devices)} registered device(s):\n")
    for i, dev in enumerate(devices):
        name = dev.get("name", "?")
        model = dev.get("model", "?")
        print(f"  [{i}] {name}   model={model}   id={dev.get('id')}")


def _pick(session, args):
    devices = session.get_devices()
    if not devices:
        sys.exit("No registered devices on this account.")
    idx = args.device
    if idx < 0 or idx >= len(devices):
        sys.exit(f"--device {idx} out of range (0..{len(devices)-1}).")
    return devices[idx]


def cmd_status(session, args):
    devices = session.get_devices()
    for i, dev in enumerate(devices):
        s = session.get_device(dev["id"])
        p = s["parameters"]
        power = getattr(p["power"], "name", p["power"])
        mode = getattr(p["mode"], "name", p["mode"])
        print(
            f"[{i}] {dev.get('name','?')} ({dev.get('model','?')}): "
            f"power={power} mode={mode} "
            f"set={p.get('temperature')}°C inside={p.get('temperatureInside')}°C "
            f"outside={p.get('temperatureOutside')}°C"
        )


def _set_power(session, args, on: bool):
    import pcomfortcloud
    dev = _pick(session, args)
    state = session.get_device(dev["id"])["parameters"]
    state["power"] = (
        pcomfortcloud.constants.Power.On if on
        else pcomfortcloud.constants.Power.Off
    )
    session.set_device(dev["id"], **state)
    print(f"{dev.get('name','device')} turned {'ON' if on else 'OFF'}.")


def cmd_on(session, args):
    _set_power(session, args, True)


def cmd_off(session, args):
    _set_power(session, args, False)


def cmd_temp(session, args):
    dev = _pick(session, args)
    state = session.get_device(dev["id"])["parameters"]
    state["temperature"] = float(args.value)
    session.set_device(dev["id"], **state)
    print(f"{dev.get('name','device')} target temperature set to {args.value}°C.")


def main():
    parser = argparse.ArgumentParser(description="Panasonic Comfort Cloud client")
    parser.add_argument("--device", type=int, default=0,
                        help="device index (default 0, see `list`)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    sub.add_parser("on")
    sub.add_parser("off")
    p_temp = sub.add_parser("temp")
    p_temp.add_argument("value", help="target temperature in °C, e.g. 24")

    args = parser.parse_args()
    session = get_session()

    {
        "list": cmd_list,
        "status": cmd_status,
        "on": cmd_on,
        "off": cmd_off,
        "temp": cmd_temp,
    }[args.command](session, args)


if __name__ == "__main__":
    main()
