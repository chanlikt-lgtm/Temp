#!/usr/bin/env python3
"""
Scan the local network and report whether a Panasonic air conditioner
(its Wi-Fi adapter, e.g. CZ-TACG1 / DNSK / built-in module) is online.

How it works
------------
1. Figures out your local IPv4 subnet (e.g. 192.168.1.0/24).
2. "Warms up" the ARP table by pinging every host on that subnet.
3. Reads the system ARP table (IP <-> MAC pairs).
4. Matches each MAC against known Panasonic manufacturer prefixes (OUIs).
5. Prints any Panasonic device found and confirms it is reachable (online).

Usage
-----
    python3 scan_panasonic_ac.py                # auto-detect subnet
    python3 scan_panasonic_ac.py 192.168.1.0/24 # scan a specific subnet

No extra libraries required (standard library only). Works on Linux and macOS.
On Windows it also works but relies on `arp -a` output.
"""

import concurrent.futures
import ipaddress
import platform
import re
import socket
import subprocess
import sys

# ---------------------------------------------------------------------------
# Known Panasonic / Panasonic-group MAC address prefixes (OUIs).
# A MAC starting with one of these very likely belongs to a Panasonic device.
# Source: IEEE OUI registry (Panasonic Corporation / Panasonic Appliances etc.)
# ---------------------------------------------------------------------------
PANASONIC_OUIS = {
    "00:80:f0", "00:0b:97", "00:13:49", "00:1a:c1", "00:1b:d3",
    "00:1f:3b", "00:21:36", "00:22:a6", "00:23:d7", "00:24:fe",
    "00:26:66", "00:80:45", "08:00:23", "10:4f:58", "20:c6:eb",
    "34:4b:50", "34:a8:eb", "3c:95:09", "40:f3:08", "48:a9:d2",
    "4c:fb:45", "54:22:f8", "58:26:0e", "5c:f8:a1", "60:38:0e",
    "64:5d:86", "70:af:25", "74:77:12", "78:bd:bc", "84:25:db",
    "88:b1:e1", "8c:c1:21", "9c:81:0e", "a0:f3:c1", "a4:07:b6",
    "a4:b1:c1", "ac:9a:22", "b0:1c:91", "b4:a9:fc", "bc:c6:db",
    "c0:3f:0e", "c8:08:e9", "cc:3a:61", "d0:3a:d7", "d8:d2:7c",
    "dc:85:de", "e0:2c:b2", "e8:c3:20", "ec:63:e5", "f0:03:8c",
    "f4:85:e4", "f8:a9:d0",
}


def normalize_mac(mac: str) -> str:
    """Return MAC as lowercase colon-separated, or '' if not a MAC."""
    hexchars = re.sub(r"[^0-9a-fA-F]", "", mac)
    if len(hexchars) != 12:
        return ""
    return ":".join(hexchars[i:i + 2] for i in range(0, 12, 2)).lower()


def oui_of(mac: str) -> str:
    return mac[:8]  # first 3 octets "xx:xx:xx"


def local_subnet() -> str:
    """Best-effort detection of the local /24 subnet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually send packets; just picks the outbound interface.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    net = ipaddress.ip_network(ip + "/24", strict=False)
    return str(net)


def ping(host: str) -> bool:
    """Ping once, quick timeout. Returns True if host replied."""
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", "1", "-w", "500", host]
    else:
        # -c 1 one packet, -W 1 wait 1s (Linux), macOS uses -t
        cmd = ["ping", "-c", "1", "-W", "1", host]
    try:
        return subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
    except Exception:
        return False


def warm_arp(subnet: str) -> None:
    """Ping every host in the subnet in parallel to populate the ARP table."""
    hosts = [str(h) for h in ipaddress.ip_network(subnet, strict=False).hosts()]
    print(f"Scanning {len(hosts)} addresses on {subnet} ... (this takes ~20s)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as pool:
        list(pool.map(ping, hosts))


def read_arp_table():
    """Return list of (ip, mac) tuples from the OS ARP table."""
    try:
        out = subprocess.run(
            ["arp", "-a"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        # Linux fallback
        out = subprocess.run(
            ["ip", "neigh"], capture_output=True, text=True
        ).stdout

    entries = []
    for line in out.splitlines():
        ip_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", line)
        mac_match = re.search(r"([0-9a-fA-F]{2}([:-])[0-9a-fA-F]{2}(\2[0-9a-fA-F]{2}){4})", line)
        if ip_match and mac_match:
            mac = normalize_mac(mac_match.group(1))
            if mac:
                entries.append((ip_match.group(1), mac))
    return entries


def main():
    subnet = sys.argv[1] if len(sys.argv) > 1 else local_subnet()
    warm_arp(subnet)

    entries = read_arp_table()
    found = [(ip, mac) for ip, mac in entries if oui_of(mac) in PANASONIC_OUIS]

    print("\n" + "=" * 52)
    if found:
        print("✅ Panasonic device(s) detected and ONLINE:\n")
        for ip, mac in found:
            online = ping(ip)
            status = "ONLINE  ✅" if online else "no ping reply ⚠️"
            print(f"   IP: {ip:15s}  MAC: {mac}  -> {status}")
        print("\nIf one of these is your air conditioner, it is on the network.")
    else:
        print("❌ No Panasonic device found on the network.")
        print("   Possible reasons:")
        print("   - The AC's Wi-Fi is off or not connected to this network.")
        print("   - Its MAC prefix isn't in the known list (tell me the MAC).")
        print("   - It is on a different subnet / Wi-Fi band (2.4GHz vs 5GHz).")
        print("\n   All devices seen on the network:")
        for ip, mac in sorted(entries, key=lambda e: tuple(int(x) for x in e[0].split('.'))):
            print(f"     {ip:15s}  {mac}")
    print("=" * 52)


if __name__ == "__main__":
    main()
