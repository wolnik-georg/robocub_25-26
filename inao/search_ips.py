# -*- encoding: UTF-8 -*-
"""
Find NAO robot IP addresses on the local network
Scans common NAO ports (9559) to detect robots
"""

import socket
import subprocess
import re
import sys


def get_local_network():
    """Get the local network IP range"""
    try:
        # Get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        # Extract network base (e.g., 192.168.1)
        network_base = ".".join(local_ip.split(".")[:-1])
        return network_base, local_ip
    except Exception as e:
        print("Error getting local network: {}".format(e))
        return None, None


def check_nao_port(ip, port=9559, timeout=0.5):
    """Check if NAO port is open on given IP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


def scan_network(network_base, start=1, end=255):
    """Scan network range for NAO robots"""
    print(
        "Scanning network {}.{}-{} for NAO robots...".format(network_base, start, end)
    )
    print("This may take a minute...\n")

    found_robots = []

    for i in range(start, end + 1):
        ip = "{}.{}".format(network_base, i)
        sys.stdout.write(
            "\rScanning: {} ({}/{})".format(ip, i - start + 1, end - start + 1)
        )
        sys.stdout.flush()

        if check_nao_port(ip):
            found_robots.append(ip)
            print("\n[FOUND] NAO robot at: {}".format(ip))

    print("\n")
    return found_robots


def ping_sweep(network_base):
    """Quick ping sweep to find active hosts"""
    print("Running quick ping sweep on {}.*...".format(network_base))
    active_hosts = []

    try:
        # Use nmap if available (faster)
        result = subprocess.check_output(
            ["nmap", "-sn", "{}.0/24".format(network_base)], stderr=subprocess.STDOUT
        )
        # Parse nmap output for active hosts
        for line in result.split("\n"):
            match = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if match:
                active_hosts.append(match.group(1))
    except (subprocess.CalledProcessError, OSError):
        # nmap not available, use ping
        print("(nmap not found, using slower ping method)")
        for i in range(1, 255):
            ip = "{}.{}".format(network_base, i)
            try:
                subprocess.check_output(
                    ["ping", "-c", "1", "-W", "1", ip], stderr=subprocess.STDOUT
                )
                active_hosts.append(ip)
            except subprocess.CalledProcessError:
                pass

    print("Found {} active hosts\n".format(len(active_hosts)))
    return active_hosts


def main():
    print("=" * 50)
    print("NAO Robot IP Finder")
    print("=" * 50)
    print()

    # Get local network
    network_base, local_ip = get_local_network()
    if not network_base:
        print("Could not determine local network.")
        print("Please enter network manually (e.g., 192.168.1):")
        network_base = raw_input("> ").strip()
    else:
        print("Your IP: {}".format(local_ip))
        print("Network: {}.*".format(network_base))
        print()

    # Ask for scan method
    print("Choose scan method:")
    print("1. Quick scan (ping sweep + NAO port check)")
    print("2. Full scan (check all IPs 1-254)")
    print("3. Custom range")
    choice = raw_input("Enter choice (1-3) [1]: ").strip() or "1"

    found_robots = []

    if choice == "1":
        # Quick scan: ping sweep first, then check NAO ports
        active_hosts = ping_sweep(network_base)
        if active_hosts:
            print("Checking active hosts for NAO port (9559)...")
            for ip in active_hosts:
                sys.stdout.write("\rChecking: {}".format(ip))
                sys.stdout.flush()
                if check_nao_port(ip):
                    found_robots.append(ip)
                    print("\n[FOUND] NAO robot at: {}".format(ip))
            print()

    elif choice == "2":
        # Full scan
        found_robots = scan_network(network_base, 1, 254)

    elif choice == "3":
        # Custom range
        start = int(raw_input("Start IP (e.g., 100): ").strip() or "1")
        end = int(raw_input("End IP (e.g., 150): ").strip() or "254")
        found_robots = scan_network(network_base, start, end)

    # Summary
    print("=" * 50)
    print("Scan Complete")
    print("=" * 50)

    if found_robots:
        print("Found {} NAO robot(s):".format(len(found_robots)))
        for ip in found_robots:
            print("  - {}".format(ip))
        print()
        print("You can now connect using:")
        print("  python continuous_vision_test_v4.py {}".format(found_robots[0]))
    else:
        print("No NAO robots found.")
        print()
        print("Troubleshooting:")
        print("  - Make sure NAO is powered on")
        print("  - Check NAO is connected to same network")
        print("  - Try checking NAO's IP via chest button")
        print("  - Firewall might be blocking port 9559")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")
