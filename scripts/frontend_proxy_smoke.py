"""Prove Docker DNS recovery using isolated HTTP stubs, without touching app data.

Requires the built Compose frontend image and docattentionlens-backend:0.1.0 locally.
These responses test transport only; they are not model evidence.
"""
import ipaddress
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def docker(*args):
    result = subprocess.run(['docker', *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def wait_for(url, expected):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.read().decode() == expected:
                    return
        except OSError:
            pass
        time.sleep(.5)
    raise RuntimeError(f'Proxy did not resolve the backend serving {expected}')


def main():
    name = 'docattentionlens-proxy-check-' + uuid4().hex[:12]
    backend, frontend = name + '-api', name + '-frontend'
    created = []
    network_created = False
    try:
        network_ids = docker('network', 'ls', '-q').splitlines()
        existing = json.loads(docker('network', 'inspect', *network_ids))
        occupied = [ipaddress.ip_network(config['Subnet'])
                    for item in existing for config in (item['IPAM']['Config'] or [])
                    if config.get('Subnet')]
        subnet = next(ipaddress.ip_network(f'10.254.{index}.0/24')
                      for index in range(256)
                      if not any(ipaddress.ip_network(f'10.254.{index}.0/24').overlaps(other)
                                 for other in occupied if other.version == 4))
        docker('network', 'create', '--subnet', str(subnet), name)
        network_created = True
        network = json.loads(docker('network', 'inspect', name))[0]
        subnet = ipaddress.ip_network(network['IPAM']['Config'][0]['Subnet'])
        addresses = [str(subnet.network_address + offset) for offset in (10, 11)]

        def start_backend(address, marker):
            program = (
                'from http.server import BaseHTTPRequestHandler, HTTPServer\n'
                'class Handler(BaseHTTPRequestHandler):\n'
                ' def do_GET(self):\n'
                '  self.send_response(200); self.end_headers(); '
                f'self.wfile.write({marker.encode()!r})\n'
                "HTTPServer(('0.0.0.0',8000),Handler).serve_forever()\n"
            )
            created.append(backend)
            docker('run', '-d', '--name', backend, '--network', name,
                   '--network-alias', 'api', '--ip', address,
                   'docattentionlens-backend:0.1.0', 'python', '-c', program)

        start_backend(addresses[0], 'backend-address-one')
        created.append(frontend)
        docker('run', '-d', '--name', frontend, '--network', name,
               '-p', '127.0.0.1::80',
               'docattentionlens-frontend')
        container = json.loads(docker('inspect', frontend))[0]
        port = container['NetworkSettings']['Ports']['80/tcp'][0]['HostPort']
        base = 'http://127.0.0.1:' + port
        wait_for(base + '/api/v1/datasets', 'backend-address-one')
        docker('rm', '-f', backend)
        start_backend(addresses[1], 'backend-address-two')
        wait_for(base + '/api/v1/datasets', 'backend-address-two')
        wait_for(base + '/health/ready', 'backend-address-two')
        current = json.loads(docker('inspect', frontend))[0]
        assert current['Id'] == container['Id']
        assert current['State']['StartedAt'] == container['State']['StartedAt']
        evidence = {
            'backend_addresses': addresses,
            'frontend_restarted': False,
            'api_and_health_recovered': True,
            'response_kind': 'isolated transport stubs, not model evidence',
        }
        path = ROOT / 'var/verification/frontend-proxy-smoke.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(evidence))
    finally:
        for container_name in reversed(list(dict.fromkeys(created))):
            subprocess.run(['docker', 'rm', '-f', container_name],
                           capture_output=True, check=False)
        if network_created:
            subprocess.run(['docker', 'network', 'rm', name],
                           capture_output=True, check=False)


if __name__ == '__main__':
    main()
