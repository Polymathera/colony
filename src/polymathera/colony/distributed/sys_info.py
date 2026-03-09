"""
Is there a way to detect if this process is running in a data center and detect the topology of
where this process is running (which rack and so on) so that a centralized process can allocate
agent teams to run close to each other and minimize network traffic?
"""
import logging
import os
import platform
import time
from typing import Any

import aiohttp


async def get_sys_info() -> dict[str, Any]:
    sys_info = {}
    sys_info["timestamp"] = time.time()
    sys_info["os_name"] = platform.system()
    sys_info["os_version"] = platform.version()
    sys_info["os_architecture"] = platform.machine()
    # Detect if running in a data center and gather topology information
    sys_info["is_data_center"] = False
    sys_info["topology"] = {}

    try:
        # Check for common data center virtualization
        if os.path.exists("/sys/class/dmi/id/product_name"):
            with open("/sys/class/dmi/id/product_name") as f:
                product_name = f.read().strip().lower()
                if any(
                    name in product_name
                    for name in ["vmware", "xen", "kvm", "virtualbox"]
                ):
                    sys_info["is_data_center"] = True

        # Try to get rack and node information
        if os.path.exists("/etc/machine-id"):
            with open("/etc/machine-id") as f:
                machine_id = f.read().strip()
            sys_info["topology"]["machine_id"] = machine_id

        # Check for OpenStack metadata
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "http://169.254.169.254/openstack/latest/meta_data.json", timeout=2
                ) as r:
                    if r.status == 200:
                        sys_info["is_data_center"] = True
                        sys_info["topology"]["openstack"] = await r.json()
            except aiohttp.ClientError:
                pass

            # Check for AWS EC2 metadata
            try:
                async with session.get(
                    "http://169.254.169.254/latest/meta-data/", timeout=2
                ) as r:
                    if r.status == 200:
                        sys_info["is_data_center"] = True
                        instance_id = await (
                            await session.get(
                                "http://169.254.169.254/latest/meta-data/instance-id"
                            )
                        ).text()
                        availability_zone = await (
                            await session.get(
                                "http://169.254.169.254/latest/meta-data/placement/availability-zone"
                            )
                        ).text()
                        sys_info["topology"]["aws_ec2"] = {
                            "instance-id": instance_id,
                            "placement": {"availability-zone": availability_zone},
                        }
            except aiohttp.ClientError:
                pass

        # If running in a data center, try to get network topology
        if sys_info["is_data_center"]:
            import netifaces

            sys_info["topology"]["network"] = {
                iface: netifaces.ifaddresses(iface) for iface in netifaces.interfaces()
            }

    except Exception as e:
        logging.warning(f"Error while gathering system topology information: {e!s}")
