"""Log in to Proton Mail through a proxy with optional proxy credentials."""

import os

from proton_camoufox import ProtonMailClient, ProxyConfig


def main() -> None:
    username = os.getenv("PROTON_USERNAME")
    password = os.getenv("PROTON_PASSWORD")
    proxy_server = os.getenv("PROXY_SERVER")
    if not username or not password or not proxy_server:
        raise SystemExit("Set PROTON_USERNAME, PROTON_PASSWORD and PROXY_SERVER before running.")

    proxy = ProxyConfig(
        server=proxy_server,
        username=os.getenv("PROXY_USERNAME") or None,
        password=os.getenv("PROXY_PASSWORD") or None,
    )
    with ProtonMailClient(
        username=username, password=password, proxy=proxy, headless=False,
    ) as client:
        client.login(allow_manual_verification=True)
        for message in client.list_messages(limit=10):
            print(message.subject)


if __name__ == "__main__":
    main()
