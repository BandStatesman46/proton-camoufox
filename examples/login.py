"""Log in to Proton Mail and print the latest message subjects."""

import os

from proton_camoufox import ProtonMailClient


def main() -> None:
    username = os.getenv("PROTON_USERNAME")
    password = os.getenv("PROTON_PASSWORD")
    if not username or not password:
        raise SystemExit("Set PROTON_USERNAME and PROTON_PASSWORD before running.")

    with ProtonMailClient(username=username, password=password, headless=False) as client:
        client.login(allow_manual_verification=True)
        for message in client.list_messages(limit=10):
            print(message.subject)


if __name__ == "__main__":
    main()
