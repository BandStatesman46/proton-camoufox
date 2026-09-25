"""Create a Proton Free account; complete verification in the visible browser."""

import os

from proton_camoufox import ProtonMailClient


def main() -> None:
    username = os.getenv("PROTON_USERNAME")
    password = os.getenv("PROTON_PASSWORD")
    if not username or not password:
        raise SystemExit("Set PROTON_USERNAME and PROTON_PASSWORD before running.")

    with ProtonMailClient(username=username, password=password, headless=False) as client:
        result = client.register(
            display_name=os.getenv("PROTON_DISPLAY_NAME") or None,
            recovery_phrase_file=os.getenv("PROTON_RECOVERY_FILE") or None,
            allow_manual_verification=True,
        )
        print("Account created. Inbox:", result.inbox_url)
        if result.recovery_phrase_file:
            print("Recovery file saved to:", result.recovery_phrase_file)


if __name__ == "__main__":
    main()
