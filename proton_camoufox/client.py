from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from camoufox.sync_api import Camoufox
import cv2
import numpy as np
from PIL import Image
from playwright.sync_api import BrowserContext, Page, TimeoutError as PlaywrightTimeoutError


class ProtonCamoufoxError(RuntimeError):
    pass


class LoginFailed(ProtonCamoufoxError):
    pass


class RegistrationFailed(ProtonCamoufoxError):
    pass


class VerificationRequired(ProtonCamoufoxError):
    """Raised when Proton asks for CAPTCHA, 2FA, or another interactive check."""


class MessageNotFound(ProtonCamoufoxError):
    pass


@dataclass(slots=True)
class ProxyConfig:
    server: str
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)

    def to_playwright(self) -> dict:
        data = {"server": self.server}
        if self.username:
            data["username"] = self.username
        if self.password:
            data["password"] = self.password
        return data


@dataclass(slots=True)
class Attachment:
    name: str
    size: Optional[str] = None


@dataclass(slots=True)
class MessageSummary:
    index: int
    subject: str
    sender_name: str = ""
    sender_email: str = ""
    sent_at: Optional[str] = None
    unread: Optional[bool] = None
    starred: Optional[bool] = None
    has_attachments: bool = False
    attachment_count: int = 0
    item_id: Optional[str] = None

    @property
    def text(self) -> str:
        return f"{self.sender_name} — {self.subject}" if self.sender_name else self.subject


@dataclass(slots=True)
class Message:
    subject: str
    body: str
    url: str
    sender_name: str = ""
    sender_email: str = ""
    recipients_to: tuple[str, ...] = ()
    recipients_cc: tuple[str, ...] = ()
    recipients_bcc: tuple[str, ...] = ()
    sent_at: Optional[str] = None
    message_id: Optional[str] = None
    starred: Optional[bool] = None
    has_attachments: bool = False
    attachment_count: int = 0
    attachments: tuple[Attachment, ...] = ()


@dataclass(slots=True)
class RegistrationResult:
    username: str
    inbox_url: str
    display_name: Optional[str] = None
    recovery_phrase_file: Optional[str] = None


@dataclass
class ProtonMailClient:
    username: str
    password: str = field(repr=False)
    proxy: Optional[ProxyConfig] = None
    headless: bool = True
    timeout_ms: int = 30_000
    login_fields_timeout_ms: Optional[int] = None
    profile_dir: Optional[str | Path] = None
    state_file: Optional[str | Path] = None
    auto_save_state: bool = True
    geoip: bool = True
    debug_on_missing_fields: bool = False
    debug_dir: str | Path = "proton_debug"
    enable_mail_categories: bool = False
    ui_tasks_timeout_ms: int = 5_000
    captcha_detection_timeout_ms: int = 3_000
    signup_form_ready_timeout_ms: int = 60_000
    interaction_delays: bool = True
    interaction_delay_ms: int = 350

    LOGIN_URL = "https://account.proton.me/mail"
    MAIL_URL = "https://mail.proton.me/"
    SIGNUP_LANDING_URL = "https://account.proton.me/signup"
    MAIL_SERVICE_LINK_SELECTOR = (
        'a[data-testid="explore-mail"][href^="https://mail.proton.me/"]'
    )
    
    MAIL_URL_RE = re.compile(
        r"^https://mail\.proton\.me/u/\d+(?:/.*)?$"
    )

    LOGIN_URL_RE = re.compile(
        r"^https://account\.proton\.me/(?:mail|login)(?:[/?#]|$)"
    )
    
    AUTHORIZE_URL_RE = re.compile(
        r"^https://account\.proton\.me/authorize(?:\?|$)"
    )

    SERVICE_SWITCH_URL_RE = re.compile(
        r"^https://account\.proton\.me/(?:[a-z]{2}/)?switch(?:[/?#]|$)"
    )

    # Selectors on https://account.proton.me/signup use DOM attributes.
    SIGNUP_FREE_PLAN_SELECTOR = 'button.card-plan:has(strong#free-text)'
    SIGNUP_FREE_SELECTED_SELECTOR = (
        'button.card-plan[aria-pressed="true"]:has(strong#free-text)'
    )
    SIGNUP_USERNAME_SELECTOR = (
        'input#username[data-testid="input-input-element"]'
    )
    SIGNUP_EMAIL_FRAME_URL_PART = "/challenge/v4/html?Type=0&Name=email"
    SIGNUP_PASSWORD_SELECTOR = (
        'input#password[type="password"][autocomplete="new-password"]'
    )
    SIGNUP_PASSWORD_CONFIRM_SELECTOR = (
        'input#password-confirm[type="password"][autocomplete="new-password"]'
    )
    SIGNUP_SUBMIT_SELECTOR = 'form[name="account-form"] button[type="submit"]'
    RECOVERY_SWITCH_SELECTOR = '[data-testid="switch-to-copy"]'
    RECOVERY_REVEAL_SELECTOR = (
        'button[data-testid="copy-recovery-phrase"].button-solid-norm'
    )
    RECOVERY_PHRASE_TEXT_SELECTOR = (
        '[data-testid="account:recovery:generatedRecoveryPhrase"]'
    )
    RECOVERY_SWITCH_TO_PDF_SELECTOR = '[data-testid="switch-to-pdf"]'
    RECOVERY_DOWNLOAD_KIT_SELECTOR = '[data-testid="download-recovery-kit-button"]'
    RECOVERY_CHECKBOX_SELECTOR = '#understood-recovery-necessity'
    RECOVERY_CONTINUE_SELECTOR = (
        'button.w-full.button-large.button-solid-norm[type="button"]'
    )
    DISPLAY_NAME_SELECTOR = '#displayName'
    DISPLAY_NAME_SUBMIT_SELECTOR = (
        'form[name="accountForm"] button[type="submit"]'
    )
    ONBOARDING_LAST_TAB_SELECTOR = (
        'dialog.onboarding-modal button[role="tab"]'
        '[aria-controls="onboarding-3"]'
    )
    ONBOARDING_USE_SELECTOR = (
        'dialog.onboarding-modal .modal-two-content'
        ' > footer > button.button-solid-norm[type="button"]'
    )

    STATE_VERSION = 1

    # Приватные поля, не участвующие в автогенерируемом __init__/repr.
    _manager: Any = field(default=None, init=False, repr=False, compare=False)
    _browser_or_context: Any = field(default=None, init=False, repr=False, compare=False)
    _context: Optional[BrowserContext] = field(default=None, init=False, repr=False, compare=False)
    _page: Optional[Page] = field(default=None, init=False, repr=False, compare=False)
    _owns_context: bool = field(default=False, init=False, repr=False, compare=False)
    _post_signup_onboarding_completed: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.profile_dir and self.state_file:
            raise ValueError(
                "Используйте либо profile_dir, либо state_file. "
                "state_file предназначен как лёгкая замена persistent profile."
            )

    # ---------- public page accessor (обратная совместимость) ----------

    @property
    def page(self) -> Optional[Page]:
        return self._page

    # ---------- context manager ----------

    def __enter__(self) -> "ProtonMailClient":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- lifecycle ----------

    def start(self) -> "ProtonMailClient":
        if self._manager is not None:
            return self

        effective_headless = self._effective_headless()

        kwargs: dict[str, Any] = {
            "headless": effective_headless,
            "geoip": self.geoip if self.proxy else False,
            "block_webrtc": True,
        }

        if self.proxy:
            kwargs["proxy"] = self.proxy.to_playwright()

        if self.profile_dir:
            kwargs["persistent_context"] = True
            kwargs["user_data_dir"] = str(Path(self.profile_dir).resolve())

        self._manager = Camoufox(**kwargs)
        self._browser_or_context = self._manager.__enter__()

        if self.profile_dir:
            self._context = self._browser_or_context
            self._owns_context = False
        else:
            playwright_state = self._read_state_file()
            context_kwargs: dict[str, Any] = {}

            if playwright_state:
                context_kwargs["storage_state"] = playwright_state

            self._context = self._browser_or_context.new_context(**context_kwargs)
            self._owns_context = True

        self._context.on("page", self._handle_new_page)

        self._page = self._acquire_single_page()
        self._page.set_default_timeout(self.timeout_ms)

        return self

    def close(self) -> None:
        if self._manager is None:
            return

        manager = self._manager

        try:
            if self.state_file and self.auto_save_state and self._context is not None:
                try:
                    self.save_state()
                except Exception:
                    pass

            if self._context is not None:
                try:
                    self._context.remove_listener("page", self._handle_new_page)
                except Exception:
                    pass

            self._close_all_pages()

        finally:
            self._manager = None
            self._browser_or_context = None
            self._context = None
            self._page = None
            self._owns_context = False

            manager.__exit__(None, None, None)

    # ---------- page/window management ----------

    def _acquire_single_page(self) -> Page:
        """
        Возвращает единственную «рабочую» страницу контекста.

        Camoufox/Playwright при старте контекста нередко уже создают
        начальную пустую страницу (about:blank). Если поверх неё мы
        вызываем context.new_page(), у нас на мгновение оказывается
        два окна — и одно из них сразу закрывается. Поэтому вместо
        безусловного создания новой страницы мы переиспользуем уже
        существующую, если она есть.
        """
        context = self._require_context()

        existing = [p for p in context.pages if not p.is_closed()]

        if existing:
            page = existing[0]
        else:
            page = context.new_page()

        self._close_pages_except(context, page)

        return page

    def _handle_new_page(self, new_page: Page) -> None:
        if self._page is None or new_page is self._page:
            return

        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=5_000)
        except PlaywrightTimeoutError:
            pass
        except Exception:
            pass

        if new_page.is_closed():
            return

    def _close_pages_except(self, context: BrowserContext, keep: Page) -> None:
        for page in list(context.pages):
            if page is keep or page.is_closed():
                continue
            self._safe_close(page)

    def _close_extra_pages(self) -> None:
        if self._context is None:
            return
        self._close_pages_except(self._context, self._page)

    def _close_all_pages(self) -> None:
        if self._context is None:
            return
        for page in list(self._context.pages):
            self._safe_close(page)

    @staticmethod
    def _safe_close(page: Page) -> None:
        try:
            if not page.is_closed():
                page.close()
        except Exception:
            pass

    # ---------- transient UI task processing ----------

    def process_ui_tasks(
        self,
        timeout_ms: Optional[int] = None,
    ) -> tuple[str, ...]:
        page = self._require_page()
        timeout_ms = 0 if timeout_ms is None else max(0, timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000
        handled: list[str] = []

        handlers = (
            (
                "post_signup_onboarding",
                self._handle_post_signup_onboarding,
            ),
            (
                "inbox_organization",
                self._handle_inbox_organization_modal,
            ),
        )

        while not page.is_closed():
            handled_this_pass = False

            for name, handler in handlers:
                if name in handled:
                    continue

                try:
                    if handler(page):
                        handled.append(name)
                        handled_this_pass = True
                except Exception:
                    continue

            if time.monotonic() >= deadline:
                break

            page.wait_for_timeout(100 if handled_this_pass else 250)

        return tuple(handled)

    def _handle_post_signup_onboarding(
        self, page: Page, *, allow_hidden: bool = False
    ) -> bool:
        last_tab = page.locator(self.ONBOARDING_LAST_TAB_SELECTOR).first
        if last_tab.count() == 0:
            return False

        dialog = page.locator('dialog.onboarding-modal').first
        if not dialog.is_visible() and not allow_hidden:
            return False

        if last_tab.get_attribute("aria-selected") != "true":
            if not last_tab.is_visible():
                return False
            self._interaction_pause(page)
            self._click_ui(page, last_tab)

        use_button = page.locator(self.ONBOARDING_USE_SELECTOR).first
        if use_button.count() == 0:
            return False
        if not dialog.is_visible() and allow_hidden:
            use_button.dispatch_event("click")
            try:
                dialog.wait_for(state="detached", timeout=5_000)
            except PlaywrightTimeoutError:
                return False
            self._post_signup_onboarding_completed = True
            return True

        try:
            use_button.wait_for(state="visible", timeout=5_000)
        except PlaywrightTimeoutError:
            return False

        self._interaction_pause(page)
        self._click_ui(page, use_button)

        try:
            last_tab.wait_for(state="hidden", timeout=5_000)
        except PlaywrightTimeoutError:
            return False

        self._post_signup_onboarding_completed = True
        return True

    def _handle_inbox_organization_modal(self, page: Page) -> bool:
        title = page.get_by_role(
            "heading",
            name="Your inbox, automatically organized",
            exact=True,
        )

        if title.count() == 0 or not title.first.is_visible():
            return False

        button_name = (
            "Yes, organize it"
            if self.enable_mail_categories
            else "Keep inbox as before"
        )
        button = page.get_by_role(
            "button",
            name=button_name,
            exact=True,
        ).first

        if not button.is_visible():
            return False

        self._click_ui(page, button)

        try:
            title.first.wait_for(state="hidden", timeout=5_000)
        except PlaywrightTimeoutError:
            pass

        return True

    # ---------- registration ----------

    def register(
        self,
        *,
        display_name: Optional[str] = None,
        recovery_phrase_file: Optional[str | Path] = None,
        allow_manual_verification: bool = True,
        verification_timeout_ms: Optional[int] = None,
    ) -> RegistrationResult:
        """
        Create one Proton Free account using the configured credentials.

        verification_timeout_ms=None waits
        without a time limit while the verification dialog remains open.
        """
        page = self._require_page()
        self._post_signup_onboarding_completed = False

        page.goto(self.SIGNUP_LANDING_URL, wait_until="domcontentloaded")
        self._raise_if_signup_blocked(page)

        free_plan = self._wait_for_any(
            page,
            self.SIGNUP_FREE_PLAN_SELECTOR,
            timeout_ms=self.timeout_ms,
        )
        if free_plan is None:
            self._raise_if_signup_blocked(page)
            raise RegistrationFailed("Не найдена карточка тарифа Proton Free.")
        self._interaction_pause(page)
        self._click_ui(page, free_plan)

        selected_free = self._wait_for_any(
            page,
            self.SIGNUP_FREE_SELECTED_SELECTOR,
            timeout_ms=min(self.timeout_ms, 5_000),
        )
        if selected_free is None:
            raise RegistrationFailed("Тариф Proton Free не был выбран.")

        self._fill_signup_form(page)

        saved_phrase_path: Optional[str] = None
        completed_steps: set[str] = set()

        for _ in range(4):
            state = self._wait_for_signup_checkpoint(
                max(self.timeout_ms, 120_000) if completed_steps else self.timeout_ms,
                allow_manual_verification,
                verification_timeout_ms,
                skip_states=completed_steps,
            )
            if state == "inbox":
                break
            if state == "recovery":
                saved_phrase_path = self._handle_recovery_phrase_step(
                    recovery_phrase_file
                )
            elif state == "display_name":
                self._handle_display_name_step(display_name)
            elif state == "service_selection":
                self._handle_service_selection_step()
            else:
                raise RegistrationFailed(f"Неизвестный этап регистрации: {state}")
            completed_steps.add(state)
        else:
            raise RegistrationFailed(
                f"Регистрация не завершилась переходом в Proton Mail. "
                f"Текущий URL: {self._require_page().url}"
            )

        page = self._require_page()
        self.process_ui_tasks(self.ui_tasks_timeout_ms)
        self._maybe_autosave_state()

        return RegistrationResult(
            username=self.username,
            inbox_url=page.url,
            display_name=display_name,
            recovery_phrase_file=saved_phrase_path,
        )

    def _fill_signup_form(self, page: Page) -> None:
        self._interaction_pause(page)
        self._wait_for_editable_signup_field(
            page,
            self.SIGNUP_USERNAME_SELECTOR,
            "username",
            self.username,
        )
        self._interaction_pause(page)
        password_input = self._wait_for_editable_signup_field(
            page,
            self.SIGNUP_PASSWORD_SELECTOR,
            "password",
            self.password,
        )

        confirm_input = self._wait_for_any(
            page,
            self.SIGNUP_PASSWORD_CONFIRM_SELECTOR,
            timeout_ms=min(self.signup_form_ready_timeout_ms, 3_000),
        )
        if confirm_input is not None:
            self._interaction_pause(page)
            confirm_input = self._wait_for_editable_signup_field(
                page,
                self.SIGNUP_PASSWORD_CONFIRM_SELECTOR,
                "password-confirm",
                self.password,
            )
            try:
                confirm_input.press("Tab", timeout=1_000)
            except PlaywrightTimeoutError:
                pass
        else:
            try:
                password_input.press("Tab", timeout=1_000)
            except PlaywrightTimeoutError:
                pass

        page.wait_for_timeout(400)
        self._raise_if_registration_error(page)

        submit = self._wait_for_any(
            page,
            self.SIGNUP_SUBMIT_SELECTOR,
            timeout_ms=self.timeout_ms,
        )
        if submit is None:
            raise RegistrationFailed("Не найдена кнопка отправки формы регистрации.")
        self._interaction_pause(page)
        try:
            self._click_ui(page, submit, timeout_ms=self.timeout_ms)
        except PlaywrightTimeoutError as exc:
            self._raise_if_registration_error(page)
            raise RegistrationFailed(
                "Не удалось отправить форму регистрации: кнопка недоступна "
                "или не отвечает."
            ) from exc

    def _wait_for_editable_signup_field(
        self,
        page: Page,
        selector: str,
        field_name: str,
        value: str,
    ):
        deadline = (
            time.monotonic()
            + self.signup_form_ready_timeout_ms / 1000
        )
        last_readonly = False

        while time.monotonic() < deadline:
            self._raise_if_signup_blocked(page)
            frames = list(page.frames)
            if field_name == "username":
                frames.sort(
                    key=lambda frame: (
                        self.SIGNUP_EMAIL_FRAME_URL_PART not in frame.url
                    )
                )

            for frame in frames:
                if frame.is_detached():
                    continue

                try:
                    matches = frame.locator(selector)
                    count = min(matches.count(), 10)
                except Exception:
                    continue

                for index in range(count):
                    locator = matches.nth(index)
                    try:
                        if locator.get_attribute("readonly") is not None:
                            last_readonly = True

                        if (
                            locator.is_visible()
                            and locator.is_enabled()
                            and locator.is_editable()
                        ):
                            remaining_ms = max(
                                1, int((deadline - time.monotonic()) * 1000)
                            )
                            attempt_ms = (
                                min(1_000, remaining_ms)
                                if not self.interaction_delays
                                else min(
                                    remaining_ms,
                                    max(
                                        2_000,
                                        len(value) * max(
                                            20, min(120, self.interaction_delay_ms // 5)
                                        ) + 1_500,
                                    ),
                                )
                            )
                            self._enter_text(
                                page, locator, value,
                                timeout_ms=attempt_ms,
                            )
                            return locator
                    except Exception:
                        continue

            page.wait_for_timeout(100)

        detail = (
            " (поле осталось readonly)"
            if last_readonly
            else ""
        )

        if self.debug_on_missing_fields:
            try:
                self.dump_dom_debug(
                    f"signup_{field_name}_not_editable"
                )
            except Exception:
                pass

        raise RegistrationFailed(
            f"Не удалось заполнить поле {field_name} за "
            f"{self.signup_form_ready_timeout_ms / 1000:.1f} сек."
            f"{detail} Проверьте загрузку формы регистрации Proton."
        )

    def _interaction_pause(
        self,
        page: Page,
        multiplier: float = 1.0,
    ) -> None:
        if not self.interaction_delays:
            return

        delay_ms = max(
            0,
            int(self.interaction_delay_ms * multiplier),
        )
        if delay_ms:
            page.wait_for_timeout(delay_ms)

    def _move_cursor_to(self, page: Page, locator) -> None:
        if not self.interaction_delays:
            return
        try:
            box = locator.bounding_box(timeout=500)
            if box:
                page.mouse.move(
                    box["x"] + box["width"] / 2,
                    box["y"] + box["height"] / 2,
                    steps=8,
                )
        except Exception:
            pass

    def _click_ui(self, page: Page, locator, timeout_ms: Optional[int] = None) -> None:
        self._move_cursor_to(page, locator)
        if timeout_ms is None:
            locator.click()
        else:
            locator.click(timeout=timeout_ms)

    def _enter_text(
        self, page: Page, locator, value: str,
        timeout_ms: Optional[int] = None,
    ) -> None:
        kwargs = {"timeout": timeout_ms} if timeout_ms is not None else {}
        if not self.interaction_delays:
            locator.fill(value, **kwargs)
            return

        self._move_cursor_to(page, locator)
        locator.click(**kwargs)
        locator.fill("", **kwargs)
        key_delay_ms = max(20, min(120, self.interaction_delay_ms // 5))
        locator.press_sequentially(value, delay=key_delay_ms, **kwargs)

    def _wait_for_signup_checkpoint(
        self,
        timeout_ms: int,
        allow_manual_verification: bool,
        verification_timeout_ms: Optional[int] = None,
        skip_states: Optional[set[str]] = None,
    ) -> str:
        deadline: Optional[float] = time.monotonic() + timeout_ms / 1000
        verification_active = False
        previous_verification_kind: Optional[str] = None
        skip_states = skip_states or set()

        while deadline is None or time.monotonic() < deadline:
            self._raise_if_signup_blocked(self._require_page())
            if self._wait_for_mail(self._require_page(), 250):
                return "inbox"

            page = self._require_page()
            verification_kind = self._signup_verification_kind(page)
            verification_visible = verification_kind is not None
            if not verification_visible:
                self._raise_if_registration_error(page)

            if self.SERVICE_SWITCH_URL_RE.match(page.url):
                if "service_selection" not in skip_states:
                    return "service_selection"

            if "display_name" in skip_states and "service_selection" not in skip_states:
                link = page.locator(self.MAIL_SERVICE_LINK_SELECTOR).first
                if link.count() and link.is_visible():
                    return "service_selection"

            recovery_checkbox = page.locator(
                self.RECOVERY_CHECKBOX_SELECTOR
            ).first
            if "recovery" not in skip_states and (
                recovery_checkbox.count()
                and recovery_checkbox.is_visible()
            ):
                return "recovery"

            display_name_input = page.locator(
                self.DISPLAY_NAME_SELECTOR
            ).first
            if "display_name" not in skip_states and (
                display_name_input.count()
                and display_name_input.is_visible()
            ):
                return "display_name"

            if verification_visible:
                if (
                    not allow_manual_verification
                    or self._effective_headless()
                ):
                    raise VerificationRequired(
                        "Proton запросил CAPTCHA/проверку. "
                        "Для ручного прохождения используйте headless=False "
                        "и allow_manual_verification=True."
                    )
                if not verification_active or (
                    verification_kind != previous_verification_kind
                    and verification_kind != "verification"
                ):
                    instruction = {
                        "captcha": "Пройдите CAPTCHA вручную",
                        "email": "Введите код подтверждения из письма вручную",
                        "phone": "Введите код подтверждения по телефону вручную",
                    }.get(verification_kind, "Пройдите проверку вручную")
                    message = (
                        f"Proton запросил дополнительную проверку: "
                        f"{instruction} в открытом браузере. "
                        "Скрипт продолжит регистрацию после её завершения."
                    )
                    if verification_kind == "captcha":
                        self._solve_captcha_if_present(page, 15000, 5)
                if not verification_active:
                    deadline = (
                        None if verification_timeout_ms is None
                        else time.monotonic() + verification_timeout_ms / 1000
                    )
                verification_active = True
                previous_verification_kind = verification_kind
            elif verification_active:
                verification_active = False
                previous_verification_kind = None
                deadline = time.monotonic() + max(timeout_ms, 120_000) / 1000

            page.wait_for_timeout(250)

        self._raise_if_registration_error(self._require_page())
        if verification_active:
            raise VerificationRequired(
                "Проверка Proton не была завершена за отведённое время."
            )
        raise RegistrationFailed(
            "Не удалось определить следующий этап регистрации Proton."
        )

    @staticmethod
    def _signup_verification_kind(page: Page) -> Optional[str]:
        for selector in ('[data-testid="verification"]', 'dialog.human-verification-modal'):
            try:
                modal = page.locator(selector).first
                if not modal.count() or not modal.is_visible():
                    continue
                for kind, tab_selector in (
                    ("captcha", '[data-testid="tab-header-captcha-button"][aria-selected="true"]'),
                    ("email", '[data-testid="tab-header-email-button"][aria-selected="true"]'),
                    ("phone", '[data-testid="tab-header-phone-button"][aria-selected="true"]'),
                    ("phone", '[data-testid="tab-header-sms-button"][aria-selected="true"]'),
                ):
                    if modal.locator(tab_selector).count():
                        return kind
                return "verification"
            except Exception:
                continue

        try:
            iframe = page.locator('iframe[src*="/core/v4/captcha"]').first
            if iframe.count() and iframe.is_visible():
                return "captcha"
        except Exception:
            pass
        return None

    @staticmethod
    def _signup_verification_visible(page: Page) -> bool:
        return ProtonMailClient._signup_verification_kind(page) is not None

    def _handle_recovery_phrase_step(
        self,
        target_file: Optional[str | Path],
    ) -> Optional[str]:
        page = self._require_page()
        saved_path: Optional[str] = None
        path = Path(target_file).expanduser().resolve() if target_file is not None else None

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

        phrase = None
        if path is None or path.suffix.lower() != ".pdf":
            phrase = self._extract_recovery_phrase(page)
            if not phrase:
                switch = page.locator(self.RECOVERY_SWITCH_SELECTOR).first
                if switch.count() and switch.is_visible():
                    self._interaction_pause(page)
                    self._click_ui(page, switch)

                reveal = self._wait_for_any(
                    page,
                    self.RECOVERY_REVEAL_SELECTOR,
                    timeout_ms=min(self.timeout_ms, 10_000),
                )
                if reveal is not None:
                    self._interaction_pause(page)
                    self._click_ui(page, reveal)

                if path is not None:
                    deadline = time.monotonic() + min(self.timeout_ms, 10_000) / 1000
                    while time.monotonic() < deadline and not phrase:
                        phrase = self._extract_recovery_phrase(page)
                        if not phrase:
                            page.wait_for_timeout(200)

        if path is not None:
            if phrase:
                path.write_text(phrase + "\n", encoding="utf-8")
                path.chmod(0o600)
                saved_path = str(path)
            else:
                switch_back = page.locator(self.RECOVERY_SWITCH_TO_PDF_SELECTOR).first
                if switch_back.count() and switch_back.is_visible():
                    self._interaction_pause(page)
                    self._click_ui(page, switch_back)

                download_button = self._wait_for_any(
                    page,
                    self.RECOVERY_DOWNLOAD_KIT_SELECTOR,
                    timeout_ms=min(self.timeout_ms, 10_000),
                )
                if download_button is None:
                    raise RegistrationFailed(
                        "Не удалось показать фразу или найти Recovery Kit PDF."
                    )

                pdf_path = (
                    path if path.suffix.lower() == ".pdf"
                    else path.with_name(path.stem + "_recovery_kit.pdf")
                )
                with tempfile.TemporaryDirectory(
                    prefix=".proton-recovery-", dir=path.parent
                ) as tmp:
                    kit_backup = Path(tmp) / "recovery-kit.pdf"
                    with page.expect_download(timeout=self.timeout_ms) as event:
                        self._interaction_pause(page)
                        self._click_ui(page, download_button)
                    event.value.save_as(kit_backup)
                    if not kit_backup.read_bytes().startswith(b"%PDF-"):
                        raise RegistrationFailed("Recovery Kit не является PDF-файлом.")
                    shutil.copyfile(kit_backup, pdf_path)
                pdf_path.chmod(0o600)
                saved_path = str(pdf_path)
                print(
                    "[proton_camoufox] Текст фразы не найден; "
                    f"Recovery Kit сохранён в PDF: {pdf_path}",
                    flush=True,
                )

        checkbox = page.locator(self.RECOVERY_CHECKBOX_SELECTOR).first
        if checkbox.count() and checkbox.is_visible() and not checkbox.is_checked():
            self._interaction_pause(page)
            checkbox.check()

        continue_button = self._wait_for_any(
            page,
            self.RECOVERY_CONTINUE_SELECTOR,
            timeout_ms=self.timeout_ms,
        )
        if continue_button is None:
            raise RegistrationFailed(
                "Не найдена кнопка продолжения после recovery phrase."
            )
        self._interaction_pause(page)
        self._click_ui(page, continue_button)
        return saved_path

    @staticmethod
    def _normalize_recovery_phrase(value: str) -> Optional[str]:
        value = re.sub(r"(?<!\w)\d{1,2}[.)]\s*", " ", value)
        words = re.findall(r"\b[a-z]{3,16}\b", value)
        return " ".join(words) if len(words) == 12 else None

    def _extract_recovery_phrase(self, page: Page) -> Optional[str]:
        candidates = (
            self.RECOVERY_PHRASE_TEXT_SELECTOR,
            '[data-testid="recovery-phrase"]',
            '[data-testid*="phrase"]:not(button)',
            '[data-testid*="recovery-phrase-word"]',
            '[class*="phrase"]:not(button)',
            'textarea[readonly]',
            'input[readonly]',
        )

        for selector in candidates:
            locator = page.locator(selector)
            words_in_group: list[str] = []
            for index in range(min(locator.count(), 24)):
                item = locator.nth(index)
                if not item.is_visible():
                    continue
                try:
                    value = (
                        item.input_value()
                        if selector in ('textarea[readonly]', 'input[readonly]')
                        else item.inner_text()
                    )
                except Exception:
                    continue
                phrase = self._normalize_recovery_phrase(value)
                if phrase:
                    return phrase
                words_in_group.append(value)
            phrase = self._normalize_recovery_phrase(" ".join(words_in_group))
            if phrase:
                return phrase

        return None

    def _handle_display_name_step(self, display_name: Optional[str]) -> None:
        page = self._require_page()
        display_input = page.locator(self.DISPLAY_NAME_SELECTOR).first
        display_input.wait_for(state="visible", timeout=self.timeout_ms)

        if display_name is not None:
            self._interaction_pause(page)
            self._enter_text(page, display_input, display_name)

        submit = self._wait_for_any(
            page,
            self.DISPLAY_NAME_SUBMIT_SELECTOR,
            timeout_ms=self.timeout_ms,
        )
        if submit is None:
            raise RegistrationFailed(
                "Не найдена кнопка сохранения отображаемого имени."
            )
        self._interaction_pause(page)
        self._click_ui(page, submit)

    def _handle_service_selection_step(self) -> None:
        page = self._require_page()
        link = self._wait_for_any(
            page,
            self.MAIL_SERVICE_LINK_SELECTOR,
            timeout_ms=self.timeout_ms,
        )
        if link is not None:
            self._interaction_pause(page)
            self._click_ui(page, link)
        elif self.SERVICE_SWITCH_URL_RE.match(page.url):
            page.goto(self.MAIL_URL, wait_until="domcontentloaded")
        else:
            raise RegistrationFailed(
                f"Не найдена ссылка на Proton Mail после регистрации: {page.url}"
            )

        if not self._wait_for_mail(page, max(self.timeout_ms, 120_000)):
            raise RegistrationFailed(
                "После выбора Proton Mail не открылась страница почты. "
                f"Текущий URL: {self._require_page().url}"
            )
        try:
            self._require_page().wait_for_load_state(
                "load", timeout=self.timeout_ms
            )
        except PlaywrightTimeoutError as exc:
            raise RegistrationFailed(
                "Proton Mail открылся, но загрузка страницы не завершилась. "
                f"Текущий URL: {self._require_page().url}"
            ) from exc
        self.process_ui_tasks(max(self.ui_tasks_timeout_ms, 10_000))
        page = self._require_page()
        dialog = page.locator('dialog.onboarding-modal').first
        if dialog.count() and dialog.is_visible():
            if not self._handle_post_signup_onboarding(page):
                raise RegistrationFailed(
                    "Не удалось нажать кнопку завершения настройки почты "
                    "в открытом окне onboarding."
                )
        if (
            not self._post_signup_onboarding_completed
            and dialog.count()
            and not dialog.is_visible()
        ):
            completed = self._handle_post_signup_onboarding(
                page, allow_hidden=True
            )
            if not completed and dialog.is_visible():
                raise RegistrationFailed(
                    "Панель выбора оформления найдена, но её кнопка не "
                    "завершила настройку почты."
                )

    def _wait_for_context_url(self, pattern, timeout_ms: int) -> Optional[Page]:
        deadline = time.monotonic() + timeout_ms / 1000

        while time.monotonic() < deadline:
            context = self._require_context()
            pages = [candidate for candidate in context.pages if not candidate.is_closed()]

            for candidate in reversed(pages):
                if pattern.match(candidate.url):
                    self._page = candidate
                    candidate.set_default_timeout(self.timeout_ms)
                    return candidate

            if not pages:
                return None
            pages[-1].wait_for_timeout(100)

        return None

    @staticmethod
    def _visible_registration_error(page: Page) -> str:
        selectors = (
            '.field-two--invalid .field-two-assist',
            '[data-testid="error-message"]',
            ".field-two-assist--error",
            '.notification--error',
            '[role="alert"][class*="error"]',
            '[role="alert"][class*="danger"]',
        )
        messages: list[str] = []
        sources = [page]
        sources.extend(
            frame for frame in getattr(page, "frames", ())
            if "Name=email" in frame.url and not frame.is_detached()
        )
        for source in sources:
            for selector in selectors:
                try:
                    locator = source.locator(selector)
                    for index in range(min(locator.count(), 20)):
                        item = locator.nth(index)
                        if not item.is_visible():
                            continue
                        value = " ".join(item.inner_text().split())
                        if value and value not in messages:
                            messages.append(value[:500])
                except Exception:
                    continue
        return "; ".join(messages)

    def _raise_if_registration_error(self, page: Page) -> None:
        error_text = self._visible_registration_error(page)
        if error_text:
            if "we are detecting potentially abusive traffic" in error_text.casefold():
                raise RegistrationFailed(
                    "Proton заблокировал дальнейшие регистрации с этой сети. "
                    f"Сообщение сервиса: {error_text}"
                )
            raise RegistrationFailed(
                f"Proton отклонил данные регистрации: {error_text}"
            )

    @staticmethod
    def _raise_if_signup_blocked(page: Page) -> None:
        alerts = page.locator(
            '[role="alert"].notification--error .notification__content'
        )
        try:
            count = min(alerts.count(), 10)
        except Exception:
            return
        for index in range(count):
            try:
                alert = alerts.nth(index)
                if not alert.is_visible():
                    continue
                message = " ".join(alert.inner_text().split())
            except Exception:
                continue
            if "we are detecting potentially abusive traffic" in message.casefold():
                raise RegistrationFailed(
                    "Proton заблокировал дальнейшие регистрации с этой сети. "
                    f"Сообщение сервиса: {message}"
                )

    # ---------- authentication ----------

    def login(self, allow_manual_verification: bool = False) -> None:
        page = self._require_page()

        if self._is_mail_url(page.url):
            self.process_ui_tasks(self.ui_tasks_timeout_ms)
            self._maybe_autosave_state()
            return

        if self._is_authorize_url(page.url):
            if self._wait_for_mail(page):
                self._maybe_autosave_state()
                return

        if not self._is_login_url(page.url):
            page.goto(
                self.LOGIN_URL,
                wait_until="domcontentloaded",
            )

        if self._is_mail_url(page.url):
            self.process_ui_tasks(self.ui_tasks_timeout_ms)
            self._maybe_autosave_state()
            return

        if self._is_authorize_url(page.url):
            if self._wait_for_mail(page, self.timeout_ms):
                self._maybe_autosave_state()
                return

        fields_timeout_ms = (
            self.login_fields_timeout_ms
            or self.timeout_ms
        )

        username_input, password_input = self._wait_for_login_fields(
            timeout_ms=fields_timeout_ms,
        )

        if username_input is None or password_input is None:
            if self._is_mail_url(page.url):
                self._maybe_autosave_state()
                return

            if self._is_authorize_url(page.url):
                if self._wait_for_mail(
                    page,
                    fields_timeout_ms,
                ):
                    self._maybe_autosave_state()
                    return

            self._check_verification()

            self._handle_missing_login_fields(
                user_missing=username_input is None,
                password_missing=password_input is None,
            )

            raise LoginFailed(
                f"Не найдены поля логина/пароля Proton за "
                f"{fields_timeout_ms / 1000:.1f} сек. "
                f"Текущий URL: {page.url}"
            )

        self._enter_text(page, username_input, self.username)
        self._enter_text(page, password_input, self.password)

        submit_button = self._wait_for_any(
            page,
            (
                'button[type="submit"]',
                'button:has-text("Sign in")',
                'button:has-text("Войти")',
            ),
            timeout_ms=fields_timeout_ms,
        )

        if submit_button is None:
            if self.debug_on_missing_fields:
                self._pause_for_dom_inspection(
                    "submit_not_found"
                )

            raise LoginFailed(
                f"Не найдена кнопка входа Proton за "
                f"{fields_timeout_ms / 1000:.1f} сек."
            )

        self._click_ui(page, submit_button)

        self._solve_captcha_if_present(page, fields_timeout_ms, 5)

        continue_button = self._wait_for_any(
            page,
            (
                'button:has-text("Continue")',
                'button:has-text("Продолжить")',
            ),
            timeout_ms=min(fields_timeout_ms, 3_000),
        )
        if continue_button:
            self._click_ui(page, continue_button)


        if self._wait_for_mail(page, self.timeout_ms):
            self._maybe_autosave_state()
            return

        if self._is_authorize_url(page.url):
            if self._wait_for_mail(
                page,
                self.timeout_ms,
            ):
                self._maybe_autosave_state()
                return

        if self._looks_like_verification():
            if (
                not allow_manual_verification
                or self._effective_headless()
            ):
                raise VerificationRequired(
                    "Proton запросил дополнительную проверку."
                )

            if not self._wait_for_mail(
                page,
                300_000,
            ):
                raise VerificationRequired(
                    "Проверка Proton не была завершена "
                    "за 5 минут."
                )

            self._maybe_autosave_state()
            return

        if self._is_login_url(page.url):
            if self._wait_for_mail(
                page,
                self.timeout_ms,
            ):
                self._maybe_autosave_state()
                return

        raise LoginFailed(
            f"Вход не завершён. Текущий URL: {page.url}"
        )

    def _solve_captcha_if_present(self, page: Page, fields_timeout_ms: int, captcha_retries: int) -> bool:
        for x in range(captcha_retries):
            frame = self.wait_for_frame(
                page,
                "pcaptcha",
                timeout_ms=min(
                    fields_timeout_ms,
                    self.captcha_detection_timeout_ms,
                ),
                required=False,
            )

            if frame is None:
                return False

            image_puzzle, canvas = self.canvas_from_iframe_to_cv2(
                page,
                "pcaptcha",
                ".challenge-canvas > canvas:nth-child(1)",
                frame=frame,
            )
            puzzle_coords = self.find_puzzle_and_slot(image_puzzle)
            self.drag_in_canvas(page, canvas, puzzle_coords["puzzle"], puzzle_coords["slot"])

            captcha_submit_button = self._wait_for_any(
                page,
                "button.btn:nth-child(1)",
                timeout_ms=fields_timeout_ms,
            )
            if captcha_submit_button is None:
                if self.debug_on_missing_fields:
                    self._pause_for_dom_inspection("captcha_submit_not_found")

                raise LoginFailed(
                    f"Не найдена кнопка подтверждения за "
                    f"{fields_timeout_ms / 1000:.1f} сек."
                )

            captcha_submit_button.click()

            is_retry = self._wait_for_any(
                page,
                (
                    ".retryContainer",
                    ".errorContainer"
                    ".final-checks-gate"
                ),
                timeout_ms=30000,
                frame=frame
            )
            if is_retry is not None:
                retry_button = self._first_visible(
                    ".btn",
                    frame=frame
                )
                retry_button.click()
                continue
            break

        return True

    def _maybe_autosave_state(self) -> None:
        if self.state_file and self.auto_save_state:
            self.save_state()

    # ---------- state persistence ----------

    def save_state(self, path: Optional[str | Path] = None) -> str:
        context = self._require_context()

        target = Path(path or self.state_file or "proton_state.json").resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            playwright_state = context.storage_state(indexed_db=True)
        except TypeError:
            playwright_state = context.storage_state()

        payload = {
            "version": self.STATE_VERSION,
            "playwright_storage_state": playwright_state,
        }

        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        target.chmod(0o600)

        return str(target)

    def clear_saved_state(self) -> bool:
        if not self.state_file:
            return False
        path = Path(self.state_file)
        if path.exists():
            path.unlink()
            return True
        return False

    def _read_state_file(self) -> Optional[dict]:
        if not self.state_file:
            return None

        path = Path(self.state_file)

        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtonCamoufoxError(
                f"Не удалось прочитать state_file {path}: {exc}"
            ) from exc

        if isinstance(payload, dict) and "cookies" in payload and "origins" in payload:
            return payload

        if not isinstance(payload, dict):
            return None

        playwright_state = payload.get("playwright_storage_state")

        return playwright_state if isinstance(playwright_state, dict) else None

    # ---------- mail actions ----------

    def inbox(self) -> None:
        self._ensure_mail()
        page = self._require_page()
        self.process_ui_tasks(min(self.ui_tasks_timeout_ms, 2_000))

        inbox_url = self._get_mailbox_url(page, "inbox")

        current_path = page.url.split("#", 1)[0].split("?", 1)[0]
        if current_path.rstrip("/") != inbox_url.rstrip("/"):
            page.goto(
                inbox_url,
                wait_until="domcontentloaded",
            )

        self._wait_for_message_list(page)

    def list_messages(self, limit: int = 20) -> list[MessageSummary]:
        self.inbox()
        return self._message_summaries(limit)

    def open_message(self, index: int = 0) -> Message:
        self.inbox()

        rows = self._all_rows()

        if index < 0 or index >= len(rows):
            raise MessageNotFound(
                f"Письмо с индексом {index} не найдено; "
                f"найдено строк: {len(rows)}"
            )

        self._click_ui(self._require_page(), rows[index])

        page = self._require_page()

        subject_locator = page.locator(
            '[data-testid="conversation-header:subject"]'
        ).first

        subject_locator.wait_for(state="visible", timeout=self.timeout_ms)

        message = page.locator('[data-testid^="message-view-"].is-opened').last
        message.wait_for(state="visible", timeout=self.timeout_ms)

        subject = subject_locator.inner_text().strip()

        sender = message.locator('[data-testid="recipients:sender"]').first

        sender_name = ""
        sender_email = ""

        if sender.count():
            sender_href = sender.get_attribute("href") or ""
            sender_email = sender_href.removeprefix("mailto:")

            sender_label = sender.locator('[data-testid="recipient-label"]').first

            if sender_label.count():
                sender_name = sender_label.inner_text().strip()

        date = message.locator('[data-testid="item-date-simple"]').first
        sent_at = date.get_attribute("datetime") if date.count() else None

        star_button = message.locator('[data-testid^="item-star-"]').first
        starred = (
            star_button.get_attribute("aria-pressed") == "true"
            if star_button.count()
            else None
        )

        attachments = self._extract_attachments(message)

        return Message(
            subject=subject,
            body=self._extract_body(message),
            url=page.url,
            sender_name=sender_name,
            sender_email=sender_email,
            recipients_to=self._extract_recipients(message),
            sent_at=sent_at,
            message_id=message.get_attribute("data-message-id"),
            starred=starred,
            has_attachments=bool(attachments),
            attachment_count=len(attachments),
            attachments=attachments,
        )

    def read_message(self, index: int = 0) -> Message:
        return self.open_message(index)

    def delete_message(self, index: int = 0) -> None:
        self.open_message(index)
        self.delete_current()

    def delete_current(self) -> None:
        page = self._require_page()

        delete_button = self._wait_for_any(
            page,
            '[data-testid="toolbar:movetotrash"]',
            timeout_ms=self.timeout_ms,
        )

        if delete_button is None:
            raise ProtonCamoufoxError(
                "Не найдена кнопка перемещения письма в корзину."
            )

        self._click_ui(page, delete_button)

    def search(self, query: str, limit: int = 20) -> list[MessageSummary]:
        self.inbox()
        page = self._require_page()

        search_input = self._wait_for_any(
            page,
            (
                '[data-testid="search-keyword"]',
                'input[placeholder*="Search" i]',
                'input[aria-label*="Search" i]',
            ),
            timeout_ms=self.timeout_ms,
        )

        if search_input is None:
            raise ProtonCamoufoxError("Не найдено поле поиска Proton.")

        self._enter_text(page, search_input, query)
        search_input.press("Enter")

        rows_locator = page.locator('[data-testid^="message-item:"]')

        try:
            rows_locator.first.wait_for(state="visible", timeout=self.timeout_ms)
        except PlaywrightTimeoutError:
            return []

        return self._message_summaries(limit)

    # ---------- debugging ----------

    def screenshot(self, path: str | Path = "proton_debug.png") -> str:
        page = self._require_page()
        path = str(Path(path).resolve())
        page.screenshot(path=path, full_page=True)
        return path

    def dump_dom_debug(self, name: str = "dom") -> dict[str, str]:
        """
        Save a screenshot, full HTML and a compact JSON list of visible inputs/buttons.
        Useful when Proton changes its selectors.
        """
        page = self._require_page()
        debug_dir = Path(self.debug_dir).resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "dom"
        screenshot_path = debug_dir / f"{safe_name}.png"
        html_path = debug_dir / f"{safe_name}.html"
        elements_path = debug_dir / f"{safe_name}_elements.json"

        page.screenshot(path=str(screenshot_path), full_page=True)
        html_path.write_text(page.content(), encoding="utf-8")

        elements = page.locator("input, button, [role=button]").evaluate_all(
            """els => els.map((el, index) => ({
                index,
                tag: el.tagName,
                id: el.id || null,
                name: el.getAttribute('name'),
                type: el.getAttribute('type'),
                placeholder: el.getAttribute('placeholder'),
                ariaLabel: el.getAttribute('aria-label'),
                title: el.getAttribute('title'),
                dataTestId: el.getAttribute('data-testid'),
                text: (el.innerText || el.textContent || '').trim().slice(0, 300),
                visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)
            }))"""
        )
        elements_path.write_text(
            json.dumps(elements, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "screenshot": str(screenshot_path),
            "html": str(html_path),
            "elements": str(elements_path),
        }

    def _handle_missing_login_fields(self, user_missing: bool, password_missing: bool) -> None:
        if not self.debug_on_missing_fields:
            return

        missing = []
        if user_missing:
            missing.append("username")
        if password_missing:
            missing.append("password")
        self._pause_for_dom_inspection("missing_" + "_".join(missing))

    def _pause_for_dom_inspection(self, reason: str) -> None:
        page = self._require_page()
        paths = self.dump_dom_debug(reason)

        print(f"\n[proton_camoufox] DEBUG: Proton DOM не совпал с текущими селекторами. {reason}")
        print(f"URL: {page.url}")
        print(f"Screenshot: {paths['screenshot']}")
        print(f"HTML:       {paths['html']}")
        print(f"Elements:   {paths['elements']}")
        print(
            "Браузер оставлен открытым. Посмотрите нужные input/button, "
            "их id/name/type/data-testid/aria-label."
        )
        try:
            input("Когда закончите проверку, нажмите Enter в консоли...")
        except (EOFError, KeyboardInterrupt):
            pass

    # ---------- internal helpers ----------

    def _is_login_url(self, url: str) -> bool:
        return bool(self.LOGIN_URL_RE.match(url))

    def _ensure_mail(self) -> None:
        page = self._require_page()

        if self._is_mail_url(page.url):
            return

        if self._is_authorize_url(page.url):
            if self._wait_for_mail(page):
                return

            raise LoginFailed(
                f"Авторизация Proton не завершилась. Текущий URL: {page.url}"
            )

        if self._is_login_url(page.url):
            self.login()
            return

        page.goto(self.MAIL_URL, wait_until="domcontentloaded")

        if self._wait_for_mail(page, self.timeout_ms):
            return

        if self._is_login_url(page.url):
            self.login()
            return

        if self._is_authorize_url(page.url):
            if self._wait_for_mail(page):
                return

        raise LoginFailed(
            f"Proton Mail не открылся. Текущий URL: {page.url}"
        )

    def _get_session_storage(self) -> dict[str, str]:
        page = self._require_page()

        if not page.url.startswith("https://mail.proton.me/"):
            return {}

        return page.evaluate(
            """
            () => Object.fromEntries(
                Object.entries(sessionStorage)
            )
            """
        )

    def _parse_message_summary(self, row, index: int) -> MessageSummary:
        subject = row.locator('[data-testid="message-row:subject"]').first.inner_text().strip()

        sender = row.locator('[data-testid="message-column:sender-address"]').first
        sender_email = sender.get_attribute("title") or ""

        try:
            sender_name = sender.evaluate(
                """element => {
                    const clone = element.cloneNode(true);
                    clone.querySelectorAll('[data-testid="proton-badge"]').forEach(
                        element => element.remove()
                    );
                    return clone.innerText.trim();
                }"""
            )
        except Exception:
            sender_name = ""

        date = row.locator('[data-testid="item-date-simple"]').first
        sent_at = date.get_attribute("datetime") if date.count() else None

        class_name = row.get_attribute("class") or ""
        unread = "unread" in class_name.split()

        starred = None
        star_button = row.locator('[data-testid^="item-star-"]').first

        if star_button.count():
            starred = star_button.get_attribute("aria-pressed") == "true"

        attachment_indicator = row.locator(
            '[data-testid="item-attachment-icon-paper-clip"]'
        ).first

        attachment_count = 0

        if attachment_indicator.count():
            try:
                attachment_count = self._parse_attachment_count(
                    attachment_indicator.inner_text()
                )
            except Exception:
                pass

        return MessageSummary(
            index=index,
            subject=subject,
            sender_name=sender_name,
            sender_email=sender_email,
            sent_at=sent_at,
            unread=unread,
            starred=starred,
            has_attachments=attachment_count > 0,
            attachment_count=attachment_count,
            item_id=row.get_attribute("data-element-id"),
        )

    def _extract_attachments(self, message) -> tuple[Attachment, ...]:
        attachments = []
        items = message.locator(
            'button[data-testid^="attachment-item:"][data-testid$="--primary-action"]'
        )

        for index in range(items.count()):
            item = items.nth(index)

            try:
                name_element = item.locator("[aria-label]").first
                name = name_element.get_attribute("aria-label") if name_element.count() else ""

                size_element = item.locator('[data-testid="attachment-item:size"]').first
                size = size_element.inner_text().strip() if size_element.count() else None

                if name:
                    attachments.append(Attachment(name=name, size=size))
            except Exception:
                continue

        return tuple(attachments)

    def _extract_recipients(self, message) -> tuple[str, ...]:
        recipients = message.locator('[data-testid="message-header:to"] a[href^="mailto:"]')

        result = []

        for index in range(recipients.count()):
            try:
                href = recipients.nth(index).get_attribute("href")

                if href:
                    email = href.removeprefix("mailto:")

                    if email and email not in result:
                        result.append(email)
            except Exception:
                continue

        return tuple(result)

    def _parse_attachment_count(self, text: str) -> int:
        match = re.search(r"Has\s+(\d+)\s+attachments?", text, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _get_mailbox_url(self, page: Page, folder: str = "inbox") -> str:
        match = re.match(r"^(https://mail\.proton\.me/u/\d+)", page.url)

        if match is None:
            raise ProtonCamoufoxError(
                f"Не удалось определить Proton account index из URL: {page.url}"
            )

        return f"{match.group(1)}/{folder}"

    def _wait_for_mail(self, page: Page, timeout_ms: Optional[int] = None) -> bool:
        timeout_ms = timeout_ms or self.timeout_ms
        deadline = time.monotonic() + timeout_ms / 1000

        while time.monotonic() < deadline:
            context = self._context
            candidates = list(context.pages) if context is not None else [page]

            for candidate in reversed(candidates):
                if candidate.is_closed() or not self._is_mail_url(candidate.url):
                    continue

                self._page = candidate
                candidate.set_default_timeout(self.timeout_ms)

                try:
                    candidate.wait_for_load_state(
                        "domcontentloaded",
                        timeout=min(5_000, timeout_ms),
                    )
                except PlaywrightTimeoutError:
                    pass

                if context is not None:
                    self._close_pages_except(context, candidate)

                remaining_ms = max(
                    0,
                    int((deadline - time.monotonic()) * 1000),
                )
                self.process_ui_tasks(
                    min(self.ui_tasks_timeout_ms, remaining_ms)
                )
                return True

            active_page = next(
                (candidate for candidate in candidates if not candidate.is_closed()),
                None,
            )
            if active_page is None:
                return False

            active_page.wait_for_timeout(100)

        return False

    def _is_authorize_url(self, url: str) -> bool:
        return bool(self.AUTHORIZE_URL_RE.match(url))

    def _is_mail_url(self, url: str) -> bool:
        return bool(self.MAIL_URL_RE.match(url))

    def _effective_headless(self) -> bool:
        return False if self.debug_on_missing_fields else self.headless

    def _require_page(self) -> Page:
        if self._page is None:
            raise ProtonCamoufoxError(
                "Клиент не запущен. Используйте with ProtonMailClient(...) или .start()."
            )
        return self._page

    def _require_context(self) -> BrowserContext:
        if self._context is None:
            raise ProtonCamoufoxError("BrowserContext ещё не создан. Сначала вызовите .start().")
        return self._context

    def _first_visible(
        self,
        selectors: str | Iterable[str],
        frame=None,
        iframe_name: Optional[str] = None,
    ):
        page = self._require_page()
        return self._find_in_frames(
            page, selectors, frame=frame, iframe_name=iframe_name,
            state="visible", max_matches=5,
        )

    @staticmethod
    def _candidate_frames(page: Page, frame=None, iframe_name: Optional[str] = None):
        if frame is not None:
            return (frame,)
        if iframe_name is not None:
            named_frame = page.frame(name=iframe_name)
            return (named_frame,) if named_frame is not None else ()
        return page.frames

    def _find_in_frames(
        self,
        page: Page,
        selectors: str | Iterable[str],
        *,
        frame=None,
        iframe_name: Optional[str] = None,
        state: str = "visible",
        max_matches: int = 10,
    ):
        if isinstance(selectors, str):
            selectors = (selectors,)

        frames = self._candidate_frames(page, frame=frame, iframe_name=iframe_name)
        for current_frame in frames:
            if current_frame is None or current_frame.is_detached():
                continue
            for selector in selectors:
                try:
                    matches = current_frame.locator(selector)
                    for index in range(min(matches.count(), max_matches)):
                        item = matches.nth(index)
                        if state == "attached":
                            return item
                        if state == "visible" and item.is_visible():
                            return item
                        if state == "hidden" and not item.is_visible():
                            return item
                except Exception:
                    continue
        return None

    def _wait_for_login_fields(self, timeout_ms: int):
        deadline = time.monotonic() + timeout_ms / 1000

        username_input = None
        password_input = None

        while time.monotonic() < deadline:
            if username_input is None:
                username_input = self._first_visible((
                    'input#username',
                    'input[name="username"]',
                    'input[type="email"]',
                ))

            if password_input is None:
                password_input = self._first_visible((
                    'input#password',
                    'input[name="password"]',
                    'input[type="password"]',
                ))

            if username_input is not None and password_input is not None:
                return username_input, password_input

            if self._looks_like_verification():
                break

            time.sleep(0.1)

        return username_input, password_input

    def wait_for_frame(
        self,
        page,
        name,
        timeout_ms=15_000,
        required=True,
    ):
        deadline = time.monotonic() + timeout_ms / 1000

        while time.monotonic() < deadline:
            frame = page.frame(name=name)
            if frame is not None and not frame.is_detached():
                return frame

            if not required and self._is_mail_url(page.url):
                return None

            page.wait_for_timeout(50)

        if required:
            raise TimeoutError(
                f'Frame "{name}" не найден за '
                f'{timeout_ms / 1000:.1f} сек.'
            )

        return None

    def _all_rows(self):
        page = self._require_page()
        rows = page.locator('[data-testid^="message-item:"]')

        result = []

        for index in range(rows.count()):
            row = rows.nth(index)

            try:
                if row.is_visible():
                    result.append(row)
            except Exception:
                continue

        return result

    def _wait_for_message_list(self, page: Page) -> None:
        deadline = time.monotonic() + self.timeout_ms / 1000
        while time.monotonic() < deadline:
            rows = page.locator('[data-testid^="message-item:"]')
            for index in range(rows.count()):
                try:
                    if rows.nth(index).is_visible():
                        return
                except Exception:
                    continue

            loaded = page.locator('[data-testid="message-list-loaded"]').first
            if loaded.count() and loaded.is_visible() and rows.count() == 0:
                return
            page.wait_for_timeout(100)

        raise ProtonCamoufoxError(
            "Список писем не загрузился или строки писем скрыты. "
            f"Текущий URL: {page.url}"
        )

    def _message_summaries(self, limit: int) -> list[MessageSummary]:
        messages: list[MessageSummary] = []
        for index, row in enumerate(self._all_rows()[:limit]):
            try:
                messages.append(self._parse_message_summary(row, index))
            except Exception as exc:
                raise ProtonCamoufoxError(
                    f"Не удалось прочитать строку письма {index + 1}: {exc}"
                ) from exc
        return messages

    def _text_from_first(self, selectors: Iterable[str]) -> str:
        item = self._first_visible(selectors)
        if item is None:
            return ""
        try:
            return item.inner_text().strip()
        except Exception:
            return ""

    def _extract_body(self, message) -> str:
        try:
            content_iframe = message.locator('iframe[data-testid="content-iframe"]').first
            content_iframe.wait_for(state="attached", timeout=self.timeout_ms)

            iframe_handle = content_iframe.element_handle()

            if iframe_handle is None:
                return ""

            frame = iframe_handle.content_frame()

            if frame is None:
                return ""

            body = frame.locator("body")
            body.wait_for(state="attached", timeout=self.timeout_ms)

            return body.inner_text().strip()

        except Exception:
            return ""

    def _looks_like_verification(self) -> bool:
        page = self._require_page()
        try:
            text = page.locator("body").inner_text().lower()
        except Exception:
            text = ""
        words = (
            "two-factor", "2fa", "verification code", "captcha", "human verification",
            "код подтверждения", "двухфактор", "проверка", "подтвердите",
        )
        return any(w in text for w in words)

    def _check_verification(self):
        if self._looks_like_verification():
            raise VerificationRequired("Proton запросил CAPTCHA/2FA/проверку пользователя.")

    def _wait_for_any(
        self,
        page: Page,
        selectors: str | Iterable[str],
        timeout_ms: int = 10_000,
        state: str = "visible",
        frame=None,
        iframe_name: Optional[str] = None,
    ):
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            match = self._find_in_frames(
                page, selectors, frame=frame, iframe_name=iframe_name,
                state=state, max_matches=10,
            )
            if match is not None:
                return match
            page.wait_for_timeout(50)
        return None

    # ---------- captcha solver ----------

    def drag_in_canvas(self, page, canvas, start, end, steps=30):
        box = canvas.bounding_box()

        if box is None:
            raise RuntimeError("Canvas не виден")

        start_x = box["x"] + start[0]
        start_y = box["y"] + start[1]

        end_x = box["x"] + end[0]
        end_y = box["y"] + end[1]

        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(end_x, end_y, steps=steps)
        page.mouse.up()

    def canvas_from_iframe_to_cv2(
        self,
        page,
        iframe_name,
        canvas_selector,
        frame=None,
    ):
        if frame is None:
            frame = self.wait_for_frame(page, iframe_name, 30_000)
        if frame is None:
            raise RuntimeError("Frame cnvs не найден")

        canvas = frame.locator(canvas_selector).first
        canvas.wait_for(state="visible", timeout=15000)
        page.wait_for_timeout(1000)

        image_bytes = canvas.screenshot()

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError("Не удалось декодировать canvas")

        return image, canvas

    def find_header_bottom(self, img):
        h, w = img.shape[:2]
        x1 = int(w * 0.35)

        reference = img[:min(15, h), x1:]
        bg_color = np.median(reference.reshape(-1, 3), axis=0).astype(np.float32)

        for y in range(5, min(137, h - 2)):
            ok = True

            for yy in range(y, y + 3):
                row = img[yy, x1:].astype(np.float32)
                diff = np.linalg.norm(row - bg_color, axis=1)

                if np.mean(diff > 12) <= 0.15:
                    ok = False
                    break

            if ok:
                return y

        Image.fromarray(img).save('dont_solved.png')
        raise RuntimeError("Не удалось определить границу верхней панели")

    def find_top_puzzle(self, img, split_y):
        _, w = img.shape[:2]

        roi_width = min(max(100, int(w * 0.35)), w)
        roi = img[:split_y, :roi_width]

        reference_x = int(w * 0.5)
        reference = img[:min(15, split_y), reference_x:]
        bg_color = np.median(reference.reshape(-1, 3), axis=0).astype(np.float32)

        diff = np.linalg.norm(roi.astype(np.float32) - bg_color, axis=2)
        mask = (diff > 12).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        best = None

        for i in range(1, count):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < 100 or width < 15 or height < 15:
                continue

            if width > 100 or height > 100:
                continue

            score = area - x * 2 - y * 2

            if best is None or score > best[0]:
                best = (score, x, y, width, height)

        if best is None:
            raise RuntimeError("Пазл не найден")

        _, x, y, width, height = best

        size = width if y + height >= split_y - 2 else max(width, height)

        return x + size / 2, y + size / 2, size

    def cluster_candidates(self, candidates, distance=4):
        clusters = []

        for candidate in candidates:
            for cluster in clusters:
                if abs(candidate["x"] - cluster["x"]) <= distance and abs(candidate["y"] - cluster["y"]) <= distance:
                    cluster["items"].append(candidate)
                    cluster["x"] = np.mean([item["x"] for item in cluster["items"]])
                    cluster["y"] = np.mean([item["y"] for item in cluster["items"]])
                    break
            else:
                clusters.append({
                    "x": candidate["x"],
                    "y": candidate["y"],
                    "items": [candidate],
                })

        return clusters

    def find_slot(self, img, split_y, puzzle_size):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        candidates = []
        expected_size = puzzle_size * 0.85

        for threshold in range(155, 231, 5):
            mask = (gray >= threshold).astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if hierarchy is None:
                continue

            hierarchy = hierarchy[0]

            for i, contour in enumerate(contours):
                if hierarchy[i][3] < 0:
                    continue

                x, y, width, height = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                if y < split_y + puzzle_size * 0.25:
                    continue

                if not puzzle_size * 0.65 <= width <= puzzle_size * 1.30:
                    continue

                if not puzzle_size * 0.65 <= height <= puzzle_size * 1.30:
                    continue

                fill_ratio = area / (width * height)

                if fill_ratio < 0.35:
                    continue

                aspect_ratio = width / height

                if not 0.65 <= aspect_ratio <= 1.50:
                    continue

                size_score = (abs(width - expected_size) + abs(height - expected_size)) / puzzle_size
                ratio_score = abs(np.log(aspect_ratio))
                fill_score = abs(fill_ratio - 0.68)

                score = size_score + ratio_score * 0.4 + fill_score * 0.3

                candidates.append({
                    "x": x + width / 2,
                    "y": y + height / 2,
                    "threshold": threshold,
                    "score": score,
                })

        if not candidates:
            raise RuntimeError("Место для пазла не найдено")

        clusters = self.cluster_candidates(candidates)

        best_cluster = min(
            clusters,
            key=lambda cluster: (
                -len({item["threshold"] for item in cluster["items"]}),
                np.mean([item["score"] for item in cluster["items"]]),
            ),
        )

        best_item = min(best_cluster["items"], key=lambda item: item["score"])

        return best_item["x"], best_item["y"]

    def find_puzzle_and_slot(self, img):
        split_y = self.find_header_bottom(img)

        puzzle_x, puzzle_y, puzzle_size = self.find_top_puzzle(img, split_y)
        slot_x, slot_y = self.find_slot(img, split_y, puzzle_size)

        return {
            "puzzle": (round(puzzle_x, 2), round(puzzle_y, 2)),
            "slot": (round(slot_x, 2), round(slot_y, 2)),
        }
