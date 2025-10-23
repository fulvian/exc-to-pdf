"""
Locale management for internationalization support.

This module provides locale detection, management, and configuration
for multi-language support in exc-to-pdf.
"""

import locale
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

import babel
from babel.core import Locale, UnknownLocaleError
from babel.support import Format
from babel.dates import format_date, format_datetime, format_time
from babel.numbers import format_currency, format_decimal, format_percent
import pytz

logger = logging.getLogger(__name__)


class LocaleManager:
    """Manages locale detection and configuration."""

    # Supported locales with their configurations
    SUPPORTED_LOCALES = {
        "en_US": {
            "name": "English (United States)",
            "date_format": "medium",
            "time_format": "medium",
            "currency": "USD",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ".",
            "thousands_separator": ",",
        },
        "it_IT": {
            "name": "Italiano (Italia)",
            "date_format": "long",
            "time_format": "short",
            "currency": "EUR",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ",",
            "thousands_separator": ".",
        },
        "de_DE": {
            "name": "Deutsch (Deutschland)",
            "date_format": "medium",
            "time_format": "short",
            "currency": "EUR",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ",",
            "thousands_separator": ".",
        },
        "fr_FR": {
            "name": "Français (France)",
            "date_format": "long",
            "time_format": "short",
            "currency": "EUR",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ",",
            "thousands_separator": " ",
        },
        "es_ES": {
            "name": "Español (España)",
            "date_format": "long",
            "time_format": "short",
            "currency": "EUR",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ",",
            "thousands_separator": ".",
        },
        "ar_EG": {
            "name": "العربية (مصر)",
            "date_format": "long",
            "time_format": "medium",
            "currency": "EGP",
            "number_format": "decimal",
            "rtl": True,
            "decimal_separator": "٫",
            "thousands_separator": "٬",
        },
        "ja_JP": {
            "name": "日本語 (日本)",
            "date_format": "long",
            "time_format": "short",
            "currency": "JPY",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ".",
            "thousands_separator": ",",
        },
        "zh_CN": {
            "name": "中文 (中国)",
            "date_format": "long",
            "time_format": "short",
            "currency": "CNY",
            "number_format": "decimal",
            "rtl": False,
            "decimal_separator": ".",
            "thousands_separator": ",",
        },
    }

    def __init__(self, default_locale: str = "en_US", timezone: str = "UTC"):
        """Initialize locale manager.

        Args:
            default_locale: Default locale to use
            timezone: Default timezone for date/time formatting
        """
        self.default_locale = default_locale
        self.timezone = timezone
        self._current_locale = None
        self._formatters = {}

        # Set default locale
        self.set_locale(default_locale)

    def detect_locale(
        self,
        input_file: Optional[str] = None,
        system_locale: bool = True,
        file_analysis: bool = True,
    ) -> str:
        """Detect appropriate locale based on various factors.

        Args:
            input_file: Path to input file for analysis
            system_locale: Whether to consider system locale
            file_analysis: Whether to analyze input file for locale hints

        Returns:
            Detected locale code
        """
        detected_locale = None

        # Priority 1: System locale (if enabled)
        if system_locale:
            try:
                system_locale_code = self._get_system_locale()
                if system_locale_code in self.SUPPORTED_LOCALES:
                    detected_locale = system_locale_code
                    logger.debug(f"Detected system locale: {detected_locale}")
            except Exception as e:
                logger.warning(f"Failed to detect system locale: {e}")

        # Priority 2: File analysis (if enabled and file provided)
        if file_analysis and input_file and not detected_locale:
            try:
                file_locale = self._analyze_file_locale(input_file)
                if file_locale in self.SUPPORTED_LOCALES:
                    detected_locale = file_locale
                    logger.debug(f"Detected file locale: {detected_locale}")
            except Exception as e:
                logger.warning(f"Failed to analyze file locale: {e}")

        # Fallback to default locale
        if not detected_locale:
            detected_locale = self.default_locale
            logger.debug(f"Using default locale: {detected_locale}")

        logger.info(
            f"Detected locale: {detected_locale} ({self.SUPPORTED_LOCALES[detected_locale]['name']})"
        )
        return detected_locale

    def set_locale(self, locale_code: str) -> None:
        """Set the current locale.

        Args:
            locale_code: Locale code to set

        Raises:
            ValueError: If locale is not supported
        """
        if locale_code not in self.SUPPORTED_LOCALES:
            available = ", ".join(self.SUPPORTED_LOCALES.keys())
            raise ValueError(
                f"Unsupported locale: {locale_code}. Supported locales: {available}"
            )

        try:
            # Create Babel Locale object
            self._current_locale = Locale.parse(locale_code)

            # Create formatter for this locale
            try:
                tz = pytz.timezone(self.timezone)
                self._formatters[locale_code] = Format(self._current_locale, tz)
            except pytz.UnknownTimeZoneError:
                logger.warning(f"Unknown timezone: {self.timezone}, using UTC")
                self._formatters[locale_code] = Format(self._current_locale, pytz.UTC)

            # Set Python locale for formatting
            try:
                locale.setlocale(locale.LC_ALL, locale_code)
            except locale.Error:
                logger.warning(f"Could not set Python locale to {locale_code}")

            logger.info(f"Locale set to: {locale_code}")

        except UnknownLocaleError as e:
            raise ValueError(f"Invalid locale: {locale_code}") from e

    def get_current_locale(self) -> str:
        """Get the current locale code.

        Returns:
            Current locale code
        """
        return (
            str(self._current_locale) if self._current_locale else self.default_locale
        )

    def get_locale_info(self, locale_code: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a locale.

        Args:
            locale_code: Locale code to get info for (uses current if None)

        Returns:
            Dictionary with locale information
        """
        locale_code = locale_code or self.get_current_locale()
        return self.SUPPORTED_LOCALES.get(locale_code, {})

    def get_formatter(self, locale_code: Optional[str] = None) -> Format:
        """Get Babel formatter for a locale.

        Args:
            locale_code: Locale code (uses current if None)

        Returns:
            Babel Format object
        """
        locale_code = locale_code or self.get_current_locale()

        if locale_code not in self._formatters:
            try:
                locale_obj = Locale.parse(locale_code)
                tz = pytz.timezone(self.timezone)
                self._formatters[locale_code] = Format(locale_obj, tz)
            except (UnknownLocaleError, pytz.UnknownTimeZoneError) as e:
                logger.warning(f"Error creating formatter for {locale_code}: {e}")
                # Fallback to default locale formatter
                return self._formatters[self.default_locale]

        return self._formatters[locale_code]

    def is_rtl(self, locale_code: Optional[str] = None) -> bool:
        """Check if locale is right-to-left.

        Args:
            locale_code: Locale code to check (uses current if None)

        Returns:
            True if locale is RTL
        """
        locale_code = locale_code or self.get_current_locale()
        return self.SUPPORTED_LOCALES.get(locale_code, {}).get("rtl", False)

    def get_currency_code(self, locale_code: Optional[str] = None) -> str:
        """Get default currency code for locale.

        Args:
            locale_code: Locale code (uses current if None)

        Returns:
            Currency code
        """
        locale_code = locale_code or self.get_current_locale()
        return self.SUPPORTED_LOCALES.get(locale_code, {}).get("currency", "USD")

    def get_supported_locales(self) -> Dict[str, Dict[str, Any]]:
        """Get all supported locales.

        Returns:
            Dictionary of supported locales with their configurations
        """
        return self.SUPPORTED_LOCALES.copy()

    def list_supported_locales(self) -> List[str]:
        """List all supported locale codes.

        Returns:
            List of supported locale codes
        """
        return list(self.SUPPORTED_LOCALES.keys())

    def _get_system_locale(self) -> Optional[str]:
        """Get system locale.

        Returns:
            System locale code or None
        """
        try:
            # Get system locale
            system_locale = locale.getdefaultlocale()
            if system_locale and system_locale[0]:
                # Convert to proper format (e.g., 'en_US' instead of 'en_US.UTF-8')
                locale_code = system_locale[0]
                if "." in locale_code:
                    locale_code = locale_code.split(".")[0]

                # Check if we support this locale
                if locale_code in self.SUPPORTED_LOCALES:
                    return locale_code

                # Try language-only version (e.g., 'en' -> 'en_US')
                language = locale_code.split("_")[0]
                for supported_locale in self.SUPPORTED_LOCALES:
                    if supported_locale.startswith(language + "_"):
                        return supported_locale

        except Exception as e:
            logger.debug(f"Could not detect system locale: {e}")

        return None

    def _analyze_file_locale(self, file_path: str) -> Optional[str]:
        """Analyze file to detect locale hints.

        Args:
            file_path: Path to file to analyze

        Returns:
            Detected locale code or None
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return None

            # For Excel files, could analyze number formats, dates, etc.
            # This is a simplified implementation
            if file_path.suffix.lower() in [".xlsx", ".xls"]:
                # Could use openpyxl to analyze cell formats
                # For now, just check filename and path for hints
                return self._analyze_path_for_locale_hints(file_path)

        except Exception as e:
            logger.debug(f"Error analyzing file locale: {e}")

        return None

    def _analyze_path_for_locale_hints(self, file_path: Path) -> Optional[str]:
        """Analyze file path for locale hints.

        Args:
            file_path: File path to analyze

        Returns:
            Detected locale code or None
        """
        path_str = str(file_path).lower()

        # Check for language indicators in path
        locale_hints = {
            "italian": "it_IT",
            "italiano": "it_IT",
            "italia": "it_IT",
            "german": "de_DE",
            "deutsch": "de_DE",
            "deutschland": "de_DE",
            "french": "fr_FR",
            "français": "fr_FR",
            "france": "fr_FR",
            "spanish": "es_ES",
            "español": "es_ES",
            "españa": "es_ES",
            "arabic": "ar_EG",
            "العربية": "ar_EG",
            "japanese": "ja_JP",
            "日本語": "ja_JP",
            "chinese": "zh_CN",
            "中文": "zh_CN",
        }

        for hint, locale_code in locale_hints.items():
            if hint in path_str:
                return locale_code

        return None

    def get_translation_dict(self, locale_code: Optional[str] = None) -> Dict[str, str]:
        """Get translation dictionary for UI strings.

        Args:
            locale_code: Locale code (uses current if None)

        Returns:
            Dictionary with translated strings
        """
        locale_code = locale_code or self.get_current_locale()

        # Basic translations - in a real implementation, this would
        # load from proper .po/.mo files
        translations = {
            "en_US": {
                "loading": "Loading...",
                "processing": "Processing...",
                "complete": "Complete",
                "error": "Error",
                "page": "Page",
                "generated_by": "Generated by exc-to-pdf",
                "table": "Table",
                "chart": "Chart",
                "sheet": "Sheet",
            },
            "it_IT": {
                "loading": "Caricamento...",
                "processing": "Elaborazione...",
                "complete": "Completato",
                "error": "Errore",
                "page": "Pagina",
                "generated_by": "Generato da exc-to-pdf",
                "table": "Tabella",
                "chart": "Grafico",
                "sheet": "Foglio",
            },
            "de_DE": {
                "loading": "Laden...",
                "processing": "Verarbeitung...",
                "complete": "Abgeschlossen",
                "error": "Fehler",
                "page": "Seite",
                "generated_by": "Erstellt von exc-to-pdf",
                "table": "Tabelle",
                "chart": "Diagramm",
                "sheet": "Blatt",
            },
            "fr_FR": {
                "loading": "Chargement...",
                "processing": "Traitement...",
                "complete": "Terminé",
                "error": "Erreur",
                "page": "Page",
                "generated_by": "Généré par exc-to-pdf",
                "table": "Tableau",
                "chart": "Graphique",
                "sheet": "Feuille",
            },
            "es_ES": {
                "loading": "Cargando...",
                "processing": "Procesando...",
                "complete": "Completado",
                "error": "Error",
                "page": "Página",
                "generated_by": "Generado por exc-to-pdf",
                "table": "Tabla",
                "chart": "Gráfico",
                "sheet": "Hoja",
            },
        }

        return translations.get(locale_code, translations["en_US"])
