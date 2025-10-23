"""
Locale-aware formatters for dates, numbers, and currencies.

This module provides formatting functions that respect locale conventions
for internationalized output.
"""

from datetime import datetime, date, time
from decimal import Decimal
from typing import Optional, Union
import logging

from babel.dates import format_date, format_datetime, format_time, get_day_names, get_month_names
from babel.numbers import format_currency, format_decimal, format_percent, format_scientific
from babel.support import Format

from .locale_manager import LocaleManager

logger = logging.getLogger(__name__)


class LocaleFormatters:
    """Provides locale-aware formatting functions."""

    def __init__(self, locale_manager: LocaleManager):
        """Initialize formatters with locale manager.

        Args:
            locale_manager: LocaleManager instance for locale information
        """
        self.locale_manager = locale_manager

    def format_date(self, date_obj: Union[date, datetime],
                   format_style: str = 'medium',
                   locale_code: Optional[str] = None) -> str:
        """Format a date according to locale conventions.

        Args:
            date_obj: Date or datetime object to format
            format_style: Format style ('short', 'medium', 'long', 'full')
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted date string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            formatter = self.locale_manager.get_formatter(locale_code)

            if isinstance(date_obj, datetime):
                return formatter.date(date_obj, format_style)
            else:
                return format_date(date_obj, format_style, locale=locale_code)

        except Exception as e:
            logger.error(f"Error formatting date: {e}")
            return str(date_obj)

    def format_time(self, time_obj: Union[time, datetime],
                   format_style: str = 'medium',
                   locale_code: Optional[str] = None) -> str:
        """Format a time according to locale conventions.

        Args:
            time_obj: Time or datetime object to format
            format_style: Format style ('short', 'medium', 'long', 'full')
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted time string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            formatter = self.locale_manager.get_formatter(locale_code)

            if isinstance(time_obj, datetime):
                return formatter.time(time_obj, format_style)
            else:
                return format_time(time_obj, format_style, locale=locale_code)

        except Exception as e:
            logger.error(f"Error formatting time: {e}")
            return str(time_obj)

    def format_datetime(self, datetime_obj: datetime,
                       date_format: str = 'medium',
                       time_format: str = 'medium',
                       locale_code: Optional[str] = None) -> str:
        """Format a datetime according to locale conventions.

        Args:
            datetime_obj: Datetime object to format
            date_format: Date format style
            time_format: Time format style
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted datetime string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            return format_datetime(datetime_obj, f"{date_format} {time_format}", locale=locale_code)

        except Exception as e:
            logger.error(f"Error formatting datetime: {e}")
            return str(datetime_obj)

    def format_number(self, number: Union[int, float, Decimal],
                     locale_code: Optional[str] = None,
                     decimal_places: Optional[int] = None) -> str:
        """Format a number according to locale conventions.

        Args:
            number: Number to format
            locale_code: Locale code (uses current if None)
            decimal_places: Number of decimal places (optional)

        Returns:
            Formatted number string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            formatter = self.locale_manager.get_formatter(locale_code)

            if decimal_places is not None:
                # Round to specified decimal places
                number = round(float(number), decimal_places)
                # Format with custom precision
                format_str = f"0.{'' * decimal_places}" if decimal_places > 0 else "0"
                return format_decimal(number, format_str, locale=locale_code)
            else:
                return formatter.decimal(number)

        except Exception as e:
            logger.error(f"Error formatting number: {e}")
            return str(number)

    def format_currency(self, amount: Union[int, float, Decimal],
                       currency_code: Optional[str] = None,
                       locale_code: Optional[str] = None,
                       format_type: str = 'standard') -> str:
        """Format a currency amount according to locale conventions.

        Args:
            amount: Amount to format
            currency_code: Currency code (uses locale default if None)
            locale_code: Locale code (uses current if None)
            format_type: Format type ('standard', 'name', 'accounting')

        Returns:
            Formatted currency string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            currency_code = currency_code or self.locale_manager.get_currency_code(locale_code)

            return format_currency(
                amount,
                currency_code,
                locale=locale_code,
                format_type=format_type
            )

        except Exception as e:
            logger.error(f"Error formatting currency: {e}")
            return f"{currency_code or ''} {amount}"

    def format_percent(self, number: Union[int, float, Decimal],
                      locale_code: Optional[str] = None,
                      decimal_places: Optional[int] = None) -> str:
        """Format a percentage according to locale conventions.

        Args:
            number: Number to format as percentage (0-1 or 0-100)
            locale_code: Locale code (uses current if None)
            decimal_places: Number of decimal places (optional)

        Returns:
            Formatted percentage string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            formatter = self.locale_manager.get_formatter(locale_code)

            if decimal_places is not None:
                # Use format_decimal for custom precision
                percentage = float(number) * 100
                format_str = f"0.{'' * decimal_places}%" if decimal_places > 0 else "0%"
                return format_decimal(percentage, format_str, locale=locale_code)
            else:
                return formatter.percent(number)

        except Exception as e:
            logger.error(f"Error formatting percentage: {e}")
            return f"{number}%"

    def format_scientific(self, number: Union[int, float, Decimal],
                         locale_code: Optional[str] = None) -> str:
        """Format a number in scientific notation according to locale conventions.

        Args:
            number: Number to format
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted scientific notation string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            return format_scientific(number, locale=locale_code)

        except Exception as e:
            logger.error(f"Error formatting scientific notation: {e}")
            return f"{number}e+0"

    def format_file_size(self, size_bytes: int,
                        locale_code: Optional[str] = None) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted file size string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()

            # Determine appropriate unit
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            size = float(size_bytes)
            unit_index = 0

            while size >= 1024.0 and unit_index < len(units) - 1:
                size /= 1024.0
                unit_index += 1

            # Format based on unit
            if unit_index == 0:  # Bytes
                return f"{int(size)} {units[unit_index]}"
            else:
                formatted_size = self.format_number(size, locale_code, decimal_places=2)
                return f"{formatted_size} {units[unit_index]}"

        except Exception as e:
            logger.error(f"Error formatting file size: {e}")
            return f"{size_bytes} B"

    def format_duration(self, seconds: float,
                       locale_code: Optional[str] = None) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted duration string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            translations = self.locale_manager.get_translation_dict(locale_code)

            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)

            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if secs > 0 or not parts:
                parts.append(f"{secs}s")

            return " ".join(parts)

        except Exception as e:
            logger.error(f"Error formatting duration: {e}")
            return f"{seconds:.2f}s"

    def get_month_name(self, month: int, locale_code: Optional[str] = None) -> str:
        """Get month name in locale.

        Args:
            month: Month number (1-12)
            locale_code: Locale code (uses current if None)

        Returns:
            Localized month name
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            month_names = get_month_names('wide', locale=locale_code)
            return month_names.get(month, '')

        except Exception as e:
            logger.error(f"Error getting month name: {e}")
            return str(month)

    def get_day_name(self, day: int, locale_code: Optional[str] = None) -> str:
        """Get day name in locale.

        Args:
            day: Day number (0-6, where 0 is Monday)
            locale_code: Locale code (uses current if None)

        Returns:
            Localized day name
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()
            day_names = get_day_names('wide', locale=locale_code)
            return day_names.get(day, '')

        except Exception as e:
            logger.error(f"Error getting day name: {e}")
            return str(day)

    def format_list(self, items: list, locale_code: Optional[str] = None) -> str:
        """Format a list of items according to locale conventions.

        Args:
            items: List of items to format
            locale_code: Locale code (uses current if None)

        Returns:
            Formatted list string
        """
        try:
            locale_code = locale_code or self.locale_manager.get_current_locale()

            if not items:
                return ""
            elif len(items) == 1:
                return str(items[0])
            elif len(items) == 2:
                # Use locale-specific conjunction
                translations = self.locale_manager.get_translation_dict(locale_code)
                if locale_code.startswith('it'):
                    return f"{items[0]} e {items[1]}"
                elif locale_code.startswith('de'):
                    return f"{items[0]} und {items[1]}"
                elif locale_code.startswith('fr'):
                    return f"{items[0]} et {items[1]}"
                elif locale_code.startswith('es'):
                    return f"{items[0]} y {items[1]}"
                else:  # English default
                    return f"{items[0]} and {items[1]}"
            else:
                # Use Oxford comma for English
                if locale_code.startswith('en'):
                    return f"{', '.join(str(item) for item in items[:-1])}, and {items[-1]}"
                else:
                    return f"{', '.join(str(item) for item in items[:-1])} {items[-1]}"

        except Exception as e:
            logger.error(f"Error formatting list: {e}")
            return ", ".join(str(item) for item in items)

    def get_text_direction(self, locale_code: Optional[str] = None) -> str:
        """Get text direction for locale.

        Args:
            locale_code: Locale code (uses current if None)

        Returns:
            'ltr' or 'rtl'
        """
        locale_code = locale_code or self.locale_manager.get_current_locale()
        return 'rtl' if self.locale_manager.is_rtl(locale_code) else 'ltr'

    def get_decimal_separator(self, locale_code: Optional[str] = None) -> str:
        """Get decimal separator for locale.

        Args:
            locale_code: Locale code (uses current if None)

        Returns:
            Decimal separator character
        """
        locale_code = locale_code or self.locale_manager.get_current_locale()
        locale_info = self.locale_manager.get_locale_info(locale_code)
        return locale_info.get('decimal_separator', '.')

    def get_thousands_separator(self, locale_code: Optional[str] = None) -> str:
        """Get thousands separator for locale.

        Args:
            locale_code: Locale code (uses current if None)

        Returns:
            Thousands separator character
        """
        locale_code = locale_code or self.locale_manager.get_current_locale()
        locale_info = self.locale_manager.get_locale_info(locale_code)
        return locale_info.get('thousands_separator', ',')