"""
Unit tests for locale manager module.
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import locale

from exc_to_pdf.i18n.locale_manager import LocaleManager
from babel.core import UnknownLocaleError


class TestLocaleManager:
    """Test cases for LocaleManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.locale_manager = LocaleManager()

    def test_init_default(self):
        """Test LocaleManager initialization with defaults."""
        assert self.locale_manager.default_locale == 'en_US'
        assert self.locale_manager.timezone == 'UTC'
        assert self.locale_manager.get_current_locale() == 'en_US'

    def test_init_custom(self):
        """Test LocaleManager initialization with custom values."""
        manager = LocaleManager(default_locale='it_IT', timezone='Europe/Rome')
        assert manager.default_locale == 'it_IT'
        assert manager.timezone == 'Europe/Rome'
        assert manager.get_current_locale() == 'it_IT'

    def test_get_supported_locales(self):
        """Test getting supported locales."""
        supported = self.locale_manager.get_supported_locales()
        assert isinstance(supported, dict)
        assert 'en_US' in supported
        assert 'it_IT' in supported
        assert 'de_DE' in supported

        # Check locale structure
        en_info = supported['en_US']
        assert 'name' in en_info
        assert 'date_format' in en_info
        assert 'currency' in en_info
        assert 'rtl' in en_info

    def test_list_supported_locales(self):
        """Test listing supported locale codes."""
        locale_codes = self.locale_manager.list_supported_locales()
        assert isinstance(locale_codes, list)
        assert 'en_US' in locale_codes
        assert 'it_IT' in locale_codes

    def test_set_locale_valid(self):
        """Test setting a valid locale."""
        self.locale_manager.set_locale('it_IT')
        assert self.locale_manager.get_current_locale() == 'it_IT'

    def test_set_locale_invalid(self):
        """Test setting an invalid locale."""
        with pytest.raises(ValueError, match="Unsupported locale"):
            self.locale_manager.set_locale('invalid_locale')

    def test_set_locale_unknown_babel_locale(self):
        """Test setting a locale unknown to Babel."""
        with patch('babel.core.Locale.parse', side_effect=UnknownLocaleError('Unknown')):
            with pytest.raises(ValueError, match="Invalid locale"):
                self.locale_manager.set_locale('en_US')

    def test_get_locale_info(self):
        """Test getting locale information."""
        info = self.locale_manager.get_locale_info('it_IT')
        assert info['name'] == 'Italiano (Italia)'
        assert info['currency'] == 'EUR'
        assert info['rtl'] == False

        # Test with current locale
        current_info = self.locale_manager.get_locale_info()
        assert current_info == self.locale_manager.get_locale_info('en_US')

    def test_is_rtl(self):
        """Test RTL detection."""
        assert not self.locale_manager.is_rtl('en_US')
        assert not self.locale_manager.is_rtl('it_IT')
        assert self.locale_manager.is_rtl('ar_EG')

        # Test with current locale
        assert not self.locale_manager.is_rtl()

    def test_get_currency_code(self):
        """Test getting currency code."""
        assert self.locale_manager.get_currency_code('en_US') == 'USD'
        assert self.locale_manager.get_currency_code('it_IT') == 'EUR'
        assert self.locale_manager.get_currency_code('ar_EG') == 'EGP'

        # Test with current locale
        assert self.locale_manager.get_currency_code() == 'USD'

    def test_get_formatter(self):
        """Test getting Babel formatter."""
        formatter = self.locale_manager.get_formatter()
        assert formatter is not None

        # Test formatter for specific locale
        it_formatter = self.locale_manager.get_formatter('it_IT')
        assert it_formatter is not None

    def test_get_translation_dict(self):
        """Test getting translation dictionary."""
        translations = self.locale_manager.get_translation_dict('it_IT')
        assert isinstance(translations, dict)
        assert 'loading' in translations
        assert 'complete' in translations
        assert translations['loading'] == 'Caricamento...'

        # Test fallback to English
        unknown_translations = self.locale_manager.get_translation_dict('unknown')
        assert unknown_translations['loading'] == 'Loading...'

    @patch('locale.getdefaultlocale')
    def test_get_system_locale(self, mock_getdefaultlocale):
        """Test system locale detection."""
        # Test with valid system locale
        mock_getdefaultlocale.return_value = ('en_US', 'UTF-8')
        system_locale = self.locale_manager._get_system_locale()
        assert system_locale == 'en_US'

        # Test with locale that needs mapping
        mock_getdefaultlocale.return_value = ('it_IT.UTF-8', 'UTF-8')
        system_locale = self.locale_manager._get_system_locale()
        assert system_locale == 'it_IT'

        # Test with language-only locale
        mock_getdefaultlocale.return_value = ('de', 'UTF-8')
        system_locale = self.locale_manager._get_system_locale()
        assert system_locale == 'de_DE'

        # Test with error
        mock_getdefaultlocale.side_effect = Exception('Error')
        system_locale = self.locale_manager._get_system_locale()
        assert system_locale is None

    def test_analyze_path_for_locale_hints(self):
        """Test path analysis for locale hints."""
        # Test with Italian file path
        italian_path = Path('/home/user/documenti_italiani/report.xlsx')
        locale = self.locale_manager._analyze_path_for_locale_hints(italian_path)
        assert locale == 'it_IT'

        # Test with German file path
        german_path = Path('/home/user/deutsche_berichte/data.xlsx')
        locale = self.locale_manager._analyze_path_for_locale_hints(german_path)
        assert locale == 'de_DE'

        # Test with no hints
        neutral_path = Path('/home/user/documents/data.xlsx')
        locale = self.locale_manager._analyze_path_for_locale_hints(neutral_path)
        assert locale is None

    @patch.object(LocaleManager, '_get_system_locale')
    @patch.object(LocaleManager, '_analyze_file_locale')
    def test_detect_locale(self, mock_analyze_file, mock_system_locale):
        """Test locale detection with different scenarios."""
        # Test system locale priority
        mock_system_locale.return_value = 'it_IT'
        mock_analyze_file.return_value = 'de_DE'

        detected = self.locale_manager.detect_locale(
            input_file='/path/to/file.xlsx',
            system_locale=True,
            file_analysis=True
        )
        assert detected == 'it_IT'

        # Test file analysis priority when system locale disabled
        detected = self.locale_manager.detect_locale(
            input_file='/path/to/file.xlsx',
            system_locale=False,
            file_analysis=True
        )
        assert detected == 'de_DE'

        # Test fallback to default
        mock_system_locale.return_value = None
        mock_analyze_file.return_value = None

        detected = self.locale_manager.detect_locale(
            input_file='/path/to/file.xlsx',
            system_locale=True,
            file_analysis=True
        )
        assert detected == self.locale_manager.default_locale

    def test_detect_locale_no_file(self):
        """Test locale detection without file."""
        with patch.object(self.locale_manager, '_get_system_locale', return_value=None):
            detected = self.locale_manager.detect_locale(
                input_file=None,
                system_locale=True,
                file_analysis=False
            )
            assert detected == self.locale_manager.default_locale

    @patch.object(LocaleManager, '_analyze_file_locale')
    def test_detect_locale_with_unsupported_system_locale(self, mock_analyze_file):
        """Test detection when system locale is not supported."""
        with patch.object(self.locale_manager, '_get_system_locale', return_value='unsupported'):
            mock_analyze_file.return_value = 'it_IT'

            detected = self.locale_manager.detect_locale(
                input_file='/path/to/file.xlsx',
                system_locale=True,
                file_analysis=True
            )
            assert detected == 'it_IT'

    def test_supported_locales_structure(self):
        """Test that all supported locales have required structure."""
        required_keys = ['name', 'date_format', 'time_format', 'currency',
                        'number_format', 'rtl', 'decimal_separator', 'thousands_separator']

        for locale_code, locale_info in self.locale_manager.SUPPORTED_LOCALES.items():
            for key in required_keys:
                assert key in locale_info, f"Missing key '{key}' in locale '{locale_code}'"

            # Validate RTL is boolean
            assert isinstance(locale_info['rtl'], bool)

            # Validate currency is 3-letter code
            assert len(locale_info['currency']) == 3

    def test_locale_codes_consistency(self):
        """Test that locale codes are consistent with Babel format."""
        for locale_code in self.locale_manager.SUPPORTED_LOCALES.keys():
            # Should be in format language_COUNTRY
            assert '_' in locale_code
            assert len(locale_code.split('_')) == 2

            # Should be parseable by Babel
            try:
                from babel.core import Locale
                Locale.parse(locale_code)
            except UnknownLocaleError:
                pytest.fail(f"Locale '{locale_code}' is not recognized by Babel")

    def test_get_formatter_caching(self):
        """Test that formatters are cached properly."""
        # Get formatter for first time
        formatter1 = self.locale_manager.get_formatter('de_DE')

        # Get formatter again
        formatter2 = self.locale_manager.get_formatter('de_DE')

        # Should be the same object (cached)
        assert formatter1 is formatter2

        # Different locale should return different formatter
        formatter3 = self.locale_manager.get_formatter('fr_FR')
        assert formatter1 is not formatter3

    def test_timezone_handling(self):
        """Test timezone handling in formatters."""
        # Test with invalid timezone
        manager = LocaleManager(default_locale='en_US', timezone='Invalid/Timezone')

        # Should not raise error but use UTC fallback
        formatter = manager.get_formatter()
        assert formatter is not None

    @patch('locale.setlocale')
    def test_python_locale_setting(self, mock_setlocale):
        """Test Python locale setting."""
        mock_setlocale.return_value = 'en_US.UTF-8'

        self.locale_manager.set_locale('en_US')
        mock_setlocale.assert_called_with(locale.LC_ALL, 'en_US')

    @patch('locale.setlocale')
    def test_python_locale_setting_error(self, mock_setlocale):
        """Test handling of Python locale setting errors."""
        mock_setlocale.side_effect = locale.Error('Unsupported locale')

        # Should not raise error, just log warning
        self.locale_manager.set_locale('en_US')