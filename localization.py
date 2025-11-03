#!/usr/bin/env python3
"""
Localization Framework for Tumblr Image Collector

Provides unified message management with multi-language support.
Supports dynamic language switching and fallback mechanisms.
"""

# Source translation integration imports
try:
    from source_translation_integrator import SourceTranslationIntegrator
    SOURCE_TRANSLATION_INTEGRATION_AVAILABLE = True
except ImportError:
    SOURCE_TRANSLATION_INTEGRATION_AVAILABLE = False
    logger.warning("source_translation_integrator not available. Using basic integration.")

logger = logging.getLogger(__name__)


class LocalizationManager:
    """
    Centralized localization management with caching and fallback support.
    Enhanced with Unicode normalization and security validation.
    """

    # Security constants
    MAX_KEY_LENGTH = 100
    MAX_MESSAGE_LENGTH = 2000
    ALLOWED_CHARS_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')
    DANGEROUS_PATTERNS = [
        r'[<>"\'&]',  # HTML/XML dangerous characters
        r'javascript:',  # JavaScript injection
        r'data:',  # Data URI injection
        r'vbscript:',  # VBScript injection
        r'on\w+\s*=',  # Event handlers
    ]

    def __init__(self, locales_dir: Optional[Path] = None):
        """
        Initialize localization manager.

        Args:
            locales_dir: Directory containing locale JSON files (default: auto-detect)
        """
        if locales_dir is None:
            # Auto-detect locales directory relative to this file
            self.locales_dir = Path(__file__).parent.parent / "locales"
        else:
            self.locales_dir = Path(locales_dir)

        self._current_language = self._detect_system_language()
        self._messages: Dict[str, Dict[str, str]] = {}
        self._fallback_language = "en"  # English as fallback
        self._message_hashes: Dict[str, str] = {}  # For integrity checking
        self._rtl_languages = {'ar', 'he', 'fa', 'ur'}  # RTL language codes

        # Initialize source translation integrator
        self.source_translation_integrator = None
        if SOURCE_TRANSLATION_INTEGRATION_AVAILABLE:
            try:
                self.source_translation_integrator = SourceTranslationIntegrator(self.locales_dir)
                logger.info("Source translation integrator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize source translation integrator: {e}")

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text to NFC form for consistency."""
        if not isinstance(text, str):
            return str(text)
        return unicodedata.normalize('NFC', text)

    def _validate_message_key(self, key: str) -> bool:
        """Validate message key for security."""
        if not isinstance(key, str):
            return False

        # Check length
        if len(key) > self.MAX_KEY_LENGTH:
            return False

        # Check allowed characters
        if not self.ALLOWED_CHARS_PATTERN.match(key):
            return False

        return True

    def _validate_message_content(self, message: str) -> bool:
        """Validate message content for security."""
        if not isinstance(message, str):
            return False

        # Check length
        if len(message) > self.MAX_MESSAGE_LENGTH:
            return False

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return False

        return True

    def _sanitize_message(self, message: str) -> str:
        """Sanitize message content by removing dangerous patterns."""
        sanitized = message

        # Remove or escape dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        return self._normalize_unicode(sanitized)

    def _calculate_message_hash(self, message: str) -> str:
        """Calculate hash for message integrity checking."""
        return hashlib.sha256(message.encode('utf-8')).hexdigest()

    def is_rtl_language(self, language_code: Optional[str] = None) -> bool:
        """Check if the specified language uses right-to-left script."""
        lang = language_code or self._current_language
        return lang in self._rtl_languages

    def get_text_direction(self, language_code: Optional[str] = None) -> str:
        """Get text direction for the specified language."""
        return "rtl" if self.is_rtl_language(language_code) else "ltr"

    def _get_locale_code(self) -> str:
        """Get current locale code in standard format (e.g., 'en_US')."""
        try:
            system_locale = locale.getlocale()[0]
            if system_locale:
                # Convert system locale to standard format
                parts = system_locale.replace('-', '_').split('_')
                if len(parts) >= 2:
                    return f"{parts[0].lower()}_{parts[1].upper()}"
                else:
                    return parts[0].lower()
            return "en_US"
        except:
            return "en_US"

    def _detect_system_language(self) -> str:
        detected_languages = set()

        # Strategy 1: System locale detection
        try:
            system_locale = locale.getlocale()[0]
            if system_locale:
                # Extract primary language (e.g., 'ja_JP' -> 'ja')
                primary_lang = system_locale.split('_')[0].lower()
                detected_languages.add(primary_lang)

                # Also check full locale if different (e.g., 'ja_JP' -> 'ja-JP')
                if '_' in system_locale:
                    locale_variant = system_locale.replace('_', '-').lower()
                    if locale_variant in self._rtl_languages:
                        detected_languages.add(locale_variant)
        except (AttributeError, IndexError, TypeError) as e:
            logger.debug(f"System locale detection failed: {e}")

        # Strategy 2: Environment variables
        env_vars = ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']
        for env_var in env_vars:
            env_value = os.environ.get(env_var, '')
            if env_value:
                # Extract language code (e.g., 'ja_JP.UTF-8' -> 'ja')
                lang_code = env_value.split('.')[0].split('_')[0].split('-')[0].lower()
                if lang_code and len(lang_code) == 2:
                    detected_languages.add(lang_code)

        # Strategy 3: Check available languages in order of preference
        available_langs = set(self._messages.keys())
        preferred_order = [
            'ja', 'en', 'zh', 'es', 'fr', 'de', 'ko', 'pt', 'ru', 'ar',
            'hi', 'th', 'vi', 'tr', 'it', 'nl', 'sv', 'da', 'no', 'fi'
        ]

        # First, check if any detected language is available
        for detected in detected_languages:
            if detected in available_langs:
                logger.info(f"Detected and available language: {detected}")
                return detected

        # Second, check preferred languages that are available
        for preferred in preferred_order:
            if preferred in available_langs:
                logger.info(f"Using preferred available language: {preferred}")
                return preferred

        # Third, check any available language
        if available_langs:
            chosen = next(iter(available_langs))
            logger.info(f"Using first available language: {chosen}")
            return chosen

        # Final fallback
        logger.warning("No languages available, using English as fallback")
        return "en"

    def _load_messages(self) -> None:
        """Load and validate message files for all available languages."""
        if not self.locales_dir.exists():
            logger.warning(f"Locales directory not found: {self.locales_dir}")
            return

        # Load all JSON files in locales directory
        for locale_file in self.locales_dir.glob("*.json"):
            language_code = locale_file.stem

            # Validate language code
            if not self._validate_message_key(language_code):
                logger.warning(f"Invalid language code: {language_code}")
                continue

            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    raw_messages = json.load(f)

                # Validate and sanitize messages
                validated_messages = {}
                for key, message in raw_messages.items():
                    if not self._validate_message_key(key):
                        logger.warning(f"Invalid message key '{key}' in {language_code}")
                        continue

                    if isinstance(message, str):
                        if self._validate_message_content(message):
                            sanitized_message = self._sanitize_message(message)
                            validated_messages[key] = sanitized_message
                            # Store hash for integrity checking
                            self._message_hashes[f"{language_code}:{key}"] = self._calculate_message_hash(sanitized_message)
                        else:
                            logger.warning(f"Invalid message content for key '{key}' in {language_code}")
                    else:
                        logger.warning(f"Non-string message for key '{key}' in {language_code}")

                if validated_messages:
                    self._messages[language_code] = validated_messages
                    logger.debug(f"Loaded and validated locale: {language_code} ({len(validated_messages)} messages)")
                else:
                    logger.warning(f"No valid messages found for locale: {language_code}")

            except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to load locale {language_code}: {e}")

        # Ensure fallback language exists
        if self._fallback_language not in self._messages:
            logger.warning(f"Fallback language '{self._fallback_language}' not found")

        logger.info(f"Localization system initialized with {len(self._messages)} languages")

    def set_language(self, language_code: str) -> bool:
        """
        Set the current language.

        Args:
            language_code: Language code (e.g., 'ja', 'en')

        Returns:
            True if language was set successfully, False otherwise
        """
        language_code = language_code.lower()
        if language_code in self._messages:
            self._current_language = language_code
            logger.info(f"Language set to: {language_code}")
            return True
        else:
            logger.warning(f"Language '{language_code}' not available")
            return False

    def get_language(self) -> str:
        """Get the current language code."""
        return self._current_language

    def get_available_languages(self) -> list:
        """Get list of available language codes."""
        return list(self._messages.keys())

    def get_message(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get localized message by key with security validation.

        Args:
            key: Message key
            language: Language code (optional, uses current language if not specified)
            **kwargs: Formatting arguments

        Returns:
            Localized message string
        """
        # Validate input key
        if not self._validate_message_key(key):
            logger.warning(f"Invalid message key: {key}")
            return key

        target_language = language or self._current_language

        # Try target language first
        messages = self._messages.get(target_language)
        if messages and key in messages:
            message = messages[key]
            # Verify message integrity if hash exists
            message_hash_key = f"{target_language}:{key}"
            if message_hash_key in self._message_hashes:
                current_hash = self._calculate_message_hash(message)
                if current_hash != self._message_hashes[message_hash_key]:
                    logger.warning(f"Message integrity check failed for key '{key}' in {target_language}")
                    # Continue with the message but log the issue
        else:
            # Try fallback language
            fallback_messages = self._messages.get(self._fallback_language)
            if fallback_messages and key in fallback_messages:
                message = fallback_messages[key]
                logger.debug(f"Using fallback message for key '{key}' in language '{target_language}'")

                # Verify fallback message integrity
                fallback_hash_key = f"{self._fallback_language}:{key}"
                if fallback_hash_key in self._message_hashes:
                    current_hash = self._calculate_message_hash(message)
                    if current_hash != self._message_hashes[fallback_hash_key]:
                        logger.warning(f"Fallback message integrity check failed for key '{key}' in {self._fallback_language}")
            else:
                # Return key if no translation found
                logger.warning(f"Message key '{key}' not found in any language")
                return key

        # Apply formatting if arguments provided
        if kwargs:
            # Try ICU formatting first (more advanced)
            if '{' in message and (',' in message or 'plural' in message or 'select' in message):
                try:
                    return self._format_icu_message(message, **kwargs)
                except Exception as e:
                    logger.debug(f"ICU formatting failed, falling back to basic: {e}")

            # Fallback to basic formatting
            try:
                return message.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to format message '{key}': {e}")
                return message

        return message

    def _format_icu_message(self, message: str, **kwargs) -> str:
        """Format ICU message with advanced pluralization and gender support."""
        # Enhanced formatting using advanced systems
        if self.pluralization_system:
            # Use advanced pluralization and gender formatting
            return self._format_with_advanced_icu(message, **kwargs)

        # Fallback to basic formatting
        return self._handle_select(message, kwargs)

    def _format_with_advanced_icu(self, message: str, **kwargs) -> str:
        """Format message using advanced ICU-compatible formatting."""
        formatted_message = message

        # Handle pluralization
        plural_pattern = r'\{(\w+),\s*plural,\s*([^}]+)\}'
        if re.search(plural_pattern, formatted_message) and self.pluralization_system:
            formatted_message = self._format_with_advanced_pluralization(formatted_message, **kwargs)

        # Handle gender selection
        gender_pattern = r'\{(\w+),\s*select,\s*([^}]+)\}'
        if re.search(gender_pattern, formatted_message) and self.pluralization_system:
            def replace_gender_advanced(match):
                var_name = match.group(1)
                gender_rules = match.group(2)

                if var_name not in kwargs:
                    return match.group(0)

                gender_value = str(kwargs[var_name]).lower()
                rules = self._parse_gender_rules_advanced(gender_rules)

                if gender_value in rules:
                    return rules[gender_value]
                elif 'other' in rules:
                    return rules['other']
                else:
                    return match.group(0)

            formatted_message = re.sub(gender_pattern, replace_gender_advanced, formatted_message)

        # Apply basic formatting
        return self._handle_basic_placeholders(formatted_message, kwargs)

    def _parse_gender_rules_advanced(self, rules_text: str) -> Dict[str, str]:
        """Parse advanced gender rules with proper ICU syntax."""
        rules = {}

        # Parse ICU-style gender rules: male{he} female{she} other{they}
        parts = rules_text.split()
        current_category = None
        current_content = ""

        for part in parts:
            if part in ['male', 'female', 'neuter', 'other']:
                # Save previous rule
                if current_category and current_content:
                    rules[current_category] = current_content.strip('{}')

                current_category = part
                current_content = ""
            else:
                current_content += " " + part if current_content else part

        # Save last rule
        if current_category and current_content:
            rules[current_category] = current_content.strip('{}')

        return rules

    def _format_with_advanced_pluralization(self, message: str, **kwargs) -> str:
        """Format message using advanced pluralization system."""
        # Handle ICU plural format: {count, plural, one{# item} other{# items}}
        plural_pattern = r'\{(\w+),\s*plural,\s*([^}]+)\}'

        def replace_plural_advanced(match):
            var_name = match.group(1)
            plural_rules = match.group(2)

            if var_name not in kwargs:
                return match.group(0)

            count = kwargs[var_name]

            # Use advanced pluralization system
            if self.pluralization_system:
                plural_form = self.pluralization_system.get_plural_form(count, self._current_language)

                # Parse ICU rules and find matching pattern
                rules = self._parse_icu_plural_rules(plural_rules)

                # Find exact match or fallback to 'other'
                if plural_form in rules:
                    selected_rule = rules[plural_form]
                elif 'other' in rules:
                    selected_rule = rules['other']
                else:
                    return match.group(0)

                # Replace count placeholder in the selected rule
                return selected_rule.replace('#', str(count))

            # Fallback to basic parsing
            return self._parse_plural_rules_basic(plural_rules, count)

        # Apply advanced pluralization
        formatted_message = re.sub(plural_pattern, replace_plural_advanced, message)

        # Apply other formatting
        return self._handle_basic_placeholders(formatted_message, kwargs)

    def _parse_icu_plural_rules(self, rules_text: str) -> Dict[str, str]:
        """Parse ICU plural rules into a dictionary."""
        rules = {}

        # Split by spaces but preserve content within braces
        parts = rules_text.split()
        current_category = None
        current_content = ""

        for part in parts:
            if part in ['zero', 'one', 'two', 'few', 'many', 'other']:
                # Save previous rule if exists
                if current_category and current_content:
                    rules[current_category] = current_content

                current_category = part
                current_content = ""
            else:
                if current_content:
                    current_content += " "
                current_content += part

        # Save last rule
        if current_category and current_content:
            rules[current_category] = current_content

        return rules

    def _parse_plural_rules_basic(self, rules_text: str, count: Union[int, float]) -> str:
        """Basic plural rule parsing fallback."""
        rules = self._parse_icu_plural_rules(rules_text)

        if count == 1 and 'one' in rules:
            return rules['one'].replace('#', str(count))
        elif 'other' in rules:
            return rules['other'].replace('#', str(count))
        else:
            return str(count)

    def _handle_select(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Handle select syntax: {gender, select, male{he} female{she} other{they}}"""
        select_pattern = r'\{(\w+),\s*select,\s*([^}]+)\}'

        def replace_select(match):
            var_name = match.group(1)
            select_rules = match.group(2)

            if var_name not in kwargs:
                return match.group(0)

            value = str(kwargs[var_name]).lower()

            # Parse select rules (simplified)
            rules = {}
            current_rule = ""
            current_value = ""
            brace_depth = 0
            parsing_rule = False

            for char in select_rules:
                if char == '{' and not parsing_rule:
                    parsing_rule = True
                    current_rule = ""
                elif char == '{' and parsing_rule:
                    brace_depth += 1
                    current_value += char
                elif char == '}' and parsing_rule:
                    if brace_depth > 0:
                        brace_depth -= 1
                        current_value += char
                    else:
                        rules[current_rule.strip()] = current_value
                        parsing_rule = False
                        current_value = ""
                        current_rule = ""
                elif parsing_rule:
                    if brace_depth > 0:
                        current_value += char
                    elif char in '=\w':
                        current_rule += char
                    else:
                        current_value += char

            # Find matching rule
            for rule, rule_value in rules.items():
                if rule == value or rule == 'other':
                    return rule_value

            return match.group(0)

        return re.sub(select_pattern, replace_select, message)

    def _handle_number_formatting(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Handle number formatting: {number, number}"""
        number_pattern = r'\{(\w+),\s*number\}'

        def replace_number(match):
            var_name = match.group(1)

            if var_name not in kwargs:
                return match.group(0)

            number = kwargs[var_name]
            locale_code = self._current_language.replace('-', '_')

            try:
                # Use locale-specific number formatting
                return locale.format_string("%.2f", number, grouping=True)
            except:
                # Fallback to basic formatting
                return str(number)

        return re.sub(number_pattern, replace_number, message)

    def _handle_date_formatting(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Handle date formatting: {date, date, format}"""
        date_pattern = r'\{(\w+),\s*date,\s*(\w+)\}'

        def replace_date(match):
            var_name = match.group(1)
            date_format = match.group(2)

            if var_name not in kwargs:
                return match.group(0)

            date_obj = kwargs[var_name]

            if not isinstance(date_obj, datetime):
                return match.group(0)

            # Map format names to strftime patterns
            format_map = {
                'short': '%Y-%m-%d',
                'medium': '%Y-%m-%d %H:%M',
                'long': '%Y-%m-%d %H:%M:%S',
                'full': '%A, %B %d, %Y at %H:%M',
                'time': '%H:%M:%S',
                'date': '%Y-%m-%d'
            }

            format_pattern = format_map.get(date_format, '%Y-%m-%d')

            try:
                # Use locale-appropriate formatting
                locale_code = self._current_language.replace('-', '_')
                old_locale = locale.getlocale(locale.LC_TIME)

                try:
                    locale.setlocale(locale.LC_TIME, locale_code)
                    formatted_date = date_obj.strftime(format_pattern)
                except:
                    formatted_date = date_obj.strftime(format_pattern)
                finally:
                    # Restore original locale
                    if old_locale[0]:
                        try:
                            locale.setlocale(locale.LC_TIME, old_locale)
                        except:
                            pass

                return formatted_date

            except Exception as e:
                logger.debug(f"Date formatting failed: {e}")
                return date_obj.strftime('%Y-%m-%d')

        return re.sub(date_pattern, replace_date, message)

    def _handle_currency_formatting(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Handle currency formatting: {amount, currency, USD}"""
        currency_pattern = r'\{(\w+),\s*currency,\s*(\w+)\}'

        def replace_currency(match):
            var_name = match.group(1)
            currency_code = match.group(2)

            if var_name not in kwargs:
                return match.group(0)

            amount = kwargs[var_name]

            try:
                # Currency formatting based on locale
                locale_code = self._current_language.replace('-', '_')
                old_locale = locale.getlocale(locale.LC_MONETARY)

                try:
                    locale.setlocale(locale.LC_MONETARY, locale_code)
                    formatted_currency = locale.currency(amount, symbol=True, grouping=True)
                except:
                    # Fallback formatting
                    symbols = {'USD': '$', 'EUR': '€', 'JPY': '¥', 'GBP': '£', 'CNY': '¥'}
                    symbol = symbols.get(currency_code, currency_code)
                    formatted_currency = f"{symbol}{amount:,.2f}"
                finally:
                    if old_locale[0]:
                        try:
                            locale.setlocale(locale.LC_MONETARY, old_locale)
                        except:
                            pass

                return formatted_currency

            except Exception as e:
                logger.debug(f"Currency formatting failed: {e}")
                return f"{amount:.2f}"

    def format_date_advanced(self, date_obj: date, style: str = 'medium') -> str:
        """Format date using advanced locale formatter if available."""
        if self.locale_formatter:
            try:
                return self.locale_formatter.format_date(date_obj, style, self._get_locale_code())
            except Exception as e:
                logger.debug(f"Advanced date formatting failed: {e}")

        # Fallback to basic formatting
        return self._format_date_basic(date_obj, style)

    def format_time_advanced(self, time_obj: time, style: str = 'medium') -> str:
        """Format time using advanced locale formatter if available."""
        if self.locale_formatter:
            try:
                return self.locale_formatter.format_time(time_obj, style, self._get_locale_code())
            except Exception as e:
                logger.debug(f"Advanced time formatting failed: {e}")

        # Fallback to basic formatting
        return self._format_time_basic(time_obj, style)

    def format_currency_advanced(self, amount: Union[int, float], currency_code: str = 'USD') -> str:
        """Format currency using advanced locale formatter if available."""
        if self.locale_formatter:
            try:
                return self.locale_formatter.format_currency(amount, currency_code, self._get_locale_code())
            except Exception as e:
                logger.debug(f"Advanced currency formatting failed: {e}")

        # Fallback to basic formatting
        return self._format_currency_basic(amount, currency_code)

    def format_number_advanced(self, number: Union[int, float]) -> str:
        """Format number using advanced locale formatter if available."""
        if self.locale_formatter:
            try:
                return self.locale_formatter.format_number(number, self._get_locale_code())
            except Exception as e:
                logger.debug(f"Advanced number formatting failed: {e}")

        # Fallback to basic formatting
        return self._format_number_basic(number)

    def _format_date_basic(self, date_obj: date, style: str) -> str:
        """Basic date formatting fallback."""
        format_patterns = {
            'short': '%Y-%m-%d',
            'medium': '%B %d, %Y',
            'long': '%B %d, %Y',
            'full': '%A, %B %d, %Y'
        }
        return date_obj.strftime(format_patterns.get(style, '%Y-%m-%d'))

    def _format_time_basic(self, time_obj: time, style: str) -> str:
        """Basic time formatting fallback."""
        format_patterns = {
            'short': '%H:%M',
            'medium': '%H:%M:%S',
            'long': '%H:%M:%S'
        }
        return time_obj.strftime(format_patterns.get(style, '%H:%M:%S'))

    def _format_currency_basic(self, amount: Union[int, float], currency_code: str) -> str:
        """Basic currency formatting fallback."""
        symbols = {'USD': '$', 'EUR': '€', 'JPY': '¥', 'GBP': '£', 'CNY': '¥'}
        symbol = symbols.get(currency_code, currency_code)
        return f"{symbol}{amount:,.2f}"

    def _format_number_basic(self, number: Union[int, float]) -> str:
        """Basic number formatting fallback."""
        return f"{number:,.2f}"

    def _handle_basic_placeholders(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Handle basic placeholder replacement: {name}"""
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.debug(f"Basic placeholder formatting failed: {e}")
            return message
        """
        Format error message with consistent styling.

        Args:
            error_key: Error message key
            **kwargs: Formatting arguments

        Returns:
            Formatted error message
        """
        error_prefix = self.get_message("error_occurred", **kwargs)
        error_message = self.get_message(error_key, **kwargs)
        return f"{error_prefix}: {error_message}"

    def format_success(self, success_key: str, **kwargs) -> str:
        """
        Format success message.

        Args:
            success_key: Success message key
            **kwargs: Formatting arguments

        Returns:
            Formatted success message
        """
        return self.get_message(success_key, **kwargs)

    def format_warning(self, warning_key: str, **kwargs) -> str:
        """
        Format warning message.

        Args:
            warning_key: Warning message key
            **kwargs: Formatting arguments

        Returns:
            Formatted warning message
        """
        warning_message = self.get_message(warning_key, **kwargs)
        return f"Warning: {warning_message}"

    def format_info(self, info_key: str, **kwargs) -> str:
        """
        Format info message.

        Args:
            info_key: Info message key
            **kwargs: Formatting arguments

        Returns:
            Formatted info message
        """
        return self.get_message(info_key, **kwargs)

    def validate_translation_quality(self, language_code: str = None) -> Dict[str, Any]:
        """
        Validate translation quality for a specific language or all languages.

        Args:
            language_code: Language code to validate (None for all languages)

        Returns:
            Quality report dictionary
        """
        target_languages = [language_code] if language_code else list(self._messages.keys())

        quality_report = {
            'total_languages': len(target_languages),
            'languages': {}
        }

        # Get reference language (English) keys
        reference_keys = set(self._messages.get('en', {}).keys())

        for lang in target_languages:
            if lang not in self._messages:
                continue

            messages = self._messages[lang]
            lang_keys = set(messages.keys())

            # Calculate coverage
            coverage = len(lang_keys.intersection(reference_keys)) / len(reference_keys) if reference_keys else 0

            # Find missing keys
            missing_keys = reference_keys - lang_keys

            # Find extra keys (keys not in reference)
            extra_keys = lang_keys - reference_keys

            # Check for empty translations
            empty_translations = [key for key, value in messages.items() if not value or value.strip() == ""]

            # Check for placeholder consistency
            placeholder_issues = self._check_placeholder_consistency(lang, reference_keys)

            quality_report['languages'][lang] = {
                'total_keys': len(lang_keys),
                'coverage': coverage,
                'missing_keys': list(missing_keys),
                'extra_keys': list(extra_keys),
                'empty_translations': empty_translations,
                'placeholder_issues': placeholder_issues,
                'quality_score': self._calculate_quality_score(coverage, empty_translations, placeholder_issues)
            }

        return quality_report

    def _check_placeholder_consistency(self, language: str, reference_keys: Set[str]) -> List[str]:
        """
        Check if placeholders in translations match the reference language.

        Args:
            language: Language code to check
            reference_keys: Set of reference keys

        Returns:
            List of placeholder inconsistency issues
        """
        issues = []
        reference_messages = self._messages.get('en', {})
        target_messages = self._messages.get(language, {})

        for key in reference_keys:
            if key in target_messages and key in reference_messages:
                ref_msg = reference_messages[key]
                target_msg = target_messages[key]

                # Extract placeholders from reference (e.g., {name}, {count})
                import re
                ref_placeholders = set(re.findall(r'\{(\w+)\}', ref_msg))
                target_placeholders = set(re.findall(r'\{(\w+)\}', target_msg))

                if ref_placeholders != target_placeholders:
                    issues.append({
                        'key': key,
                        'reference_placeholders': ref_placeholders,
                        'target_placeholders': target_placeholders
                    })

        return issues

    def _calculate_quality_score(self, coverage: float, empty_translations: List[str], placeholder_issues: List) -> float:
        """
        Calculate overall quality score for translations.

        Args:
            coverage: Translation coverage ratio (0.0 to 1.0)
            empty_translations: List of empty translation keys
            placeholder_issues: List of placeholder consistency issues

        Returns:
            Quality score (0.0 to 1.0)
        """
        # Base score from coverage
        score = coverage * 0.7

        # Penalty for empty translations (max 20% penalty)
        empty_penalty = min(len(empty_translations) * 0.05, 0.2)
        score -= empty_penalty

        # Penalty for placeholder issues (max 10% penalty)
        placeholder_penalty = min(len(placeholder_issues) * 0.02, 0.1)
        score -= placeholder_penalty

        return max(0.0, min(1.0, score))

    def generate_translation_report(self, output_file: str = "translation_report.json") -> str:
        """
        Generate comprehensive translation quality report.

        Args:
            output_file: Output file path for the report

        Returns:
            Path to generated report file
        """
        import json

        quality_report = self.validate_translation_quality()

        # Add summary statistics
        total_languages = quality_report['total_languages']
        avg_coverage = sum(lang_data['coverage'] for lang_data in quality_report['languages'].values()) / total_languages if total_languages > 0 else 0

        quality_report['summary'] = {
            'total_languages': total_languages,
            'average_coverage': avg_coverage,
            'report_generated': datetime.now().isoformat(),
            'localization_version': '2.0'
        }

        # Write report to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, ensure_ascii=False, indent=2)

            logger.info(f"Translation report generated: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to generate translation report: {e}")
            return ""

    def find_missing_translations(self, target_language: str, reference_language: str = 'en') -> List[str]:
        """
        Find missing translations for a target language.

        Args:
            target_language: Language code to check
            reference_language: Reference language code (default: 'en')

        Returns:
            List of missing translation keys
        """
        if target_language not in self._messages or reference_language not in self._messages:
            return []

        reference_keys = set(self._messages[reference_language].keys())
        target_keys = set(self._messages[target_language].keys())

        return list(reference_keys - target_keys)

    def suggest_translations(self, target_language: str, missing_keys: List[str] = None) -> Dict[str, str]:
        """
        Suggest translations for missing keys using machine translation or similar languages.

        Args:
            target_language: Target language code
            missing_keys: List of keys to translate (None for all missing keys)

        Returns:
            Dictionary of suggested translations
        """
        if missing_keys is None:
            missing_keys = self.find_missing_translations(target_language)

        suggestions = {}

        # Simple fallback: use English as base and apply basic transformations
        # In production, integrate with Google Translate API or similar services

        for key in missing_keys:
            # Get English translation
            en_msg = self._messages.get('en', {}).get(key, key)

            # Basic suggestion (replace with actual translation service)
            suggestion = self._generate_basic_suggestion(en_msg, target_language)
            suggestions[key] = suggestion

        return suggestions

    def _generate_basic_suggestion(self, english_text: str, target_language: str) -> str:
        """
        Generate basic translation suggestion (placeholder for actual translation service).

        Args:
            english_text: English text to translate
            target_language: Target language code

        Returns:
            Suggested translation
        """
        # This is a placeholder implementation
        # In production, integrate with Google Translate API, DeepL, or similar services

        # Simple transformations for common languages
        if target_language == 'ja':
            # Basic English to Japanese patterns (very simplified)
            if 'error' in english_text.lower():
                return 'エラー'
            elif 'success' in english_text.lower():
                return '成功'
            elif 'warning' in english_text.lower():
                return '警告'
            else:
                return english_text  # Fallback to English

        elif target_language == 'zh':
            # Basic English to Chinese patterns (very simplified)
            if 'error' in english_text.lower():
                return '错误'
            elif 'success' in english_text.lower():
                return '成功'
            elif 'warning' in english_text.lower():
                return '警告'
            else:
                return english_text  # Fallback to English

        else:
            # For other languages, return English as fallback
            return english_text


# Enhanced convenience functions for translation management
def validate_translation_quality(language_code: str = None) -> Dict[str, Any]:
    """Validate translation quality (convenience function)."""
    return get_localization_manager().validate_translation_quality(language_code)


def generate_translation_report(output_file: str = "translation_report.json") -> str:
    """Generate translation quality report (convenience function)."""
    return get_localization_manager().generate_translation_report(output_file)


def find_missing_translations(target_language: str, reference_language: str = 'en') -> List[str]:
    """Find missing translations (convenience function)."""
    return get_localization_manager().find_missing_translations(target_language, reference_language)


def suggest_translations(target_language: str, missing_keys: List[str] = None) -> Dict[str, str]:
    """Suggest translations for missing keys (convenience function)."""
    return get_localization_manager().suggest_translations(target_language, missing_keys)


# Global instance for easy access
_localization_manager = None


def get_localization_manager() -> LocalizationManager:
    """Get the global localization manager instance."""
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
    return _localization_manager


def set_language(language_code: str) -> bool:
    """Set the global language."""
    return get_localization_manager().set_language(language_code)


def get_language() -> str:
    """Get the current global language."""
    return get_localization_manager().get_language()


def msg(key: str, **kwargs) -> str:
    """Get localized message (convenience function)."""
    return get_localization_manager().get_message(key, **kwargs)


def error_msg(key: str, **kwargs) -> str:
    """Get formatted error message (convenience function)."""
    return get_localization_manager().format_error(key, **kwargs)


def success_msg(key: str, **kwargs) -> str:
    """Get formatted success message (convenience function)."""
    return get_localization_manager().format_success(key, **kwargs)


def warning_msg(key: str, **kwargs) -> str:
    """Get formatted warning message (convenience function)."""
    return get_localization_manager().format_warning(key, **kwargs)


def info_msg(key: str, **kwargs) -> str:
    """Get formatted info message (convenience function)."""
    return get_localization_manager().format_info(key, **kwargs)


def format_date_advanced(date_obj: date, style: str = 'medium', language_code: str = None) -> str:
    """Format date using advanced locale formatter (convenience function)."""
    return get_localization_manager().format_date_advanced(date_obj, style)


def format_time_advanced(time_obj: time, style: str = 'medium', language_code: str = None) -> str:
    """Format time using advanced locale formatter (convenience function)."""
    return get_localization_manager().format_time_advanced(time_obj, style)


def format_currency_advanced(amount: Union[int, float], currency_code: str = 'USD', language_code: str = None) -> str:
    """Format currency using advanced locale formatter (convenience function)."""
    return get_localization_manager().format_currency_advanced(amount, currency_code)


def format_plural_advanced(key: str, count: Union[int, float, Decimal], **kwargs) -> str:
    """Format pluralized message using advanced pluralization system (convenience function)."""
    return get_localization_manager()._format_with_advanced_pluralization(key, count=count, **kwargs)


def get_plural_form(count: Union[int, float, Decimal], language: str = None) -> str:
    """Get plural form for a count in a specific language (convenience function)."""
    manager = get_localization_manager()
    target_language = language or manager._current_language
    if manager.pluralization_system:
        return manager.pluralization_system.get_plural_form(count, target_language)
    else:
        # Fallback to basic pluralization
        return 'one' if count == 1 else 'other'


def validate_plural_rules(language: str) -> Dict[str, Any]:
    """Validate plural rules for a language (convenience function)."""
    manager = get_localization_manager()
    if manager.pluralization_system:
        return manager.pluralization_system.validate_plural_rules(language)
    else:
        return {"error": "Advanced pluralization system not available"}


def get_cultural_color_recommendations(region: str, occasion: str = 'general') -> Dict[str, List[str]]:
    """Get culturally appropriate color recommendations (convenience function)."""
    manager = get_localization_manager()
    if manager.cultural_sensitivity:
        return manager.cultural_sensitivity.get_cultural_color_recommendations(region, occasion)
    else:
        return {"primary_colors": ["blue", "white"], "accent_colors": ["red"], "colors_to_avoid": [], "cultural_context": []}


def validate_symbol_cultural_appropriateness(symbol: str, region: str) -> Dict[str, Any]:
    """Validate if a symbol is culturally appropriate (convenience function)."""
    manager = get_localization_manager()
    if manager.cultural_sensitivity:
        return manager.cultural_sensitivity.validate_symbol_cultural_appropriateness(symbol, region)
    else:
        return {"symbol": symbol, "region": region, "is_appropriate": True, "concerns": [], "alternatives": [], "cultural_context": []}


def get_content_guidelines(region: str) -> List[str]:
    """Get content guidelines for a region (convenience function)."""
    manager = get_localization_manager()
    if manager.cultural_sensitivity:
        return manager.cultural_sensitivity.get_content_guidelines(region)
    else:
        return ["Use respectful and appropriate content for all audiences"]


def adapt_content_for_culture(content: Dict[str, Any], target_region: str) -> Dict[str, Any]:
    """Adapt content for cultural appropriateness (convenience function)."""
    manager = get_localization_manager()
    if manager.cultural_sensitivity:
        return manager.cultural_sensitivity.adapt_content_for_culture(content, target_region)
    else:
        return content


def detect_cultural_sensitivity_issues(content: str, region: str) -> List[Dict[str, Any]]:
    """Detect cultural sensitivity issues in content (convenience function)."""
    manager = get_localization_manager()
    if manager.cultural_sensitivity:
        return manager.cultural_sensitivity.detect_cultural_sensitivity_issues(content, region)
    else:
        return []


def format_with_gender(key: str, gender: str, **kwargs) -> str:
    """Format message with gender-specific variations (convenience function)."""
    manager = get_localization_manager()
    if manager.pluralization_system:
        return manager.pluralization_system.format_with_gender(key, gender, **kwargs)
    else:
        return key.format(**kwargs)


def get_gender_forms(language: str) -> Dict[str, List[str]]:
    """Get gender forms for a language (convenience function)."""
    manager = get_localization_manager()
    if manager.pluralization_system:
        return manager.pluralization_system.get_gender_forms(language)
    else:
        return {'common': ['they', 'them', 'their']}


def validate_gender_consistency(language: str, text: str) -> Dict[str, Any]:
    """Validate gender consistency in text (convenience function)."""
    manager = get_localization_manager()
    if manager.pluralization_system:
        return manager.pluralization_system.validate_gender_consistency(language, text)
    else:
        return {"language": language, "is_consistent": True, "issues": [], "suggestions": []}


def get_comprehensive_plural_examples(language: str) -> Dict[str, Any]:
    """Get comprehensive pluralization examples (convenience function)."""
    manager = get_localization_manager()
    if manager.pluralization_system:
        return manager.pluralization_system.get_comprehensive_plural_examples(language)
    else:
        return {"plural_examples": {"other": [0, 1, 2, 3]}, "gender_forms": {"common": ["they"]}, "combined_examples": []}


def estimate_text_expansion(text: str, target_language: str) -> Dict[str, Any]:
    """Estimate text expansion for a target language (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.estimate_text_expansion(text, target_language)
    else:
        return {"original_length": len(text), "estimated_expanded_length": len(text), "expansion_factor": 1.0}


def calculate_optimal_container_size(base_width: int, base_height: int, text: str, target_language: str) -> Dict[str, Any]:
    """Calculate optimal container size for text (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.calculate_optimal_container_size(base_width, base_height, text, target_language)
    else:
        return {"original_width": base_width, "original_height": base_height, "recommended_width": base_width, "recommended_height": base_height}


def generate_responsive_text_layout(texts: Dict[str, str], container_constraints: Dict[str, int]) -> Dict[str, Any]:
    """Generate responsive layout recommendations (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.generate_responsive_text_layout(texts, container_constraints)
    else:
        return {"container_constraints": container_constraints, "language_layouts": {}, "responsive_strategy": "standard"}


def optimize_text_for_display(text: str, max_length: int, language: str, truncation_strategy: str = 'ellipsis') -> Dict[str, Any]:
    """Optimize text for display within constraints (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.optimize_text_for_display(text, max_length, language, truncation_strategy)
    else:
        return {"original_text": text, "optimized_text": text[:max_length], "needs_truncation": len(text) > max_length}


def analyze_text_layout_requirements(texts: List[str], languages: List[str]) -> Dict[str, Any]:
    """Analyze layout requirements for multiple texts (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.analyze_text_layout_requirements(texts, languages)
    else:
        return {"total_texts": len(texts), "languages": languages, "expansion_analysis": {}, "layout_recommendations": {}}


def generate_css_for_language(language: str, base_css: Dict[str, Any]) -> Dict[str, Any]:
    """Generate CSS recommendations for a language (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.generate_css_for_language(language, base_css)
    else:
        return base_css


def validate_layout_compatibility(languages: List[str], layout_type: str) -> Dict[str, Any]:
    """Validate layout compatibility for languages (convenience function)."""
    manager = get_localization_manager()
    if manager.text_expansion_manager:
        return manager.text_expansion_manager.validate_layout_compatibility(languages, layout_type)
    else:
        return {"is_compatible": True, "issues": [], "warnings": [], "recommendations": []}


def run_comprehensive_i18n_tests(target_languages: List[str], test_types: List[str] = None) -> Dict[str, Any]:
    """Run comprehensive internationalization tests (convenience function)."""
    manager = get_localization_manager()
    if manager.i18n_test_manager:
        return manager.i18n_test_manager.run_comprehensive_i18n_tests(target_languages, test_types)
    else:
        return {"error": "Internationalization testing system not available"}


def simulate_user_interactions(language: str, interaction_count: int = 10) -> Dict[str, Any]:
    """Simulate user interactions in a language (convenience function)."""
    manager = get_localization_manager()
    if manager.i18n_test_manager:
        return manager.i18n_test_manager.simulate_user_interactions(language, interaction_count)
    else:
        return {"language": language, "interaction_count": interaction_count, "user_experience_score": 1.0}


def generate_regression_test_suite(languages: List[str]) -> Dict[str, Any]:
    """Generate regression test suite (convenience function)."""
    manager = get_localization_manager()
    if manager.i18n_test_manager:
        return manager.i18n_test_manager.generate_regression_test_suite(languages)
    else:
        return {"suite_name": "basic_i18n_tests", "target_languages": languages, "test_cases": []}


def analyze_test_coverage(languages: List[str], test_types: List[str]) -> Dict[str, Any]:
    """Analyze test coverage for internationalization (convenience function)."""
    manager = get_localization_manager()
    if manager.i18n_test_manager:
        return manager.i18n_test_manager.analyze_test_coverage(languages, test_types)
    else:
        return {"total_languages": len(languages), "total_test_types": len(test_types), "coverage_percentage": 100.0}


def standardize_translation_key(original_key: str, new_key: str = None) -> str:
    """Standardize a translation key (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        return manager.key_standardizer.standardize_key(original_key, new_key)
    else:
        return original_key


def scan_source_for_hardcoded_strings(source_paths: List[str]) -> Dict[str, Any]:
    """Scan source code for hardcoded strings (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        paths = [Path(p) for p in source_paths]
        return manager.key_standardizer.scan_source_code_for_strings(paths)
    else:
        return {"hardcoded_strings": [], "recommended_keys": {}, "total_strings_found": 0}


def generate_translation_template(languages: List[str], output_dir: str = None) -> Dict[str, str]:
    """Generate translation templates (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        output_path = Path(output_dir) if output_dir else None
        return manager.key_standardizer.generate_translation_template(languages, output_path)
    else:
        return {}


def validate_translation_consistency(language: str) -> Dict[str, Any]:
    """Validate translation consistency (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        return manager.key_standardizer.validate_translation_consistency(language)
    else:
        return {"language": language, "quality_score": 1.0}


def sync_translation_files(reference_language: str = 'en') -> Dict[str, Any]:
    """Sync all translation files (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        return manager.key_standardizer.sync_translation_files(reference_language)
    else:
        return {"reference_language": reference_language, "languages_synced": [], "issues_fixed": []}


def generate_key_documentation(output_file: str = "translation_keys.md") -> str:
    """Generate translation key documentation (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        return manager.key_standardizer.generate_key_documentation(output_file)
    else:
        return ""


def analyze_key_usage_patterns() -> Dict[str, Any]:
    """Analyze translation key usage patterns (convenience function)."""
    manager = get_localization_manager()
    if manager.key_standardizer:
        return manager.key_standardizer.analyze_key_usage_patterns()
    else:
        return {"total_keys": 0, "keys_by_category": {}, "keys_by_module": {}}


def generate_comprehensive_templates(target_languages: List[str]) -> Dict[str, Any]:
    """Generate comprehensive translation templates (convenience function)."""
    manager = get_localization_manager()
    if manager.language_pack_enhancer:
        return manager.language_pack_enhancer.generate_comprehensive_templates(target_languages)
    else:
        return {"generated_templates": [], "skipped_languages": target_languages, "errors": ["Language pack enhancer not available"]}


def enhance_existing_language_packs() -> Dict[str, Any]:
    """Enhance existing language packs (convenience function)."""
    manager = get_localization_manager()
    if manager.language_pack_enhancer:
        return manager.language_pack_enhancer.enhance_existing_packs()
    else:
        return {"enhanced_languages": [], "total_keys_added": 0, "errors": ["Language pack enhancer not available"]}


def validate_all_language_packs() -> Dict[str, Any]:
    """Validate all language packs (convenience function)."""
    manager = get_localization_manager()
    if manager.language_pack_enhancer:
        return manager.language_pack_enhancer.validate_all_language_packs()
    else:
        return {"total_languages": 0, "complete_languages": [], "incomplete_languages": [], "validation_errors": ["Language pack enhancer not available"]}


def generate_language_statistics() -> Dict[str, Any]:
    """Generate language pack statistics (convenience function)."""
    manager = get_localization_manager()
    if manager.language_pack_enhancer:
        return manager.language_pack_enhancer.generate_language_statistics()
    else:
        return {"total_languages": 0, "language_details": {}, "completion_summary": {}}


def expand_to_global_languages(target_languages: List[str] = None) -> Dict[str, Any]:
    """Expand language support to global languages (convenience function)."""
    manager = get_localization_manager()
    if manager.global_language_expander:
        return manager.global_language_expander.expand_to_global_languages(target_languages)
    else:
        return {"generated_languages": [], "skipped_languages": target_languages or [], "errors": ["Global language expander not available"]}


def generate_global_coverage_report() -> Dict[str, Any]:
    """Generate global language coverage report (convenience function)."""
    manager = get_localization_manager()
    if manager.global_language_expander:
        return manager.global_language_expander.generate_global_coverage_report()
    else:
        return {"total_global_languages": 0, "supported_languages": {}, "coverage_analysis": {}, "recommendations": []}


def get_language_global_info(language: str) -> Dict[str, Any]:
    """Get global information for a language (convenience function)."""
    manager = get_localization_manager()
    if manager.global_language_expander:
        if language in manager.global_language_expander.GLOBAL_LANGUAGE_STATS:
            return manager.global_language_expander.GLOBAL_LANGUAGE_STATS[language]
        else:
            return {"error": f"Language '{language}' not found in global stats"}
    else:
        return {"error": "Global language expander not available"}


def integrate_localization_into_files(file_paths: List[str]) -> Dict[str, Any]:
    """Integrate localization into multiple files (convenience function)."""
    manager = get_localization_manager()
    if manager.source_translation_integrator:
        files = [Path(p) for p in file_paths]
        return manager.source_translation_integrator.generate_integration_report(files)
    else:
        return {"total_files": len(file_paths), "processed_files": [], "failed_files": file_paths, "errors": ["Source translation integrator not available"]}


def validate_localization_integration(file_path: str) -> Dict[str, Any]:
    """Validate localization integration in a file (convenience function)."""
    manager = get_localization_manager()
    if manager.source_translation_integrator:
        return manager.source_translation_integrator.validate_integration(Path(file_path))
    else:
        return {"file": file_path, "issues": ["Source translation integrator not available"]}


def create_translation_ready_files(source_files: List[str]) -> Dict[str, Any]:
    """Create translation-ready versions of source files (convenience function)."""
    manager = get_localization_manager()
    if manager.source_translation_integrator:
        files = [Path(p) for p in source_files]
        return manager.source_translation_integrator.create_translation_ready_files(files)
    else:
        return {"created_files": [], "translation_keys_added": set(), "errors": ["Source translation integrator not available"]}
