# Tumblr Image Collector Mobile App
# Kivy-based cross-platform mobile application
# Atlassian Design System implementation

import os
import sys
import json
import threading
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

from tumblr_image_collector import TumblrImageCollector

# Set window size for desktop testing
Window.size = (400, 700)  # Mobile-like dimensions for desktop testing

# Atlassian Design System Tokens for Kivy
class AtlassianDesignTokens:
    """Atlassian Design System tokens adapted for Kivy"""

    # Spacing (8px base converted to dp)
    SPACE_025 = dp(2)
    SPACE_050 = dp(4)
    SPACE_075 = dp(6)
    SPACE_100 = dp(8)
    SPACE_150 = dp(12)
    SPACE_200 = dp(16)
    SPACE_300 = dp(24)
    SPACE_400 = dp(32)
    SPACE_500 = dp(40)
    SPACE_600 = dp(48)

    # Colors (Atlassian Design System)
    COLOR_TEXT = get_color_from_hex('#172B4D')
    COLOR_TEXT_SECONDARY = get_color_from_hex('#6B778C')
    COLOR_TEXT_DISABLED = get_color_from_hex('#A5ADBA')
    COLOR_TEXT_INVERSE = get_color_from_hex('#FFFFFF')

    COLOR_SURFACE = get_color_from_hex('#FFFFFF')
    COLOR_SURFACE_OVERLAY = get_color_from_hex('#F4F5F7')
    COLOR_SURFACE_SUNKEN = get_color_from_hex('#DFE1E6')
    COLOR_SURFACE_RAISED = get_color_from_hex('#EBECF0')

    COLOR_BORDER = get_color_from_hex('#DFE1E6')

    COLOR_BACKGROUND_SELECTED = get_color_from_hex('#DEEBFF')
    COLOR_BACKGROUND_ACCENT_GRAY_SUBTLEST = get_color_from_hex('#F8F9FA')

    COLOR_LINK = get_color_from_hex('#0052CC')
    COLOR_LINK_HOVER = get_color_from_hex('#0065FF')
    COLOR_LINK_PRESSED = get_color_from_hex('#0747A6')

    COLOR_ICON = get_color_from_hex('#6B778C')
    COLOR_ICON_ACCENT_GRAY = get_color_from_hex('#5E6C84')
    COLOR_ICON_ACCENT_BLUE = get_color_from_hex('#0052CC')

    # Interaction colors
    COLOR_BACKGROUND_NEUTRAL = [9/255, 30/255, 66/255, 0.08]
    COLOR_BACKGROUND_NEUTRAL_HOVERED = [9/255, 30/255, 66/255, 0.12]
    COLOR_BACKGROUND_NEUTRAL_PRESSED = [9/255, 30/255, 66/255, 0.16]

    COLOR_BACKGROUND_BRAND = get_color_from_hex('#0052CC')
    COLOR_BACKGROUND_BRAND_HOVERED = get_color_from_hex('#0065FF')
    COLOR_BACKGROUND_BRAND_PRESSED = get_color_from_hex('#0747A6')

    COLOR_BACKGROUND_SUCCESS = get_color_from_hex('#36B37E')
    COLOR_BACKGROUND_SUCCESS_HOVERED = get_color_from_hex('#4BCE97')
    COLOR_BACKGROUND_SUCCESS_PRESSED = get_color_from_hex('#00875A')

    COLOR_BACKGROUND_WARNING = get_color_from_hex('#FFAB00')
    COLOR_BACKGROUND_WARNING_HOVERED = get_color_from_hex('#FFC400')
    COLOR_BACKGROUND_WARNING_PRESSED = get_color_from_hex('#FF8B00')

    COLOR_BACKGROUND_DANGER = get_color_from_hex('#FF5630')
    COLOR_BACKGROUND_DANGER_HOVERED = get_color_from_hex('#FF7452')
    COLOR_BACKGROUND_DANGER_PRESSED = get_color_from_hex('#DE350B')

    # Typography
    FONT_FAMILY = 'Roboto'  # Default Kivy font, similar to Atlassian
    FONT_SIZE_075 = dp(11)
    FONT_SIZE_100 = dp(14)
    FONT_SIZE_200 = dp(16)
    FONT_SIZE_300 = dp(18)
    FONT_SIZE_400 = dp(20)
    FONT_SIZE_500 = dp(24)
    FONT_SIZE_600 = dp(29)

    FONT_WEIGHT_NORMAL = 'normal'
    FONT_WEIGHT_MEDIUM = 'normal'  # Kivy limitation
    FONT_WEIGHT_SEMIBOLD = 'normal'  # Kivy limitation
    FONT_WEIGHT_BOLD = 'bold'

    # Border radius
    BORDER_RADIUS = dp(3)
    BORDER_RADIUS_050 = dp(2)
    BORDER_RADIUS_100 = dp(4)
    BORDER_RADIUS_200 = dp(6)
    BORDER_RADIUS_300 = dp(8)
    BORDER_RADIUS_400 = dp(12)

    # Elevation shadows (simplified for Kivy)
    @staticmethod
    def get_shadow_color(elevation=1):
        """Get shadow color based on elevation level"""
        shadows = {
            1: [0, 0, 0, 0.12],  # Raised
            2: [0, 0, 0, 0.16],  # Overlay
            3: [0, 0, 0, 0.20],  # Modal
        }
        return shadows.get(elevation, [0, 0, 0, 0.12])

class TumblrMobileApp(App):
    """Main Kivy application for Tumblr Image Collector Mobile"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collector = None
        self.current_job = None
        self.settings = self.load_settings()

    def build(self):
        """Build the main application"""
        # Set theme colors using Atlassian Design System
        tokens = AtlassianDesignTokens()
        self.theme_colors = {
            'primary': tokens.COLOR_BACKGROUND_BRAND,
            'primary_hovered': tokens.COLOR_BACKGROUND_BRAND_HOVERED,
            'primary_pressed': tokens.COLOR_BACKGROUND_BRAND_PRESSED,
            'secondary': tokens.COLOR_BACKGROUND_NEUTRAL,
            'secondary_hovered': tokens.COLOR_BACKGROUND_NEUTRAL_HOVERED,
            'secondary_pressed': tokens.COLOR_BACKGROUND_NEUTRAL_PRESSED,
            'success': tokens.COLOR_BACKGROUND_SUCCESS,
            'success_hovered': tokens.COLOR_BACKGROUND_SUCCESS_HOVERED,
            'success_pressed': tokens.COLOR_BACKGROUND_SUCCESS_PRESSED,
            'danger': tokens.COLOR_BACKGROUND_DANGER,
            'danger_hovered': tokens.COLOR_BACKGROUND_DANGER_HOVERED,
            'danger_pressed': tokens.COLOR_BACKGROUND_DANGER_PRESSED,
            'warning': tokens.COLOR_BACKGROUND_WARNING,
            'warning_hovered': tokens.COLOR_BACKGROUND_WARNING_HOVERED,
            'warning_pressed': tokens.COLOR_BACKGROUND_WARNING_PRESSED,
            'info': tokens.COLOR_LINK,
            'light': tokens.COLOR_BACKGROUND_ACCENT_GRAY_SUBTLEST,
            'dark': tokens.COLOR_TEXT,
            'surface': tokens.COLOR_SURFACE,
            'surface_overlay': tokens.COLOR_SURFACE_OVERLAY,
            'surface_sunken': tokens.COLOR_SURFACE_SUNKEN,
            'border': tokens.COLOR_BORDER,
            'text': tokens.COLOR_TEXT,
            'text_secondary': tokens.COLOR_TEXT_SECONDARY,
            'text_disabled': tokens.COLOR_TEXT_DISABLED
        }

        # Create screen manager
        self.screen_manager = ScreenManager()

        # Add screens
        self.screen_manager.add_widget(MainScreen(name='main'))
        self.screen_manager.add_widget(ScanScreen(name='scan'))
        self.screen_manager.add_widget(ResultsScreen(name='results'))
        self.screen_manager.add_widget(SettingsScreen(name='settings'))

        return self.screen_manager

    def load_settings(self):
        """Load application settings"""
        settings_file = Path('mobile_settings.json')
        default_settings = {
            'consumer_key': '',
            'consumer_secret': '',
            'oauth_token': '',
            'oauth_token_secret': '',
            'download_path': 'downloads',
            'auto_scan': False,
            'max_downloads': 50,
            'image_quality': 'original'
        }

        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    return {**default_settings, **json.load(f)}
            except Exception as e:
                print(f"Error loading settings: {e}")

        return default_settings

    def save_settings(self):
        """Save application settings"""
        try:
            with open('mobile_settings.json', 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def show_popup(self, title, content, size_hint=(0.8, 0.4)):
        """Show a popup dialog"""
        popup = Popup(
            title=title,
            content=Label(text=content, halign='center'),
            size_hint=size_hint
        )
        popup.open()
        return popup

    def show_error(self, message):
        """Show error message"""
        self.show_popup('Error', message)

    def show_success(self, message):
        """Show success message"""
        self.show_popup('Success', message)

class MainScreen(Screen):
    """Main screen with navigation and quick actions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        """Build the main screen UI with Atlassian Design System"""
        tokens = AtlassianDesignTokens()

        # Main layout with proper spacing
        layout = BoxLayout(
            orientation='vertical',
            padding=tokens.SPACE_400,
            spacing=tokens.SPACE_300
        )

        # Header section (Atlassian card style)
        header_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=0.25,
            padding=tokens.SPACE_300,
            spacing=tokens.SPACE_200
        )

        # Title with Atlassian typography
        title = Label(
            text='Tumblr Image Collector',
            font_size=tokens.FONT_SIZE_500,
            bold=True,
            color=tokens.COLOR_TEXT,
            halign='center',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))

        subtitle = Label(
            text='Mobile Edition',
            font_size=tokens.FONT_SIZE_200,
            color=tokens.COLOR_TEXT_SECONDARY,
            halign='center',
            valign='middle'
        )
        subtitle.bind(size=subtitle.setter('text_size'))

        header_layout.add_widget(title)
        header_layout.add_widget(subtitle)

        # Add header background styling (simulate card)
        header_bg = FloatLayout()
        header_bg.add_widget(header_layout)
        layout.add_widget(header_bg)

        # Quick actions grid (Atlassian button styles)
        actions_grid = GridLayout(
            cols=2,
            size_hint_y=0.6,
            spacing=tokens.SPACE_200,
            padding=[0, tokens.SPACE_100, 0, tokens.SPACE_100]
        )

        # Scan button (Primary style)
        scan_btn = Button(
            text='🔍\nScan Blog',
            font_size=tokens.FONT_SIZE_200,
            bold=True,
            background_color=tokens.COLOR_BACKGROUND_BRAND,
            color=tokens.COLOR_TEXT_INVERSE,
            size_hint_y=None,
            height=tokens.SPACE_600,
            halign='center',
            valign='middle'
        )
        scan_btn.bind(size=scan_btn.setter('text_size'))
        scan_btn.bind(on_press=self.go_to_scan)
        actions_grid.add_widget(scan_btn)

        # Results button (Success style)
        results_btn = Button(
            text='📋\nView Results',
            font_size=tokens.FONT_SIZE_200,
            bold=True,
            background_color=tokens.COLOR_BACKGROUND_SUCCESS,
            color=tokens.COLOR_TEXT_INVERSE,
            size_hint_y=None,
            height=tokens.SPACE_600,
            halign='center',
            valign='middle'
        )
        results_btn.bind(size=results_btn.setter('text_size'))
        results_btn.bind(on_press=self.go_to_results)
        actions_grid.add_widget(results_btn)

        # Settings button (Secondary style)
        settings_btn = Button(
            text='⚙️\nSettings',
            font_size=tokens.FONT_SIZE_200,
            bold=True,
            background_color=tokens.COLOR_SURFACE,
            color=tokens.COLOR_TEXT,
            size_hint_y=None,
            height=tokens.SPACE_600,
            halign='center',
            valign='middle'
        )
        settings_btn.bind(size=settings_btn.setter('text_size'))
        settings_btn.bind(on_press=self.go_to_settings)
        actions_grid.add_widget(settings_btn)

        # About button (Info style)
        about_btn = Button(
            text='ℹ️\nAbout',
            font_size=tokens.FONT_SIZE_200,
            bold=True,
            background_color=tokens.COLOR_LINK,
            color=tokens.COLOR_TEXT_INVERSE,
            size_hint_y=None,
            height=tokens.SPACE_600,
            halign='center',
            valign='middle'
        )
        about_btn.bind(size=about_btn.setter('text_size'))
        about_btn.bind(on_press=self.show_about)
        actions_grid.add_widget(about_btn)

        layout.add_widget(actions_grid)

        # Status area (Atlassian style)
        status_layout = BoxLayout(
            size_hint_y=0.15,
            padding=[tokens.SPACE_200, tokens.SPACE_150]
        )

        status_card = FloatLayout()
        status_bg = Button(
            background_color=tokens.COLOR_SURFACE_OVERLAY,
            disabled=True
        )
        status_card.add_widget(status_bg)

        self.status_label = Label(
            text='Ready to collect Tumblr media',
            font_size=tokens.FONT_SIZE_100,
            color=tokens.COLOR_TEXT_SECONDARY,
            halign='center',
            valign='middle',
            bold=True
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        status_card.add_widget(self.status_label)

        status_layout.add_widget(status_card)
        layout.add_widget(status_layout)

        self.add_widget(layout)

    def go_to_scan(self, instance):
        """Navigate to scan screen"""
        self.manager.current = 'scan'

    def go_to_results(self, instance):
        """Navigate to results screen"""
        self.manager.current = 'results'

    def go_to_settings(self, instance):
        """Navigate to settings screen"""
        self.manager.current = 'settings'

    def show_about(self, instance):
        """Show about dialog"""
        about_text = """
Tumblr Image Collector Mobile v2.1

Features:
• Tumblr blog scanning
• Image and video download
• Batch processing
• Offline viewing

© 2024 Tumblr Collector Team
        """
        popup = Popup(
            title='About',
            content=Label(text=about_text, halign='left'),
            size_hint=(0.8, 0.6)
        )
        popup.open()

class ScanScreen(Screen):
    """Screen for scanning Tumblr blogs"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        """Build the scan screen UI with Atlassian Design System"""
        tokens = AtlassianDesignTokens()

        layout = BoxLayout(
            orientation='vertical',
            padding=tokens.SPACE_400,
            spacing=tokens.SPACE_300
        )

        # Header with back button (Atlassian navigation pattern)
        header = BoxLayout(
            size_hint_y=None,
            height=tokens.SPACE_600,
            spacing=tokens.SPACE_200
        )

        back_btn = Button(
            text='←',
            font_size=tokens.FONT_SIZE_300,
            bold=True,
            background_color=tokens.COLOR_SURFACE,
            color=tokens.COLOR_TEXT,
            size_hint_x=0.2,
            on_press=self.go_back
        )

        title = Label(
            text='Scan Tumblr Blog',
            font_size=tokens.FONT_SIZE_400,
            bold=True,
            color=tokens.COLOR_TEXT,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))

        header.add_widget(back_btn)
        header.add_widget(title)
        layout.add_widget(header)

        # Main content scroll area
        scroll = ScrollView(size_hint=(1, 0.8))
        content_layout = BoxLayout(
            orientation='vertical',
            spacing=tokens.SPACE_400,
            size_hint_y=None
        )
        content_layout.bind(minimum_height=content_layout.setter('height'))

        # Blog input card
        blog_card = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=tokens.SPACE_600,
            padding=tokens.SPACE_300,
            spacing=tokens.SPACE_200
        )

        blog_bg = Button(
            background_color=tokens.COLOR_SURFACE,
            disabled=True
        )
        blog_card.add_widget(blog_bg)

        blog_label = Label(
            text='Blog Name',
            font_size=tokens.FONT_SIZE_100,
            bold=True,
            color=tokens.COLOR_TEXT,
            size_hint_y=None,
            height=tokens.SPACE_200,
            halign='left',
            valign='middle'
        )
        blog_label.bind(size=blog_label.setter('text_size'))

        self.blog_input = TextInput(
            hint_text='e.g., staff, my-favorite-blog',
            multiline=False,
            font_size=tokens.FONT_SIZE_200,
            size_hint_y=None,
            height=tokens.SPACE_400,
            background_color=tokens.COLOR_SURFACE,
            foreground_color=tokens.COLOR_TEXT,
            cursor_color=tokens.COLOR_TEXT,
            hint_text_color=tokens.COLOR_TEXT_DISABLED
        )

        blog_help = Label(
            text='Enter the Tumblr blog name (without .tumblr.com)',
            font_size=tokens.FONT_SIZE_075,
            color=tokens.COLOR_TEXT_SECONDARY,
            size_hint_y=None,
            height=tokens.SPACE_150,
            halign='left',
            valign='middle'
        )
        blog_help.bind(size=blog_help.setter('text_size'))

        blog_card.add_widget(blog_label)
        blog_card.add_widget(self.blog_input)
        blog_card.add_widget(blog_help)
        content_layout.add_widget(blog_card)

        # Options card
        options_card = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            padding=tokens.SPACE_300,
            spacing=tokens.SPACE_300
        )

        options_bg = Button(
            background_color=tokens.COLOR_SURFACE,
            disabled=True
        )
        options_card.add_widget(options_bg)

        # Tags input
        tags_section = BoxLayout(
            orientation='vertical',
            spacing=tokens.SPACE_100,
            size_hint_y=None,
            height=tokens.SPACE_500
        )

        tags_label = Label(
            text='Tags (optional)',
            font_size=tokens.FONT_SIZE_100,
            bold=True,
            color=tokens.COLOR_TEXT,
            halign='left',
            valign='middle'
        )
        tags_label.bind(size=tags_label.setter('text_size'))

        self.tags_input = TextInput(
            hint_text='art, photography, anime',
            multiline=False,
            font_size=tokens.FONT_SIZE_200,
            background_color=tokens.COLOR_SURFACE,
            foreground_color=tokens.COLOR_TEXT,
            cursor_color=tokens.COLOR_TEXT,
            hint_text_color=tokens.COLOR_TEXT_DISABLED
        )

        tags_help = Label(
            text='Comma-separated tags to filter posts',
            font_size=tokens.FONT_SIZE_075,
            color=tokens.COLOR_TEXT_SECONDARY,
            halign='left',
            valign='middle'
        )
        tags_help.bind(size=tags_help.setter('text_size'))

        tags_section.add_widget(tags_label)
        tags_section.add_widget(self.tags_input)
        tags_section.add_widget(tags_help)
        options_card.add_widget(tags_section)

        # Checkboxes
        checkboxes_layout = GridLayout(
            cols=2,
            spacing=tokens.SPACE_200,
            size_hint_y=None,
            height=tokens.SPACE_300
        )

        # Include likes checkbox
        likes_box = BoxLayout(orientation='horizontal', spacing=tokens.SPACE_100)
        self.include_likes_cb = CheckBox(
            size_hint_x=None,
            width=tokens.SPACE_200,
            color=tokens.COLOR_BACKGROUND_BRAND
        )
        likes_label = Label(
            text='Include liked posts',
            font_size=tokens.FONT_SIZE_100,
            color=tokens.COLOR_TEXT,
            halign='left',
            valign='middle'
        )
        likes_label.bind(size=likes_label.setter('text_size'))

        likes_box.add_widget(self.include_likes_cb)
        likes_box.add_widget(likes_label)
        checkboxes_layout.add_widget(likes_box)

        # Videos checkbox
        videos_box = BoxLayout(orientation='horizontal', spacing=tokens.SPACE_100)
        self.videos_cb = CheckBox(
            size_hint_x=None,
            width=tokens.SPACE_200,
            color=tokens.COLOR_BACKGROUND_BRAND
        )
        videos_label = Label(
            text='Download videos',
            font_size=tokens.FONT_SIZE_100,
            color=tokens.COLOR_TEXT,
            halign='left',
            valign='middle'
        )
        videos_label.bind(size=videos_label.setter('text_size'))

        videos_box.add_widget(self.videos_cb)
        videos_box.add_widget(videos_label)
        checkboxes_layout.add_widget(videos_box)

        options_card.add_widget(checkboxes_layout)
        content_layout.add_widget(options_card)

        scroll.add_widget(content_layout)
        layout.add_widget(scroll)

        # Progress area (hidden initially)
        self.progress_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=0.15,
            padding=tokens.SPACE_200,
            spacing=tokens.SPACE_100,
            opacity=0
        )

        progress_bg = Button(
            background_color=tokens.COLOR_SURFACE_OVERLAY,
            disabled=True
        )
        self.progress_layout.add_widget(progress_bg)

        self.progress_label = Label(
            text='Ready to scan',
            font_size=tokens.FONT_SIZE_100,
            color=tokens.COLOR_TEXT_SECONDARY,
            bold=True,
            halign='center',
            valign='middle'
        )
        self.progress_label.bind(size=self.progress_label.setter('text_size'))

        self.progress_bar = ProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=tokens.SPACE_150
        )

        self.progress_layout.add_widget(self.progress_label)
        self.progress_layout.add_widget(self.progress_bar)
        layout.add_widget(self.progress_layout)

        # Scan button (Primary action)
        self.scan_btn = Button(
            text='🔍 Start Scan',
            font_size=tokens.FONT_SIZE_300,
            bold=True,
            background_color=tokens.COLOR_BACKGROUND_BRAND,
            color=tokens.COLOR_TEXT_INVERSE,
            size_hint_y=0.12,
            halign='center',
            valign='middle',
            on_press=self.start_scan
        )
        self.scan_btn.bind(size=self.scan_btn.setter('text_size'))
        layout.add_widget(self.scan_btn)

        self.add_widget(layout)

    def go_back(self, instance):
        """Go back to main screen"""
        self.manager.current = 'main'

    def start_scan(self, instance):
        """Start the scanning process"""
        blog_name = self.blog_input.text.strip()
        if not blog_name:
            self.show_error('Please enter a blog name')
            return

        # Disable scan button
        self.scan_btn.disabled = True
        self.scan_btn.text = 'Scanning...'

        # Show progress
        self.progress_layout.opacity = 1
        self.progress_label.text = 'Initializing scan...'

        # Start scan in thread
        threading.Thread(target=self.perform_scan, args=(blog_name,), daemon=True).start()

    def perform_scan(self, blog_name):
        """Perform the actual scan"""
        try:
            # Update progress
            Clock.schedule_once(lambda dt: self.update_progress(10, 'Connecting to Tumblr...'))

            # Initialize collector if needed
            if not self.manager.parent.collector:
                self.manager.parent.collector = TumblrImageCollector()

            collector = self.manager.parent.collector

            # Get tags
            tags = [tag.strip() for tag in self.tags_input.text.split(',') if tag.strip()]

            # Scan posts
            Clock.schedule_once(lambda dt: self.update_progress(30, f'Scanning {blog_name}...'))

            posts = collector.get_blog_posts(blog_name, limit=20)
            if posts is None:
                raise Exception("Rate limit exceeded or API error")

            Clock.schedule_once(lambda dt: self.update_progress(60, f'Found {len(posts)} posts'))

            # Process posts
            media_items = []
            for i, post in enumerate(posts):
                progress = 60 + (i / len(posts)) * 30
                Clock.schedule_once(lambda dt, p=progress: self.update_progress(p, f'Processing post {i+1}/{len(posts)}'))

                # Extract media from post
                post_media = self.extract_media_from_post(post)
                media_items.extend(post_media)

            # Apply filters
            filtered_items = self.apply_filters(media_items)

            Clock.schedule_once(lambda dt: self.update_progress(100, f'Scan complete! Found {len(filtered_items)} items'))

            # Store results
            self.manager.parent.current_results = {
                'blog_name': blog_name,
                'media_items': filtered_items,
                'total_posts': len(posts)
            }

            # Navigate to results
            Clock.schedule_once(lambda dt: self.go_to_results(), 1)

        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_scan_error(str(e)))

    def extract_media_from_post(self, post):
        """Extract media items from a Tumblr post"""
        media_items = []

        post_type = post.get('type')

        if post_type == 'photo':
            for photo in post.get('photos', []):
                original_size = photo.get('original_size', {})
                if original_size.get('url'):
                    media_items.append({
                        'type': 'image',
                        'url': original_size['url'],
                        'width': original_size.get('width'),
                        'height': original_size.get('height'),
                        'post_data': {
                            'id': post.get('id'),
                            'timestamp': post.get('timestamp'),
                            'tags': post.get('tags', [])
                        }
                    })

        elif post_type == 'video' and self.videos_cb.active:
            video_url = post.get('video_url')
            if video_url:
                media_items.append({
                    'type': 'video',
                    'url': video_url,
                    'thumbnail': post.get('thumbnail_url'),
                    'post_data': {
                        'id': post.get('id'),
                        'timestamp': post.get('timestamp'),
                        'tags': post.get('tags', [])
                    }
                })

        return media_items

    def apply_filters(self, media_items):
        """Apply filters to media items"""
        filtered = media_items

        # Apply tag filters if specified
        tags = [tag.strip() for tag in self.tags_input.text.split(',') if tag.strip()]
        if tags:
            filtered = [
                item for item in filtered
                if any(tag.lower() in [t.lower() for t in item['post_data'].get('tags', [])]
                      for tag in tags)
            ]

        return filtered

    def update_progress(self, value, text):
        """Update progress bar and label"""
        self.progress_bar.value = value
        self.progress_label.text = text

    def show_scan_error(self, error):
        """Show scan error"""
        self.progress_label.text = f'Error: {error}'
        self.progress_label.color = get_color_from_hex('#dc3545')
        self.scan_btn.disabled = False
        self.scan_btn.text = '🔍 Start Scan'

    def go_to_results(self):
        """Navigate to results screen"""
        self.manager.current = 'results'

    def show_error(self, message):
        """Show error message"""
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()

class ResultsScreen(Screen):
    """Screen for displaying scan results"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_items = set()
        self.build_ui()

    def build_ui(self):
        """Build the results screen UI"""
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))

        # Header
        header = BoxLayout(size_hint_y=0.1)
        back_btn = Button(
            text='← Back',
            size_hint_x=0.3,
            on_press=self.go_back
        )
        self.title_label = Label(
            text='Scan Results',
            bold=True,
            font_size=dp(18)
        )
        header.add_widget(back_btn)
        header.add_widget(self.title_label)
        layout.add_widget(header)

        # Summary
        self.summary_label = Label(
            text='No results yet',
            size_hint_y=0.1,
            color=get_color_from_hex('#6c757d')
        )
        layout.add_widget(self.summary_label)

        # Actions
        actions = BoxLayout(size_hint_y=0.1, spacing=dp(10))
        self.select_all_btn = Button(
            text='Select All',
            on_press=self.select_all
        )
        self.download_btn = Button(
            text='📥 Download Selected',
            background_color=get_color_from_hex('#28a745'),
            disabled=True,
            on_press=self.download_selected
        )
        actions.add_widget(self.select_all_btn)
        actions.add_widget(self.download_btn)
        layout.add_widget(actions)

        # Media grid (scrollable)
        scroll = ScrollView(size_hint=(1, 0.7))
        self.media_grid = GridLayout(cols=2, spacing=dp(10), size_hint_y=None)
        self.media_grid.bind(minimum_height=self.media_grid.setter('height'))
        scroll.add_widget(self.media_grid)
        layout.add_widget(scroll)

        self.add_widget(layout)

    def on_enter(self):
        """Called when entering the screen"""
        self.load_results()

    def load_results(self):
        """Load and display scan results"""
        results = getattr(self.manager.parent, 'current_results', None)
        if not results:
            self.summary_label.text = 'No scan results available'
            return

        self.title_label.text = f'Results for {results["blog_name"]}'
        self.summary_label.text = f'Found {len(results["media_items"])} items from {results["total_posts"]} posts'

        # Clear previous items
        self.media_grid.clear_widgets()
        self.selected_items.clear()

        # Add media items
        for item in results['media_items']:
            media_item = self.create_media_item(item)
            self.media_grid.add_widget(media_item)

        self.update_download_button()

    def create_media_item(self, item):
        """Create a media item widget"""
        layout = FloatLayout(size_hint_y=None, height=dp(120))

        # Background
        bg = Button(
            background_color=get_color_from_hex('#f8f9fa'),
            pos_hint={'x': 0, 'y': 0},
            size_hint=(1, 1)
        )
        bg.bind(on_press=lambda x: self.toggle_selection(item['url']))
        layout.add_widget(bg)

        # Thumbnail (placeholder)
        thumbnail = Button(
            text=self.get_type_emoji(item['type']),
            font_size=dp(24),
            background_color=get_color_from_hex('#dee2e6'),
            pos_hint={'center_x': 0.5, 'center_y': 0.6},
            size_hint=(0.6, 0.5)
        )
        layout.add_widget(thumbnail)

        # Type label
        type_label = Label(
            text=item['type'].upper(),
            font_size=dp(10),
            color=get_color_from_hex('#6c757d'),
            pos_hint={'center_x': 0.5, 'y': 0.1},
            size_hint=(0.8, 0.1)
        )
        layout.add_widget(type_label)

        # Selection indicator
        self.selection_indicator = Label(
            text='',
            font_size=dp(16),
            color=get_color_from_hex('#28a745'),
            pos_hint={'right': 1, 'top': 1},
            size_hint=(0.2, 0.2)
        )
        layout.add_widget(self.selection_indicator)

        # Store reference for selection updates
        layout.item_url = item['url']
        layout.selection_indicator = self.selection_indicator

        return layout

    def get_type_emoji(self, item_type):
        """Get emoji for media type"""
        return {'image': '🖼️', 'video': '🎥', 'gif': '🎞️'}.get(item_type, '📄')

    def toggle_selection(self, item_url):
        """Toggle selection of an item"""
        if item_url in self.selected_items:
            self.selected_items.remove(item_url)
        else:
            self.selected_items.add(item_url)

        self.update_selections()
        self.update_download_button()

    def select_all(self, instance):
        """Select or deselect all items"""
        results = getattr(self.manager.parent, 'current_results', None)
        if not results:
            return

        if len(self.selected_items) == len(results['media_items']):
            # Deselect all
            self.selected_items.clear()
        else:
            # Select all
            self.selected_items.clear()
            for item in results['media_items']:
                self.selected_items.add(item['url'])

        self.update_selections()
        self.update_download_button()

    def update_selections(self):
        """Update visual selection indicators"""
        for child in self.media_grid.children:
            if hasattr(child, 'item_url'):
                if child.item_url in self.selected_items:
                    child.selection_indicator.text = '✓'
                    child.children[0].background_color = get_color_from_hex('#d4edda')  # Light green
                else:
                    child.selection_indicator.text = ''
                    child.children[0].background_color = get_color_from_hex('#f8f9fa')  # Light gray

    def update_download_button(self):
        """Update download button state"""
        has_selection = len(self.selected_items) > 0
        self.download_btn.disabled = not has_selection

        if has_selection:
            self.download_btn.text = f'📥 Download ({len(self.selected_items)})'
        else:
            self.download_btn.text = '📥 Download Selected'

    def download_selected(self, instance):
        """Download selected items"""
        if not self.selected_items:
            return

        # Show download progress
        popup = Popup(
            title='Downloading',
            content=Label(text=f'Downloading {len(self.selected_items)} items...\nThis feature is under development.'),
            size_hint=(0.8, 0.4)
        )
        popup.open()

    def go_back(self, instance):
        """Go back to scan screen"""
        self.manager.current = 'scan'

class SettingsScreen(Screen):
    """Screen for application settings"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        """Build the settings screen UI"""
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))

        # Header
        header = BoxLayout(size_hint_y=0.1)
        back_btn = Button(
            text='← Back',
            size_hint_x=0.3,
            on_press=self.go_back
        )
        title = Label(
            text='Settings',
            bold=True,
            font_size=dp(18)
        )
        header.add_widget(back_btn)
        header.add_widget(title)
        layout.add_widget(header)

        # Settings content (scrollable)
        scroll = ScrollView(size_hint=(1, 0.8))
        settings_layout = BoxLayout(orientation='vertical', spacing=dp(20), size_hint_y=None)
        settings_layout.bind(minimum_height=settings_layout.setter('height'))

        # API Settings
        api_section = BoxLayout(orientation='vertical', spacing=dp(10))
        api_section.add_widget(Label(text='Tumblr API Settings', bold=True, font_size=dp(16)))

        self.consumer_key_input = TextInput(hint_text='Consumer Key')
        self.consumer_secret_input = TextInput(hint_text='Consumer Secret', password=True)
        self.oauth_token_input = TextInput(hint_text='OAuth Token')
        self.oauth_token_secret_input = TextInput(hint_text='OAuth Token Secret', password=True)

        api_section.add_widget(self.consumer_key_input)
        api_section.add_widget(self.consumer_secret_input)
        api_section.add_widget(self.oauth_token_input)
        api_section.add_widget(self.oauth_token_secret_input)

        settings_layout.add_widget(api_section)

        # Download Settings
        download_section = BoxLayout(orientation='vertical', spacing=dp(10))
        download_section.add_widget(Label(text='Download Settings', bold=True, font_size=dp(16)))

        self.download_path_input = TextInput(hint_text='Download Path')
        self.max_downloads_input = TextInput(hint_text='Max Downloads', input_filter='int')

        download_section.add_widget(self.download_path_input)
        download_section.add_widget(self.max_downloads_input)

        settings_layout.add_widget(download_section)

        scroll.add_widget(settings_layout)
        layout.add_widget(scroll)

        # Save button
        save_btn = Button(
            text='💾 Save Settings',
            font_size=dp(18),
            background_color=get_color_from_hex('#28a745'),
            size_hint_y=0.1,
            on_press=self.save_settings
        )
        layout.add_widget(save_btn)

        self.add_widget(layout)

        # Load current settings
        self.load_settings()

    def load_settings(self):
        """Load current settings into form"""
        settings = self.manager.parent.settings

        self.consumer_key_input.text = settings.get('consumer_key', '')
        self.consumer_secret_input.text = settings.get('consumer_secret', '')
        self.oauth_token_input.text = settings.get('oauth_token', '')
        self.oauth_token_secret_input.text = settings.get('oauth_token_secret', '')
        self.download_path_input.text = settings.get('download_path', 'downloads')
        self.max_downloads_input.text = str(settings.get('max_downloads', 50))

    def save_settings(self, instance):
        """Save settings from form"""
        settings = {
            'consumer_key': self.consumer_key_input.text.strip(),
            'consumer_secret': self.consumer_secret_input.text.strip(),
            'oauth_token': self.oauth_token_input.text.strip(),
            'oauth_token_secret': self.oauth_token_secret_input.text.strip(),
            'download_path': self.download_path_input.text.strip(),
            'max_downloads': int(self.max_downloads_input.text or 50)
        }

        self.manager.parent.settings = settings
        self.manager.parent.save_settings()

        # Show success message
        popup = Popup(
            title='Success',
            content=Label(text='Settings saved successfully!'),
            size_hint=(0.8, 0.4)
        )
        popup.open()

    def go_back(self, instance):
        """Go back to main screen"""
        self.manager.current = 'main'

if __name__ == '__main__':
    TumblrMobileApp().run()
