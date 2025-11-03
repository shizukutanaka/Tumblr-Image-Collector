# Tumblr Image Collector Mobile App

A cross-platform mobile application built with Kivy for collecting images and videos from Tumblr blogs.

## Features

- **Tumblr Blog Scanning**: Scan any public Tumblr blog for media content
- **Multi-Media Support**: Download images, videos, and GIFs
- **Tag Filtering**: Filter posts by specific tags
- **Batch Processing**: Download multiple items simultaneously
- **Offline Viewing**: Store media locally for offline access
- **Cross-Platform**: Works on Android, iOS, Windows, macOS, and Linux

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
cd mobile_app
pip install -r requirements.txt
```

### For Mobile Deployment

#### Android (using Buildozer)

1. Install Buildozer:
```bash
pip install buildozer
```

2. Initialize buildozer:
```bash
buildozer init
```

3. Build APK:
```bash
buildozer android debug
```

#### iOS (using kivy-ios)

1. Install kivy-ios:
```bash
pip install kivy-ios
```

2. Build for iOS:
```bash
python -m kivy.tools.build_ios
```

## Usage

### Desktop Testing

```bash
python main.py
```

### Mobile App

1. **Setup API Credentials**:
   - Go to Settings screen
   - Enter your Tumblr API credentials (Consumer Key, Consumer Secret, OAuth tokens)

2. **Scan a Blog**:
   - Go to Scan screen
   - Enter blog name (without .tumblr.com)
   - Optionally add tags to filter
   - Toggle video downloads if desired
   - Tap "Start Scan"

3. **View Results**:
   - Review found media items
   - Select items to download
   - Tap "Download Selected"

4. **Access Downloads**:
   - Media is saved to the configured download path
   - View downloaded content in your device's gallery

## App Structure

```
mobile_app/
├── main.py              # Main Kivy application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── buildozer.spec      # Buildozer configuration (for Android builds)
```

## Configuration

### API Settings

- **Consumer Key**: Your Tumblr app consumer key
- **Consumer Secret**: Your Tumblr app consumer secret
- **OAuth Token**: OAuth access token
- **OAuth Token Secret**: OAuth access token secret

### Download Settings

- **Download Path**: Local directory for saving media
- **Max Downloads**: Maximum concurrent downloads

## Permissions

The app requires the following permissions:

- **Internet**: To access Tumblr API and download media
- **Storage**: To save downloaded media to device storage
- **Network State**: To check internet connectivity

## Troubleshooting

### Common Issues

1. **"No module named 'kivy'"**:
   - Install Kivy: `pip install kivy`

2. **Tumblr API errors**:
   - Verify your API credentials in Settings
   - Check your internet connection
   - Ensure the blog is public

3. **Downloads failing**:
   - Check available storage space
   - Verify download path permissions
   - Try downloading fewer items at once

4. **App crashes on mobile**:
   - Ensure all dependencies are installed
   - Check device compatibility
   - Try running on a different device

### Debug Mode

Enable debug logging by setting the environment variable:

```bash
export KIVY_LOG_LEVEL=debug
python main.py
```

## Development

### Adding New Features

1. Create new screens by extending the `Screen` class
2. Add screen to the `ScreenManager` in `TumblrMobileApp.build()`
3. Implement UI logic in the screen's methods

### Testing

Run the app in desktop mode for testing:

```bash
python main.py
```

Use the window controls to simulate mobile interactions.

### Building for Production

#### Android APK

```bash
buildozer android release
```

#### iOS App

```bash
python -m kivy.tools.build_ios
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple platforms
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the main project README for API details
