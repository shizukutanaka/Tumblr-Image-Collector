# Tumblr Image Collector

Production-grade Tumblr image collection tool with enterprise security and personal edition features.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

### Core Functionality
- Official Tumblr API integration with rate limiting
- Parallel downloads with configurable workers (up to 20 concurrent)
- Automatic resume of interrupted downloads
- Smart filtering by resolution, file size, tags, date range
- Perceptual hash-based duplicate detection (O(1) average time)
- Image optimization with format conversion
- Multi-tier memory and disk caching

### Security
- AES-256 encryption for credentials
- System keyring integration (Windows/macOS/Linux)
- SSRF protection with private IP blocking
- Input validation and ReDoS prevention
- Rate limiting with token bucket algorithm
- DDoS mitigation and automatic IP blocking
- Complete audit logging

### Personal Edition Features
- Auto-performance tuning based on system resources
- Adaptive worker scaling
- Favorite blog management
- Scheduled downloads (daily/weekly/monthly)
- Auto-organization by date and tags
- SQLite library with advanced search
- Automatic backups
- Thumbnail generation
- Wallpaper collection extraction
- Privacy mode with log sanitization

## Quick Start

### Installation

```bash
git clone <repository-url>
cd "tumblr image collector"
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Configuration

#### Option 1: Interactive Wizard (Recommended)
```bash
python config.py
```

#### Option 2: Environment Variables
```bash
export TUMBLR_CONSUMER_KEY="your_consumer_key"
export TUMBLR_CONSUMER_SECRET="your_consumer_secret"
```

#### Option 3: Manual Configuration
Create `config.json`:
```json
{
  "consumer_key": "your_consumer_key",
  "consumer_secret": "your_consumer_secret",
  "output_folder_name": "tumblr_images",
  "max_download_workers": 5
}
```

### Basic Usage

```bash
# Download from a blog
python tumblr_image_collector.py blog_name

# Filter by tags
python tumblr_image_collector.py blog_name --tags photo art

# Date range filter
python tumblr_image_collector.py blog_name --start-date 2024-01-01 --end-date 2024-12-31

# Download liked posts
python tumblr_image_collector.py --include-likes

# Custom output directory
python tumblr_image_collector.py blog_name --output ./my_images

# Adjust worker count
python tumblr_image_collector.py blog_name --workers 10

# Interactive mode
python tumblr_image_collector.py --interactive
```

### Personal Edition Usage

```python
from personal_features import get_personal_manager
from personal_security import get_security_manager
from personal_optimizer import get_optimizer
from personal_convenience import get_convenience_features
import json

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Security manager - encrypt credentials
security = get_security_manager("./tumblr_images", config)
security.encrypt_credentials("your_key", "your_secret")
credentials = security.decrypt_credentials()

# Auto-optimize performance
optimizer = get_optimizer(config)
optimizer.auto_tune()

# Convenience features
convenience = get_convenience_features("./tumblr_images", config)

# Add favorite blog
convenience.add_favorite(
    blog_name="my-favorite-blog",
    tags=["art", "illustration"],
    auto_download=True
)

# Schedule daily download at 3 AM
convenience.schedule_download(
    blog_name="my-favorite-blog",
    schedule_type="daily",
    time="03:00"
)

# Library management
manager = get_personal_manager("./tumblr_images", config)
stats = manager.get_statistics()
print(f"Total images: {stats['total_images']}")
```

## System Requirements

- Python 3.8 or higher (3.10+ recommended)
- 4GB RAM minimum (8GB+ for heavy workloads)
- 20GB free disk space minimum

## Core Dependencies

```
pytumblr>=0.1.2       # Tumblr API client
requests>=2.32.3      # HTTP library
Pillow>=10.4.0        # Image processing
imagehash>=4.3.1      # Perceptual hashing
psutil>=5.9.8         # System monitoring
cryptography>=42.0.0  # AES-256 encryption
keyring>=25.0.0       # System keyring
```

## Architecture

### Core Modules
- `tumblr_image_collector.py` - Main application
- `config.py` - Configuration wizard
- `image_classifier.py` - Image analysis and NSFW detection
- `url_validator.py` - URL validation
- `download_manager.py` - Download orchestration
- `cache_manager.py` - Multi-tier caching
- `image_optimizer.py` - Image processing

### Production Modules
- `production_url_manager.py` - URL security and lifecycle
- `production_security.py` - Security hardening
- `production_error_handler.py` - Error handling
- `production_monitoring.py` - Metrics and health checks

### Personal Edition Modules
- `personal_features.py` - Library management
- `personal_security.py` - Encryption and privacy
- `personal_optimizer.py` - Performance optimization
- `personal_convenience.py` - User convenience features

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific tests
pytest test_image_classifier.py -v
pytest test_production_systems.py -v

# Code quality
flake8 .
```

## Performance

- Download Speed: Up to 20 concurrent workers
- Duplicate Detection: O(1) average time
- Cache Performance: <1ms (memory), <10ms (disk)
- Memory Usage: ~100MB baseline + 10-20MB per worker

## Security Best Practices

### Credential Protection
```bash
# Use environment variables (recommended)
export TUMBLR_CONSUMER_KEY="your_key"
export TUMBLR_CONSUMER_SECRET="your_secret"

# Or use encryption
python -c "from personal_security import get_security_manager; \
           import json; \
           config = json.load(open('config.json')); \
           security = get_security_manager('.', config); \
           security.encrypt_credentials('key', 'secret')"
```

### Privacy Mode
```json
{
  "security": {
    "enable_privacy_mode": true,
    "clear_logs_after_days": 30,
    "secure_delete": true
  }
}
```

## Folder Structure

```
tumblr_images/
├── images/              # Original images by blog
├── by_date/            # Auto-organized by date
├── by_tags/            # Auto-organized by tags
├── duplicates/         # Duplicate images
├── thumbnails/         # Auto-generated thumbnails
├── backups/            # Automatic backups
├── wallpapers/         # High-resolution collection
├── favorites/          # Favorite images
├── .security/          # Encrypted credentials
├── personal_library.db # SQLite database
├── favorites.json      # Favorite blogs
└── schedule.json       # Download schedules
```

## Troubleshooting

### Rate Limiting
Reduce worker count or increase delay between requests:
```bash
python tumblr_image_collector.py blog_name --workers 3
```

### Memory Issues
```python
optimizer = get_optimizer(config)
optimizer.cleanup_memory()
```

### Circuit Breaker Open
Wait for recovery timeout (default: 60 seconds) or check service health.

### View Logs
```bash
tail -f tumblr_collector.log | grep ERROR
```

## Documentation

- [PERSONAL_USER_GUIDE.md](PERSONAL_USER_GUIDE.md) - Complete personal edition guide
- [README_PERSONAL.md](README_PERSONAL.md) - Personal edition overview
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Development guide
- [SECURITY_IMPROVEMENTS.md](SECURITY_IMPROVEMENTS.md) - Security details
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Detailed installation
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [ROADMAP.md](ROADMAP.md) - Future plans

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests before committing
pytest --cov
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: Project README and guides
- Issues: For bug reports and feature requests
- Community: For discussions and questions
