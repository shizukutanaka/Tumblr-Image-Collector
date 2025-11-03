# Quick Start Guide - 2024-2025 Improvements

## Installation

```bash
# Install or update dependencies
pip install -r requirements.txt

# Verify installation
python -c "import httpx; import structlog; import pydantic; print('✓ All dependencies installed')"
```

## 5-Minute Setup

### 1. Initialize Logging

```python
from advanced_logging import initialize_logging, get_logger

# One-time setup
initialize_logging(
    log_dir="logs",
    log_level="INFO",
    json_output=True,
)

# Use in your code
logger = get_logger(__name__)
logger.info("app_started", version="2.0")
```

### 2. Create Configuration

**Option A: Create config.json programmatically**

```python
from pydantic_config import TumblrCollectorConfig, TumblrConfig

config = TumblrCollectorConfig(
    tumblr=TumblrConfig(
        consumer_key="your_key",
        consumer_secret="your_secret",
        token="your_token",
        token_secret="your_token_secret",
    ),
    download={
        "max_workers": 10,
        "output_folder": "./downloads",
    },
    filters={
        "min_resolution": [1920, 1080],
        "filter_mode": "safe",
    },
)

# Save for next time
config.to_file("config.json")

# Print summary
print(config.get_summary())
```

**Option B: Load existing config**

```python
from pydantic_config import TumblrCollectorConfig

# From file
config = TumblrCollectorConfig.from_file("config.json")

# From environment variables
config = TumblrCollectorConfig.from_env()
```

### 3. Setup Caching

```python
from advanced_cache import initialize_cache

# Initialize once
cache = initialize_cache(
    cache_dir=".cache",
    max_cache_size_gb=5.0,
    ttl_hours=24,
)

# Use in your downloads
blog_data = cache.get("blog:my-blog")
if blog_data is None:
    # Fetch from API
    blog_data = fetch_blog_info("my-blog")
    cache.set("blog:my-blog", blog_data, namespace="api")
```

### 4. Use Advanced HTTP Client

```python
import asyncio
from advanced_http_client import AsyncHTTPClient, ClientConfig

async def download_tumblr_content():
    config = ClientConfig(
        max_workers=10,
        rate_limit_calls=1000,
    )

    async with AsyncHTTPClient(config) as client:
        # Download with automatic retry and rate limiting
        response = await client.get("https://api.tumblr.com/v2/blog/info")

        # Download file with progress
        await client.download_file(
            "https://example.com/image.jpg",
            "./downloads/image.jpg",
        )

# Run async code
asyncio.run(download_tumblr_content())
```

### 5. Remove Duplicates

```python
from advanced_deduplication import ImageDeduplicator
from pathlib import Path

deduplicator = ImageDeduplicator(method="hybrid")

# Find all images
images = list(Path("downloads").glob("**/*.jpg"))

# Find duplicates
duplicates = deduplicator.find_duplicates(images)

# Show report
report = deduplicator.get_duplicate_report(duplicates)
print(f"Found {report['total_duplicates']} duplicates")
print(f"Can free {report['total_wasted_mb']:.1f}MB")

# Remove duplicates
stats = deduplicator.remove_duplicates(duplicates, keep_originals=True)
print(f"Removed {stats['removed']} files")
```

## Common Tasks

### Monitor Download Progress

```python
from advanced_logging import get_performance_logger

perf_logger = get_performance_logger()

# Log download statistics
perf_logger.log_download_stats(
    total_files=1000,
    total_bytes=5_000_000_000,  # 5GB
    duration_seconds=3600,
    success_count=980,
    failure_count=20,
)
```

### Check Cache Performance

```python
from advanced_cache import get_cache

cache = get_cache()

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"Cached items: {stats['total_entries']}")
print(f"Cache size: {stats['total_size_mb']:.1f}MB")

# Cleanup expired items
removed = cache.cleanup_expired()
print(f"Removed {removed} expired items")
```

### Handle API Errors Gracefully

```python
from advanced_http_client import AsyncHTTPClient
import asyncio

async def robust_api_call(url, max_retries=3):
    client = AsyncHTTPClient()

    try:
        # Automatic retry on transient failures
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except RuntimeError as e:
        if "Circuit breaker" in str(e):
            print("API temporarily unavailable, please retry later")
        raise
    finally:
        await client.close()

# Use it
try:
    data = asyncio.run(robust_api_call("https://api.tumblr.com/v2/..."))
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Cache Function Results

```python
from advanced_cache import cache_function

@cache_function(namespace="api", ttl_hours=1)
def get_blog_posts(blog_name):
    # This function's result is automatically cached
    return api.get_posts(blog_name)

# First call: slow (calls API)
posts1 = get_blog_posts("my-blog")

# Second call: fast (from cache)
posts2 = get_blog_posts("my-blog")
```

## Debugging

### Enable Debug Logging

```python
from advanced_logging import initialize_logging

initialize_logging(
    log_level="DEBUG",  # Show all messages
    json_output=False,  # Human-readable format
)
```

### Check Configuration Validity

```python
from pydantic_config import validate_config_file

is_valid, message = validate_config_file("config.json")
if not is_valid:
    print(f"Config error: {message}")
else:
    print("Configuration is valid!")
```

### Clear Cache

```python
from advanced_cache import get_cache

cache = get_cache()

# Clear all cache
cache.clear()

# Clear specific namespace
cache.clear(namespace="api")
```

### View Logs

```bash
# View recent logs
tail -f logs/tumblr_collector.log

# View errors only
grep ERROR logs/tumblr_collector.log

# View as JSON (for analysis)
cat logs/tumblr_collector.log | jq
```

## Performance Tips

### 1. Optimize Caching

```python
from advanced_cache import initialize_cache

# Use larger cache for fewer API calls
cache = initialize_cache(
    max_cache_size_gb=10.0,  # 10GB
    ttl_hours=48,            # 48 hour TTL
)
```

### 2. Increase Download Concurrency

```python
from advanced_http_client import AsyncHTTPClient, ClientConfig

# Higher concurrency = faster downloads
config = ClientConfig(
    max_connections=200,  # Up from default 100
    max_keepalive_connections=50,
)
```

### 3. Use Batch Operations

```python
from advanced_http_client import AsyncHTTPClient

async def batch_download():
    client = AsyncHTTPClient()

    # Download multiple files concurrently
    requests = [
        ("GET", f"https://api.tumblr.com/v2/blog/{i}", {})
        for i in range(1, 11)
    ]

    responses = await client.batch_requests(
        requests,
        max_concurrent=20,  # Control concurrency
    )
```

### 4. Efficient Duplicate Detection

```python
from advanced_deduplication import ImageDeduplicator

deduplicator = ImageDeduplicator(
    method="exact",  # Fastest for initial pass
)

# First pass: find exact duplicates quickly
exact_dupes = deduplicator.find_duplicates(images, method="exact")

# Second pass: find near-duplicates (if needed)
similar_dupes = deduplicator.find_duplicates(images, method="perceptual")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Slow downloads** | Increase `max_connections` in ClientConfig |
| **High memory usage** | Reduce `max_workers` or `max_connections` |
| **Cache not working** | Check cache directory exists: `ls .cache` |
| **Configuration errors** | Run: `python -m pydantic_config` |
| **No deduplicates found** | Lower `similarity_threshold` or use CNN method |
| **Circuit breaker open** | Wait 60 seconds, check API status |

## Next Steps

1. **Read full documentation**: `IMPROVEMENTS_2025.md`
2. **Run tests**: `pytest tests/test_improvements_2025.py -v`
3. **Monitor metrics**: Use `PerformanceLogger` for observability
4. **Customize**: Adjust config, cache size, rate limits for your needs

## Example Complete Script

```python
#!/usr/bin/env python
"""Complete example using all 2025 improvements."""

import asyncio
from pathlib import Path
from advanced_logging import initialize_logging, get_logger
from advanced_cache import initialize_cache, get_cache
from advanced_http_client import AsyncHTTPClient, ClientConfig
from advanced_deduplication import ImageDeduplicator
from pydantic_config import TumblrCollectorConfig

async def main():
    # Setup
    initialize_logging(log_level="INFO")
    logger = get_logger(__name__)
    cache = initialize_cache()

    logger.info("app_started", message="Starting Tumblr Image Collector 2.0")

    # Load config
    try:
        config = TumblrCollectorConfig.from_file("config.json")
    except FileNotFoundError:
        logger.error("config_missing", file="config.json")
        return

    # Download with async HTTP client
    http_config = ClientConfig(
        max_connections=config.download.max_workers,
        rate_limit_calls=config.tumblr.rate_limit_calls,
    )

    async with AsyncHTTPClient(http_config) as client:
        # Download images (with retry and rate limiting)
        logger.info("download_starting", blog="my-blog")

        # Simulate downloading
        # response = await client.get("https://api.tumblr.com/v2/blog/info")

    # Check cache stats
    cache_stats = cache.get_stats()
    logger.info(
        "cache_stats",
        hit_rate=cache_stats["hit_rate_percent"],
        size_mb=cache_stats["total_size_mb"],
    )

    # Find and remove duplicates
    images = list(Path(config.download.output_folder).glob("**/*.jpg"))
    if images:
        logger.info("dedup_starting", image_count=len(images))

        deduplicator = ImageDeduplicator()
        duplicates = deduplicator.find_duplicates(images)

        if duplicates:
            stats = deduplicator.remove_duplicates(duplicates)
            logger.info(
                "dedup_complete",
                removed=stats["removed"],
                freed_mb=stats["freed_bytes"] / (1024 * 1024),
            )

    logger.info("app_complete", message="Done!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Support

- **Documentation**: See `IMPROVEMENTS_2025.md`
- **Tests**: Run `pytest tests/test_improvements_2025.py`
- **Logs**: Check `logs/` directory for detailed information
- **Errors**: View `logs/errors.log` for error details

Happy collecting! 🚀
