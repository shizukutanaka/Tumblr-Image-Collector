// Tumblr Image Collector Content Script
class TumblrMediaScanner {
    constructor() {
        this.mediaItems = [];
        this.setupMessageListener();
    }

    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'scanMedia') {
                this.scanMedia(request.filters)
                    .then(result => sendResponse(result))
                    .catch(error => sendResponse({ success: false, error: error.message }));
                return true; // Keep the message channel open for async response
            }
        });
    }

    async scanMedia(filters) {
        try {
            this.mediaItems = [];

            // Scan for different types of media
            if (filters.images) {
                await this.scanImages(filters);
            }

            if (filters.videos) {
                await this.scanVideos(filters);
            }

            if (filters.gifs) {
                await this.scanGifs(filters);
            }

            // Apply additional filters
            this.applyFilters(filters);

            return {
                success: true,
                media: this.mediaItems,
                count: this.mediaItems.length
            };

        } catch (error) {
            console.error('Tumblr Media Scanner error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async scanImages(filters) {
        // Find all images on the page
        const images = document.querySelectorAll('img[src]');

        for (const img of images) {
            try {
                const src = img.src || img.dataset.src;

                if (!src || !this.isTumblrMediaUrl(src)) {
                    continue;
                }

                // Get image dimensions
                const dimensions = await this.getImageDimensions(img);

                // Skip if doesn't meet minimum size requirements
                if (dimensions.width < filters.minWidth || dimensions.height < filters.minHeight) {
                    continue;
                }

                const mediaItem = {
                    type: 'image',
                    url: this.getFullResolutionUrl(src),
                    thumbnail: src,
                    width: dimensions.width,
                    height: dimensions.height,
                    alt: img.alt || '',
                    title: img.title || ''
                };

                this.mediaItems.push(mediaItem);

            } catch (error) {
                console.warn('Error processing image:', img.src, error);
            }
        }
    }

    async scanVideos(filters) {
        // Find video elements
        const videos = document.querySelectorAll('video[src], video source');

        for (const video of videos) {
            try {
                const src = video.src || video.currentSrc;

                if (!src || !this.isTumblrMediaUrl(src)) {
                    continue;
                }

                const mediaItem = {
                    type: 'video',
                    url: src,
                    thumbnail: this.extractVideoThumbnail(video),
                    width: video.videoWidth || video.width,
                    height: video.videoHeight || video.height
                };

                this.mediaItems.push(mediaItem);

            } catch (error) {
                console.warn('Error processing video:', error);
            }
        }

        // Also check for embedded video players
        await this.scanEmbeddedVideos(filters);
    }

    async scanEmbeddedVideos(filters) {
        // Look for Tumblr's embedded video iframes
        const iframes = document.querySelectorAll('iframe[src*="tumblr.com/video"]');

        for (const iframe of iframes) {
            try {
                const src = iframe.src;

                if (!src) continue;

                // Extract video URL from iframe src
                const videoUrl = this.extractVideoUrlFromEmbed(src);

                if (videoUrl && this.isTumblrMediaUrl(videoUrl)) {
                    const mediaItem = {
                        type: 'video',
                        url: videoUrl,
                        thumbnail: this.extractThumbnailFromEmbed(iframe),
                        width: iframe.width || 400,
                        height: iframe.height || 225
                    };

                    this.mediaItems.push(mediaItem);
                }

            } catch (error) {
                console.warn('Error processing embedded video:', error);
            }
        }
    }

    async scanGifs(filters) {
        // GIFs are often served as images but with .gif extension
        const gifImages = document.querySelectorAll('img[src*=".gif"]');

        for (const img of gifImages) {
            try {
                const src = img.src || img.dataset.src;

                if (!src || !src.includes('.gif') || !this.isTumblrMediaUrl(src)) {
                    continue;
                }

                const dimensions = await this.getImageDimensions(img);

                if (dimensions.width < filters.minWidth || dimensions.height < filters.minHeight) {
                    continue;
                }

                const mediaItem = {
                    type: 'gif',
                    url: this.getFullResolutionUrl(src),
                    thumbnail: src,
                    width: dimensions.width,
                    height: dimensions.height,
                    alt: img.alt || '',
                    title: img.title || ''
                };

                this.mediaItems.push(mediaItem);

            } catch (error) {
                console.warn('Error processing GIF:', error);
            }
        }
    }

    applyFilters(filters) {
        // Remove duplicates if requested
        if (filters.skipDuplicates) {
            const seen = new Set();
            this.mediaItems = this.mediaItems.filter(item => {
                if (seen.has(item.url)) {
                    return false;
                }
                seen.add(item.url);
                return true;
            });
        }

        // Apply size filters (additional check)
        this.mediaItems = this.mediaItems.filter(item => {
            return (!filters.minWidth || item.width >= filters.minWidth) &&
                   (!filters.minHeight || item.height >= filters.minHeight);
        });
    }

    isTumblrMediaUrl(url) {
        if (!url) return false;

        try {
            const urlObj = new URL(url);

            // Check for Tumblr domains
            if (urlObj.hostname.includes('tumblr.com') ||
                urlObj.hostname.includes('media.tumblr.com') ||
                urlObj.hostname.includes('data.tumblr.com')) {
                return true;
            }

            // Check for common Tumblr media patterns
            if (url.includes('media.tumblr.com') ||
                url.includes('data.tumblr.com') ||
                url.includes('/tumblr_')) {
                return true;
            }

            return false;

        } catch (error) {
            return false;
        }
    }

    getFullResolutionUrl(url) {
        if (!url) return url;

        try {
            // Tumblr often serves smaller versions, try to get the original
            // Remove size suffixes like _500, _250, etc.
            const cleanUrl = url.replace(/_\d+(?=\.[a-zA-Z]+$)/, '');

            // If it's already a full resolution URL, return as-is
            if (!cleanUrl.includes('_') || cleanUrl.includes('original')) {
                return cleanUrl;
            }

            return cleanUrl;

        } catch (error) {
            return url;
        }
    }

    async getImageDimensions(img) {
        return new Promise((resolve) => {
            if (img.naturalWidth && img.naturalHeight) {
                resolve({
                    width: img.naturalWidth,
                    height: img.naturalHeight
                });
            } else {
                img.addEventListener('load', () => {
                    resolve({
                        width: img.naturalWidth,
                        height: img.naturalHeight
                    });
                });

                img.addEventListener('error', () => {
                    resolve({
                        width: img.width || 0,
                        height: img.height || 0
                    });
                });

                // Timeout fallback
                setTimeout(() => {
                    resolve({
                        width: img.width || 0,
                        height: img.height || 0
                    });
                }, 3000);
            }
        });
    }

    extractVideoThumbnail(video) {
        // Try to get poster attribute first
        if (video.poster) {
            return video.poster;
        }

        // Try to find thumbnail in nearby elements
        const container = video.closest('.video-container, .media, article');
        if (container) {
            const thumbnail = container.querySelector('img[alt*="thumbnail"], img[alt*="preview"]');
            if (thumbnail) {
                return thumbnail.src;
            }
        }

        return null;
    }

    extractVideoUrlFromEmbed(embedUrl) {
        try {
            // Tumblr embed URLs often contain the actual video URL as parameters
            const url = new URL(embedUrl);

            // Look for video URL in parameters or path
            if (url.searchParams.has('video_url')) {
                return url.searchParams.get('video_url');
            }

            // Try to construct video URL from embed parameters
            const videoId = url.pathname.split('/').pop();
            if (videoId) {
                return `https://vt.tumblr.com/tumblr_${videoId}.mp4`;
            }

        } catch (error) {
            console.warn('Failed to extract video URL from embed:', error);
        }

        return null;
    }

    extractThumbnailFromEmbed(iframe) {
        // Try to find thumbnail from parent elements
        const container = iframe.closest('.video-container, .media, article');
        if (container) {
            const thumbnail = container.querySelector('img');
            if (thumbnail) {
                return thumbnail.src;
            }
        }

        return null;
    }
}

// Initialize scanner
const scanner = new TumblrMediaScanner();

// Also provide direct access for debugging
window.tumblrMediaScanner = scanner;
