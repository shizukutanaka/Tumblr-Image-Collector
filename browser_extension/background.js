// Tumblr Image Collector Background Script
class TumblrCollectorBackground {
    constructor() {
        this.setupEventListeners();
        this.initializeStorage();
    }

    setupEventListeners() {
        // Extension installation
        chrome.runtime.onInstalled.addListener((details) => {
            if (details.reason === 'install') {
                this.handleFirstInstall();
            } else if (details.reason === 'update') {
                this.handleUpdate(details.previousVersion);
            }
        });

        // Tab updates (for automatic scanning)
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete' && tab.url?.includes('tumblr.com')) {
                this.handleTumblrTabUpdate(tabId, tab);
            }
        });

        // Download monitoring
        chrome.downloads.onChanged.addListener((downloadDelta) => {
            this.handleDownloadChange(downloadDelta);
        });

        // Context menu (if enabled)
        this.setupContextMenu();
    }

    async initializeStorage() {
        try {
            const result = await chrome.storage.sync.get(['settings']);
            if (!result.settings) {
                // Set default settings
                await chrome.storage.sync.set({
                    settings: {
                        autoScan: false,
                        showNotifications: true,
                        downloadPath: 'Tumblr_Collector',
                        maxConcurrentDownloads: 3,
                        lastVersion: chrome.runtime.getManifest().version
                    }
                });
            }
        } catch (error) {
            console.error('Failed to initialize storage:', error);
        }
    }

    handleFirstInstall() {
        // Create welcome notification
        chrome.notifications.create({
            type: 'basic',
            iconUrl: chrome.runtime.getURL('icons/icon128.png'),
            title: 'Tumblr Image Collector Installed!',
            message: 'Right-click on Tumblr pages to start collecting media.'
        });

        // Open options page for setup
        chrome.tabs.create({
            url: chrome.runtime.getURL('options.html'),
            active: true
        });
    }

    handleUpdate(previousVersion) {
        const currentVersion = chrome.runtime.getManifest().version;

        chrome.notifications.create({
            type: 'basic',
            iconUrl: chrome.runtime.getURL('icons/icon128.png'),
            title: 'Tumblr Image Collector Updated!',
            message: `Updated from ${previousVersion} to ${currentVersion}. Check out new features!`
        });
    }

    async handleTumblrTabUpdate(tabId, tab) {
        try {
            const settings = await chrome.storage.sync.get(['settings']);
            const autoScan = settings.settings?.autoScan;

            if (autoScan) {
                // Send message to content script to scan automatically
                chrome.tabs.sendMessage(tabId, {
                    action: 'autoScan',
                    tabInfo: {
                        url: tab.url,
                        title: tab.title
                    }
                }).catch(error => {
                    // Content script might not be ready yet
                    console.debug('Auto-scan failed (content script not ready):', error);
                });
            }
        } catch (error) {
            console.error('Failed to handle Tumblr tab update:', error);
        }
    }

    handleDownloadChange(downloadDelta) {
        const downloadId = downloadDelta.id;

        if (downloadDelta.state?.current === 'complete') {
            this.notifyDownloadComplete(downloadId);
        } else if (downloadDelta.state?.current === 'interrupted') {
            this.notifyDownloadFailed(downloadId);
        }
    }

    async notifyDownloadComplete(downloadId) {
        try {
            const settings = await chrome.storage.sync.get(['settings']);
            if (!settings.settings?.showNotifications) return;

            const download = await chrome.downloads.search({ id: downloadId });
            if (download.length > 0) {
                const filename = download[0].filename.split('/').pop();

                chrome.notifications.create({
                    type: 'basic',
                    iconUrl: chrome.runtime.getURL('icons/icon128.png'),
                    title: 'Download Complete',
                    message: `Successfully downloaded: ${filename}`
                });
            }
        } catch (error) {
            console.error('Failed to show download complete notification:', error);
        }
    }

    async notifyDownloadFailed(downloadId) {
        try {
            const settings = await chrome.storage.sync.get(['settings']);
            if (!settings.settings?.showNotifications) return;

            chrome.notifications.create({
                type: 'basic',
                iconUrl: chrome.runtime.getURL('icons/icon128.png'),
                title: 'Download Failed',
                message: 'One or more downloads failed. Check the popup for details.'
            });
        } catch (error) {
            console.error('Failed to show download failed notification:', error);
        }
    }

    setupContextMenu() {
        // Create context menu items
        chrome.contextMenus.create({
            id: 'scan-page',
            title: 'Scan Tumblr Page for Media',
            contexts: ['page'],
            documentUrlPatterns: ['*://*.tumblr.com/*']
        });

        chrome.contextMenus.create({
            id: 'download-image',
            title: 'Download with Tumblr Collector',
            contexts: ['image'],
            targetUrlPatterns: ['*://*.tumblr.com/*', '*://*media.tumblr.com/*']
        });

        // Handle context menu clicks
        chrome.contextMenus.onClicked.addListener((info, tab) => {
            switch (info.menuItemId) {
                case 'scan-page':
                    this.handleContextScan(tab);
                    break;
                case 'download-image':
                    this.handleContextDownload(info.srcUrl, tab);
                    break;
            }
        });
    }

    async handleContextScan(tab) {
        try {
            // Open popup programmatically
            await chrome.action.openPopup();

            // Trigger scan after a short delay
            setTimeout(() => {
                chrome.runtime.sendMessage({
                    action: 'triggerScan'
                });
            }, 500);

        } catch (error) {
            console.error('Failed to handle context scan:', error);
        }
    }

    async handleContextDownload(imageUrl, tab) {
        try {
            // Start download directly
            const filename = this.generateContextFilename(imageUrl);

            chrome.downloads.download({
                url: imageUrl,
                filename: filename,
                saveAs: false
            });

        } catch (error) {
            console.error('Failed to handle context download:', error);
        }
    }

    generateContextFilename(url) {
        const extension = url.split('.').pop().split('?')[0] || 'jpg';
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const hostname = new URL(url).hostname;

        return `Tumblr_Collector/${hostname}/context_download_${timestamp}.${extension}`;
    }

    // API for other parts of the extension
    async getSettings() {
        try {
            const result = await chrome.storage.sync.get(['settings']);
            return result.settings || {};
        } catch (error) {
            console.error('Failed to get settings:', error);
            return {};
        }
    }

    async updateSettings(newSettings) {
        try {
            const current = await this.getSettings();
            const updated = { ...current, ...newSettings };
            await chrome.storage.sync.set({ settings: updated });
            return true;
        } catch (error) {
            console.error('Failed to update settings:', error);
            return false;
        }
    }

    // Statistics tracking
    async trackEvent(eventType, data) {
        try {
            const stats = await chrome.storage.local.get(['stats']);
            const currentStats = stats.stats || {};

            if (!currentStats[eventType]) {
                currentStats[eventType] = { count: 0, lastUsed: null };
            }

            currentStats[eventType].count++;
            currentStats[eventType].lastUsed = new Date().toISOString();

            if (data) {
                currentStats[eventType].data = data;
            }

            await chrome.storage.local.set({ stats: currentStats });

        } catch (error) {
            console.error('Failed to track event:', error);
        }
    }

    // Export collected data
    async exportData() {
        try {
            const [syncData, localData] = await Promise.all([
                chrome.storage.sync.get(null),
                chrome.storage.local.get(null)
            ]);

            const exportData = {
                version: chrome.runtime.getManifest().version,
                exportDate: new Date().toISOString(),
                syncStorage: syncData,
                localStorage: localData
            };

            // Create download
            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });

            const downloadUrl = URL.createObjectURL(dataBlob);
            const filename = `tumblr_collector_backup_${new Date().toISOString().split('T')[0]}.json`;

            chrome.downloads.download({
                url: downloadUrl,
                filename: filename,
                saveAs: true
            });

        } catch (error) {
            console.error('Failed to export data:', error);
        }
    }
}

// Initialize background script
const background = new TumblrCollectorBackground();

// Make it globally available for debugging
window.tumblrCollectorBackground = background;
