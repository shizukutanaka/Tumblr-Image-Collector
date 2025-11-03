// Tumblr Image Collector Options Script
class OptionsManager {
    constructor() {
        this.defaultSettings = {
            // General settings
            autoScan: false,
            showNotifications: true,
            downloadPath: 'Tumblr_Collector',
            maxConcurrentDownloads: 3,

            // Default filters
            defaultFilterImages: true,
            defaultFilterVideos: true,
            defaultFilterGifs: true,
            defaultMinWidth: 0,
            defaultMinHeight: 0,
            defaultSkipDuplicates: false,

            // Advanced settings
            debugMode: false,
            contextMenu: true,

            // Metadata
            lastVersion: chrome.runtime.getManifest().version,
            installDate: new Date().toISOString()
        };

        this.initializeElements();
        this.setupEventListeners();
        this.loadSettings();
        this.displayVersion();
    }

    initializeElements() {
        // General settings
        this.autoScan = document.getElementById('autoScan');
        this.showNotifications = document.getElementById('showNotifications');
        this.downloadPath = document.getElementById('downloadPath');
        this.maxConcurrentDownloads = document.getElementById('maxConcurrentDownloads');

        // Default filters
        this.defaultFilterImages = document.getElementById('defaultFilterImages');
        this.defaultFilterVideos = document.getElementById('defaultFilterVideos');
        this.defaultFilterGifs = document.getElementById('defaultFilterGifs');
        this.defaultMinWidth = document.getElementById('defaultMinWidth');
        this.defaultMinHeight = document.getElementById('defaultMinHeight');
        this.defaultSkipDuplicates = document.getElementById('defaultSkipDuplicates');

        // Advanced settings
        this.debugMode = document.getElementById('debugMode');
        this.contextMenu = document.getElementById('contextMenu');

        // Actions
        this.saveButton = document.getElementById('saveSettings');
        this.resetButton = document.getElementById('resetSettings');
        this.exportButton = document.getElementById('exportData');
        this.clearButton = document.getElementById('clearData');
        this.openExtensionPage = document.getElementById('openExtensionPage');

        // Status
        this.statusMessage = document.getElementById('statusMessage');
        this.versionNumber = document.getElementById('versionNumber');
    }

    setupEventListeners() {
        // Save and reset buttons
        this.saveButton.addEventListener('click', () => this.saveSettings());
        this.resetButton.addEventListener('click', () => this.resetToDefaults());

        // Data management
        this.exportButton.addEventListener('click', () => this.exportData());
        this.clearButton.addEventListener('click', () => this.clearAllData());

        // Links
        this.openExtensionPage.addEventListener('click', (e) => {
            e.preventDefault();
            // This would link to Chrome Web Store page
            alert('Chrome Web Store page will be available soon!');
        });

        // Auto-save on change (optional)
        const inputs = document.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('change', () => {
                this.showStatus('Settings changed. Click "Save Settings" to apply.', 'info');
            });
        });
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get(['settings']);
            const settings = result.settings || {};

            // Merge with defaults for any missing settings
            const mergedSettings = { ...this.defaultSettings, ...settings };

            // Apply settings to form
            this.applySettingsToForm(mergedSettings);

            this.showStatus('Settings loaded successfully.', 'success');

        } catch (error) {
            console.error('Failed to load settings:', error);
            this.showStatus('Failed to load settings. Using defaults.', 'error');
            this.applySettingsToForm(this.defaultSettings);
        }
    }

    applySettingsToForm(settings) {
        // General settings
        this.autoScan.checked = settings.autoScan || false;
        this.showNotifications.checked = settings.showNotifications !== false;
        this.downloadPath.value = settings.downloadPath || 'Tumblr_Collector';
        this.maxConcurrentDownloads.value = settings.maxConcurrentDownloads || 3;

        // Default filters
        this.defaultFilterImages.checked = settings.defaultFilterImages !== false;
        this.defaultFilterVideos.checked = settings.defaultFilterVideos !== false;
        this.defaultFilterGifs.checked = settings.defaultFilterGifs !== false;
        this.defaultMinWidth.value = settings.defaultMinWidth || 0;
        this.defaultMinHeight.value = settings.defaultMinHeight || 0;
        this.defaultSkipDuplicates.checked = settings.defaultSkipDuplicates || false;

        // Advanced settings
        this.debugMode.checked = settings.debugMode || false;
        this.contextMenu.checked = settings.contextMenu !== false;
    }

    getSettingsFromForm() {
        return {
            // General settings
            autoScan: this.autoScan.checked,
            showNotifications: this.showNotifications.checked,
            downloadPath: this.downloadPath.value.trim() || 'Tumblr_Collector',
            maxConcurrentDownloads: parseInt(this.maxConcurrentDownloads.value) || 3,

            // Default filters
            defaultFilterImages: this.defaultFilterImages.checked,
            defaultFilterVideos: this.defaultFilterVideos.checked,
            defaultFilterGifs: this.defaultFilterGifs.checked,
            defaultMinWidth: parseInt(this.defaultMinWidth.value) || 0,
            defaultMinHeight: parseInt(this.defaultMinHeight.value) || 0,
            defaultSkipDuplicates: this.defaultSkipDuplicates.checked,

            // Advanced settings
            debugMode: this.debugMode.checked,
            contextMenu: this.contextMenu.checked,

            // Metadata
            lastVersion: chrome.runtime.getManifest().version,
            lastModified: new Date().toISOString()
        };
    }

    async saveSettings() {
        try {
            const settings = this.getSettingsFromForm();

            // Validate settings
            if (!this.validateSettings(settings)) {
                return;
            }

            await chrome.storage.sync.set({ settings });

            // Update context menu if setting changed
            await this.updateContextMenu(settings.contextMenu);

            this.showStatus('Settings saved successfully!', 'success');

            // Track the save event
            chrome.runtime.sendMessage({
                action: 'trackEvent',
                eventType: 'settings_saved',
                data: { timestamp: new Date().toISOString() }
            });

        } catch (error) {
            console.error('Failed to save settings:', error);
            this.showStatus('Failed to save settings. Please try again.', 'error');
        }
    }

    validateSettings(settings) {
        // Validate download path
        if (!settings.downloadPath.trim()) {
            this.showStatus('Download folder name cannot be empty.', 'error');
            this.downloadPath.focus();
            return false;
        }

        // Validate concurrent downloads
        if (settings.maxConcurrentDownloads < 1 || settings.maxConcurrentDownloads > 10) {
            this.showStatus('Max concurrent downloads must be between 1 and 10.', 'error');
            this.maxConcurrentDownloads.focus();
            return false;
        }

        // Validate dimensions
        if (settings.defaultMinWidth < 0 || settings.defaultMinHeight < 0) {
            this.showStatus('Minimum dimensions cannot be negative.', 'error');
            return false;
        }

        return true;
    }

    async resetToDefaults() {
        if (!confirm('Are you sure you want to reset all settings to defaults? This cannot be undone.')) {
            return;
        }

        try {
            await chrome.storage.sync.set({ settings: this.defaultSettings });
            this.applySettingsToForm(this.defaultSettings);
            await this.updateContextMenu(this.defaultSettings.contextMenu);

            this.showStatus('Settings reset to defaults.', 'success');

        } catch (error) {
            console.error('Failed to reset settings:', error);
            this.showStatus('Failed to reset settings.', 'error');
        }
    }

    async updateContextMenu(enabled) {
        try {
            if (enabled) {
                // Context menu is created in background.js, just ensure it's enabled
                await chrome.runtime.sendMessage({
                    action: 'updateContextMenu',
                    enabled: true
                });
            } else {
                // Remove context menu items
                await chrome.contextMenus.removeAll();
            }
        } catch (error) {
            console.warn('Failed to update context menu:', error);
        }
    }

    async exportData() {
        try {
            // Request export from background script
            await chrome.runtime.sendMessage({ action: 'exportData' });
            this.showStatus('Data export started. Check your downloads folder.', 'success');

        } catch (error) {
            console.error('Failed to export data:', error);
            this.showStatus('Failed to export data.', 'error');
        }
    }

    async clearAllData() {
        const message = 'This will permanently delete all your settings, cached data, and statistics. This cannot be undone. Are you sure?';

        if (!confirm(message)) {
            return;
        }

        try {
            // Clear all storage
            await Promise.all([
                chrome.storage.sync.clear(),
                chrome.storage.local.clear()
            ]);

            // Reset form to defaults
            this.applySettingsToForm(this.defaultSettings);

            this.showStatus('All data cleared. Extension reset to fresh install state.', 'success');

        } catch (error) {
            console.error('Failed to clear data:', error);
            this.showStatus('Failed to clear all data.', 'error');
        }
    }

    showStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message status-${type}`;
        this.statusMessage.style.display = 'block';

        // Auto-hide success messages after 3 seconds
        if (type === 'success') {
            setTimeout(() => {
                this.statusMessage.style.display = 'none';
            }, 3000);
        }
    }

    displayVersion() {
        const manifest = chrome.runtime.getManifest();
        this.versionNumber.textContent = manifest.version;
    }

    // Utility methods
    static formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    static formatDate(dateString) {
        try {
            return new Date(dateString).toLocaleDateString();
        } catch (error) {
            return 'Unknown';
        }
    }
}

// Initialize options page
document.addEventListener('DOMContentLoaded', () => {
    new OptionsManager();
});
