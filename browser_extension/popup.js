// Tumblr Image Collector Popup Script
class TumblrCollectorPopup {
    constructor() {
        this.mediaItems = [];
        this.selectedItems = new Set();
        this.isDownloading = false;

        this.initializeElements();
        this.setupEventListeners();
        this.loadSettings();
        this.updateUI();
    }

    initializeElements() {
        // Main controls
        this.scanButton = document.getElementById('scanPage');
        this.downloadAllButton = document.getElementById('downloadAll');
        this.downloadSelectedButton = document.getElementById('downloadSelected');

        // Filters
        this.filterImages = document.getElementById('filterImages');
        this.filterVideos = document.getElementById('filterVideos');
        this.filterGifs = document.getElementById('filterGifs');
        this.minWidth = document.getElementById('minWidth');
        this.minHeight = document.getElementById('minHeight');
        this.skipDuplicates = document.getElementById('skipDuplicates');

        // Media list
        this.mediaContainer = document.getElementById('mediaContainer');
        this.mediaCount = document.getElementById('mediaCount');

        // Progress
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.progressStats = document.getElementById('progressStats');

        // Other
        this.statusElement = document.getElementById('status');
        this.logContainer = document.getElementById('logContainer');
        this.clearButton = document.getElementById('clearList');
        this.optionsButton = document.getElementById('openOptions');
    }

    setupEventListeners() {
        // Main controls
        this.scanButton.addEventListener('click', () => this.scanCurrentPage());
        this.downloadAllButton.addEventListener('click', () => this.downloadAll());
        this.downloadSelectedButton.addEventListener('click', () => this.downloadSelected());

        // Filters
        [this.filterImages, this.filterVideos, this.filterGifs].forEach(filter => {
            filter.addEventListener('change', () => this.applyFilters());
        });

        [this.minWidth, this.minHeight].forEach(input => {
            input.addEventListener('input', () => this.applyFilters());
        });

        // Other actions
        this.clearButton.addEventListener('click', () => this.clearMediaList());
        this.optionsButton.addEventListener('click', () => this.openOptions());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'a':
                        e.preventDefault();
                        this.selectAll();
                        break;
                    case 'd':
                        e.preventDefault();
                        this.deselectAll();
                        break;
                }
            }
        });
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get([
                'filterImages', 'filterVideos', 'filterGifs',
                'minWidth', 'minHeight', 'skipDuplicates'
            ]);

            this.filterImages.checked = result.filterImages !== false;
            this.filterVideos.checked = result.filterVideos !== false;
            this.filterGifs.checked = result.filterGifs !== false;
            this.minWidth.value = result.minWidth || '';
            this.minHeight.value = result.minHeight || '';
            this.skipDuplicates.checked = result.skipDuplicates === true;

        } catch (error) {
            this.log('Warning', 'Failed to load settings: ' + error.message);
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.sync.set({
                filterImages: this.filterImages.checked,
                filterVideos: this.filterVideos.checked,
                filterGifs: this.filterGifs.checked,
                minWidth: this.minWidth.value,
                minHeight: this.minHeight.value,
                skipDuplicates: this.skipDuplicates.checked
            });
        } catch (error) {
            this.log('Error', 'Failed to save settings: ' + error.message);
        }
    }

    async scanCurrentPage() {
        try {
            this.setStatus('Scanning...');
            this.scanButton.disabled = true;

            // Get current tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (!tab.url.includes('tumblr.com')) {
                throw new Error('This is not a Tumblr page. Please navigate to a Tumblr blog.');
            }

            // Send message to content script
            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'scanMedia',
                filters: this.getCurrentFilters()
            });

            if (response.success) {
                this.mediaItems = response.media || [];
                this.updateMediaList();
                this.setStatus(`Found ${this.mediaItems.length} items`);
                this.log('Info', `Scanned page: ${this.mediaItems.length} media items found`);
            } else {
                throw new Error(response.error || 'Scan failed');
            }

        } catch (error) {
            this.setStatus('Scan failed');
            this.log('Error', 'Scan failed: ' + error.message);
            this.showNotification('Scan Failed', error.message, 'error');
        } finally {
            this.scanButton.disabled = false;
        }
    }

    getCurrentFilters() {
        return {
            images: this.filterImages.checked,
            videos: this.filterVideos.checked,
            gifs: this.filterGifs.checked,
            minWidth: parseInt(this.minWidth.value) || 0,
            minHeight: parseInt(this.minHeight.value) || 0,
            skipDuplicates: this.skipDuplicates.checked
        };
    }

    updateMediaList() {
        this.mediaContainer.innerHTML = '';
        this.selectedItems.clear();

        if (this.mediaItems.length === 0) {
            this.mediaContainer.innerHTML = '<div class="no-media">No media found on this page</div>';
            this.updateUI();
            return;
        }

        this.mediaItems.forEach((item, index) => {
            const mediaElement = this.createMediaElement(item, index);
            this.mediaContainer.appendChild(mediaElement);
        });

        this.updateUI();
    }

    createMediaElement(item, index) {
        const div = document.createElement('div');
        div.className = 'media-item';
        div.dataset.index = index;

        const typeEmoji = this.getTypeEmoji(item.type);
        const sizeText = item.width && item.height ? `${item.width}×${item.height}` : 'Unknown size';

        div.innerHTML = `
            <input type="checkbox" class="media-checkbox" data-index="${index}">
            <img src="${item.thumbnail || ''}" class="media-thumbnail" alt="Thumbnail" onerror="this.style.display='none'">
            <div class="media-info">
                <div class="media-type">${typeEmoji} ${item.type}</div>
                <div class="media-url">${item.url}</div>
                <div class="media-size">${sizeText}</div>
            </div>
        `;

        // Event listeners
        const checkbox = div.querySelector('.media-checkbox');
        checkbox.addEventListener('change', () => this.onItemSelectionChange(index));

        div.addEventListener('click', (e) => {
            if (e.target.type !== 'checkbox') {
                checkbox.checked = !checkbox.checked;
                this.onItemSelectionChange(index);
            }
        });

        return div;
    }

    getTypeEmoji(type) {
        switch (type.toLowerCase()) {
            case 'image': return '🖼️';
            case 'video': return '🎥';
            case 'gif': return '🎞️';
            default: return '📄';
        }
    }

    onItemSelectionChange(index) {
        const checkbox = document.querySelector(`[data-index="${index}"] .media-checkbox`);
        const mediaItem = document.querySelector(`[data-index="${index}"]`);

        if (checkbox.checked) {
            this.selectedItems.add(index);
            mediaItem.classList.add('selected');
        } else {
            this.selectedItems.delete(index);
            mediaItem.classList.remove('selected');
        }

        this.updateUI();
    }

    selectAll() {
        this.mediaItems.forEach((_, index) => {
            const checkbox = document.querySelector(`[data-index="${index}"] .media-checkbox`);
            if (checkbox && !checkbox.checked) {
                checkbox.checked = true;
                this.onItemSelectionChange(index);
            }
        });
    }

    deselectAll() {
        this.mediaItems.forEach((_, index) => {
            const checkbox = document.querySelector(`[data-index="${index}"] .media-checkbox`);
            if (checkbox && checkbox.checked) {
                checkbox.checked = false;
                this.onItemSelectionChange(index);
            }
        });
    }

    applyFilters() {
        // Save settings when filters change
        this.saveSettings();

        // Re-scan if we have media items
        if (this.mediaItems.length > 0) {
            this.scanCurrentPage();
        }
    }

    async downloadAll() {
        if (this.mediaItems.length === 0) return;
        await this.downloadMedia(Array.from({length: this.mediaItems.length}, (_, i) => i));
    }

    async downloadSelected() {
        if (this.selectedItems.size === 0) return;
        await this.downloadMedia(Array.from(this.selectedItems));
    }

    async downloadMedia(indices) {
        if (this.isDownloading) return;

        try {
            this.isDownloading = true;
            this.setStatus('Downloading...');
            this.showProgress();

            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const itemsToDownload = indices.map(i => this.mediaItems[i]);

            let completed = 0;
            const total = itemsToDownload.length;

            for (const item of itemsToDownload) {
                try {
                    this.updateProgress(completed, total, `Downloading ${item.url}`);
                    await this.downloadSingleItem(item);
                    completed++;
                    this.log('Info', `Downloaded: ${item.url}`);
                } catch (error) {
                    this.log('Error', `Failed to download ${item.url}: ${error.message}`);
                }
            }

            this.setStatus(`Download complete: ${completed}/${total} items`);
            this.showNotification('Download Complete', `${completed} items downloaded successfully`, 'success');

        } catch (error) {
            this.setStatus('Download failed');
            this.log('Error', 'Download failed: ' + error.message);
            this.showNotification('Download Failed', error.message, 'error');
        } finally {
            this.isDownloading = false;
            this.hideProgress();
        }
    }

    async downloadSingleItem(item) {
        return new Promise((resolve, reject) => {
            chrome.downloads.download({
                url: item.url,
                filename: this.generateFilename(item),
                saveAs: false
            }, (downloadId) => {
                if (chrome.runtime.lastError) {
                    reject(new Error(chrome.runtime.lastError.message));
                } else {
                    resolve(downloadId);
                }
            });
        });
    }

    generateFilename(item) {
        const extension = item.url.split('.').pop().split('?')[0] || 'jpg';
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const type = item.type.toLowerCase();

        // Create organized folder structure
        const folder = `Tumblr_Collector/${new URL(item.url).hostname}/${type}s`;

        return `${folder}/${timestamp}_${type}.${extension}`;
    }

    updateProgress(completed, total, text) {
        const percentage = total > 0 ? (completed / total) * 100 : 0;
        this.progressFill.style.width = `${percentage}%`;
        this.progressText.textContent = text;
        this.progressStats.textContent = `(${completed}/${total})`;
    }

    showProgress() {
        this.progressSection.style.display = 'block';
        this.progressFill.style.width = '0%';
    }

    hideProgress() {
        this.progressSection.style.display = 'none';
    }

    updateUI() {
        const hasMedia = this.mediaItems.length > 0;
        const hasSelection = this.selectedItems.size > 0;

        this.downloadAllButton.disabled = !hasMedia || this.isDownloading;
        this.downloadSelectedButton.disabled = !hasSelection || this.isDownloading;
        this.mediaCount.textContent = this.mediaItems.length;
    }

    clearMediaList() {
        this.mediaItems = [];
        this.selectedItems.clear();
        this.updateMediaList();
        this.setStatus('List cleared');
    }

    openOptions() {
        chrome.tabs.create({ url: chrome.runtime.getURL('options.html') });
    }

    setStatus(status) {
        this.statusElement.textContent = status;
        this.statusElement.className = 'status-indicator';

        // Update color based on status
        if (status.includes('failed') || status.includes('error')) {
            this.statusElement.classList.add('status-error');
        } else if (status.includes('complete') || status.includes('success')) {
            this.statusElement.classList.add('status-success');
        } else if (status.includes('...')) {
            this.statusElement.classList.add('status-working');
        }
    }

    log(level, message) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';

        const timestamp = new Date().toLocaleTimeString();
        entry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level-${level.toLowerCase()}">[${level}]</span>
            <span class="log-message">${message}</span>
        `;

        this.logContainer.appendChild(entry);
        this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }

    showNotification(title, message, type = 'info') {
        // Use Chrome notifications API
        chrome.notifications.create({
            type: 'basic',
            iconUrl: chrome.runtime.getURL('icons/icon128.png'),
            title: title,
            message: message
        });
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TumblrCollectorPopup();
});
