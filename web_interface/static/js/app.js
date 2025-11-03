// Tumblr Image Collector Web Interface JavaScript
// Atlassian Design System interaction patterns implementation

class AtlassianFocusManager {
    static trapFocus(container) {
        const focusableElements = container.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        const handleTabKey = (e) => {
            if (e.key === 'Tab') {
                if (e.shiftKey) {
                    if (document.activeElement === firstElement) {
                        lastElement.focus();
                        e.preventDefault();
                    }
                } else {
                    if (document.activeElement === lastElement) {
                        firstElement.focus();
                        e.preventDefault();
                    }
                }
            }
        };

        container.addEventListener('keydown', handleTabKey);
        return () => container.removeEventListener('keydown', handleTabKey);
    }
}

class AtlassianNotification {
    constructor(container) {
        this.container = container;
        this.notifications = new Map();
    }

    show(message, type = 'info', duration = 5000) {
        const id = Date.now().toString();
        const notification = this.createNotificationElement(message, type, id);

        this.notifications.set(id, notification);
        this.container.appendChild(notification);

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => this.remove(id), duration);
        }

        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
            notification.style.opacity = '1';
        });

        return id;
    }

    createNotificationElement(message, type, id) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'assertive');
        notification.style.transform = 'translateX(100%)';
        notification.style.opacity = '0';
        notification.style.transition = 'all 0.3s ease';

        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-title">${this.getTitleForType(type)}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close" aria-label="Close notification" onclick="app.removeNotification('${id}')">
                ×
            </button>
        `;

        return notification;
    }

    getTitleForType(type) {
        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Information'
        };
        return titles[type] || 'Information';
    }

    remove(id) {
        const notification = this.notifications.get(id);
        if (notification) {
            notification.style.transform = 'translateX(100%)';
            notification.style.opacity = '0';
            setTimeout(() => {
                notification.remove();
                this.notifications.delete(id);
            }, 300);
        }
    }

    clear() {
        this.notifications.forEach((_, id) => this.remove(id));
    }
}

class AtlassianModal {
    constructor(modalElement) {
        this.modal = modalElement;
        this.overlay = modalElement;
        this.isOpen = false;
        this.focusTrapRemover = null;

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Close on overlay click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.close();
            }
        });

        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });
    }

    open() {
        this.isOpen = true;
        this.modal.style.display = 'flex';
        this.modal.setAttribute('aria-hidden', 'false');

        // Focus trap
        this.focusTrapRemover = AtlassianFocusManager.trapFocus(this.modal);

        // Focus first focusable element
        const firstFocusable = this.modal.querySelector('button, [href], input, select, textarea');
        if (firstFocusable) {
            firstFocusable.focus();
        }

        // Prevent body scroll
        document.body.style.overflow = 'hidden';
    }

    close() {
        this.isOpen = false;
        this.modal.style.display = 'none';
        this.modal.setAttribute('aria-hidden', 'true');

        // Remove focus trap
        if (this.focusTrapRemover) {
            this.focusTrapRemover();
        }

        // Restore body scroll
        document.body.style.overflow = '';

        // Return focus to trigger element
        const trigger = document.querySelector('[aria-expanded="true"]');
        if (trigger) {
            trigger.focus();
            trigger.setAttribute('aria-expanded', 'false');
        }
    }
}

class AtlassianFormValidator {
    static validateField(field) {
        const value = field.value.trim();
        const isRequired = field.hasAttribute('required');

        // Clear previous errors
        field.classList.remove('error');

        if (isRequired && !value) {
            this.showFieldError(field, 'This field is required');
            return false;
        }

        // Custom validation based on field type/id
        if (field.id === 'blogName' && value) {
            if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
                this.showFieldError(field, 'Blog name can only contain letters, numbers, hyphens, and underscores');
                return false;
            }
        }

        return true;
    }

    static showFieldError(field, message) {
        field.classList.add('error');

        // Add error message if not exists
        let errorElement = field.parentNode.querySelector('.form-error');
        if (!errorElement) {
            errorElement = document.createElement('div');
            errorElement.className = 'form-error';
            field.parentNode.appendChild(errorElement);
        }
        errorElement.textContent = message;
    }

    static clearFieldError(field) {
        field.classList.remove('error');
        const errorElement = field.parentNode.querySelector('.form-error');
        if (errorElement) {
            errorElement.remove();
        }
    }

    static validateForm(form) {
        const fields = form.querySelectorAll('input[required], select[required], textarea[required]');
        let isValid = true;

        fields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });

        return isValid;
    }
}

class TumblrWebInterface {
    constructor() {
        this.activeJobs = new Map();
        this.selectedItems = new Set();
        this.currentJobId = null;

        // Atlassian components
        this.notifications = new AtlassianNotification(document.getElementById('notifications'));

        this.initializeElements();
        this.setupEventListeners();
        this.loadSettings();
        this.startJobPolling();
    }

    initializeElements() {
        // Forms
        this.scanForm = document.getElementById('scanForm');
        this.settingsForm = document.getElementById('settingsForm');

        // Buttons
        this.settingsBtn = document.getElementById('settingsBtn');
        this.monitoringBtn = document.getElementById('monitoringBtn');
        this.selectAllBtn = document.getElementById('selectAllBtn');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.confirmDownloadBtn = document.getElementById('confirmDownloadBtn');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');

        // Modals
        this.settingsModal = new AtlassianModal(document.getElementById('settingsModal'));
        this.modalCloses = document.querySelectorAll('.modal-close');

        // Other elements
        this.jobsList = document.getElementById('jobsList');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsCount = document.getElementById('resultsCount');
        this.mediaGrid = document.getElementById('mediaGrid');
        this.downloadOptions = document.getElementById('downloadOptions');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingMessage = document.getElementById('loadingMessage');
    }

    setupEventListeners() {
        // Form submissions
        this.scanForm.addEventListener('submit', (e) => this.handleScanSubmit(e));
        this.settingsForm.addEventListener('submit', (e) => this.handleSettingsSubmit(e));

        // Button clicks
        this.settingsBtn.addEventListener('click', () => this.showSettingsModal());
        this.monitoringBtn.addEventListener('click', () => this.toggleMonitoringView());
        this.selectAllBtn.addEventListener('click', () => this.selectAllItems());
        this.downloadBtn.addEventListener('click', () => this.showDownloadOptions());
        this.confirmDownloadBtn.addEventListener('click', () => this.confirmDownload());
        this.saveSettingsBtn.addEventListener('click', () => this.saveSettings());

        // Modal closes
        this.modalCloses.forEach(close => {
            close.addEventListener('click', () => this.hideSettingsModal());
        });

        // Click outside modal to close
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.hideSettingsModal();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'a':
                        e.preventDefault();
                        this.selectAllItems();
                        break;
                }
            }
        });
    }

    async handleScanSubmit(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const scanData = {
            blog_name: formData.get('blogName').trim(),
            tags: formData.get('tags').split(',').map(tag => tag.trim()).filter(tag => tag),
            include_likes: document.getElementById('includeLikes').checked,
            date_range: {
                start: formData.get('startDate') || null,
                end: formData.get('endDate') || null
            }
        };

        if (!scanData.blog_name) {
            this.showNotification('Error', 'Blog name is required', 'error');
            return;
        }

        try {
            this.showLoading('Starting scan...');
            const response = await fetch('/api/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(scanData)
            });

            const result = await response.json();

            if (response.ok) {
                this.currentJobId = result.job_id;
                this.showNotification('Success', `Scan started for blog: ${scanData.blog_name}`, 'success');
                this.updateJobsList();
            } else {
                throw new Error(result.error || 'Scan failed');
            }

        } catch (error) {
            this.showNotification('Error', error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async updateJobsList() {
        try {
            const response = await fetch('/api/jobs');
            const data = await response.json();

            this.jobsList.innerHTML = '';

            if (data.jobs.length === 0) {
                this.jobsList.innerHTML = '<p class="no-jobs">No active jobs</p>';
                return;
            }

            data.jobs.forEach(job => {
                const jobElement = this.createJobElement(job);
                this.jobsList.appendChild(jobElement);
            });

        } catch (error) {
            console.error('Failed to update jobs list:', error);
        }
    }

    createJobElement(job) {
        const div = document.createElement('div');
        div.className = 'job-item';
        div.dataset.jobId = job.job_id;

        const statusClass = job.status.toLowerCase();
        const progressPercent = job.progress || 0;

        div.innerHTML = `
            <div class="job-info">
                <h4>${job.blog_name || 'Unknown Blog'}</h4>
                <div class="job-status ${statusClass}">${job.status}</div>
                <div class="job-progress">
                    <div class="job-progress-fill" style="width: ${progressPercent}%"></div>
                </div>
                <small>Started: ${new Date(job.created_at).toLocaleString()}</small>
            </div>
            <div class="job-actions">
                ${job.status === 'completed' ? '<button class="btn btn-success btn-sm view-results">View Results</button>' : ''}
                ${job.status === 'running' ? '<button class="btn btn-secondary btn-sm">Running...</button>' : ''}
                ${job.status === 'failed' ? '<button class="btn btn-danger btn-sm view-error">View Error</button>' : ''}
            </div>
        `;

        // Event listeners
        const viewResultsBtn = div.querySelector('.view-results');
        if (viewResultsBtn) {
            viewResultsBtn.addEventListener('click', () => this.viewJobResults(job.job_id));
        }

        const viewErrorBtn = div.querySelector('.view-error');
        if (viewErrorBtn) {
            viewErrorBtn.addEventListener('click', () => this.viewJobError(job.job_id));
        }

        return div;
    }

    async viewJobResults(jobId) {
        try {
            const response = await fetch(`/api/job/${jobId}`);
            const job = await response.json();

            if (job.status === 'completed' && job.results.media_items) {
                this.displayResults(job.results.media_items, jobId);
                this.currentJobId = jobId;
            } else {
                this.showNotification('Error', 'Job results not available', 'error');
            }

        } catch (error) {
            this.showNotification('Error', 'Failed to load job results', 'error');
        }
    }

    displayResults(mediaItems, jobId) {
        this.resultsSection.style.display = 'block';
        this.resultsCount.textContent = mediaItems.length;
        this.mediaGrid.innerHTML = '';

        if (mediaItems.length === 0) {
            this.mediaGrid.innerHTML = '<p class="no-results">No media items found</p>';
            return;
        }

        mediaItems.forEach((item, index) => {
            const mediaElement = this.createMediaElement(item, index);
            this.mediaGrid.appendChild(mediaElement);
        });

        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    createMediaElement(item, index) {
        const div = document.createElement('div');
        div.className = 'media-item';
        div.dataset.index = index;

        const typeClass = item.type.toLowerCase();
        const thumbnail = item.thumbnail || item.url;
        const size = item.width && item.height ? `${item.width}×${item.height}` : '';

        div.innerHTML = `
            <img src="${thumbnail}" alt="Media thumbnail" class="media-thumbnail" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg=='">
            <div class="media-info">
                <span class="media-type ${typeClass}">${item.type}</span>
                <div class="media-url">${this.truncateUrl(item.url)}</div>
                <div class="media-meta">${size}</div>
            </div>
        `;

        div.addEventListener('click', () => this.toggleItemSelection(index));

        return div;
    }

    truncateUrl(url) {
        if (url.length <= 30) return url;
        return url.substring(0, 27) + '...';
    }

    toggleItemSelection(index) {
        const mediaItem = document.querySelector(`[data-index="${index}"]`);

        if (this.selectedItems.has(index)) {
            this.selectedItems.delete(index);
            mediaItem.classList.remove('selected');
        } else {
            this.selectedItems.add(index);
            mediaItem.classList.add('selected');
        }

        this.updateDownloadButton();
    }

    selectAllItems() {
        const mediaItems = document.querySelectorAll('.media-item');

        if (this.selectedItems.size === mediaItems.length) {
            // Deselect all
            this.selectedItems.clear();
            mediaItems.forEach(item => item.classList.remove('selected'));
        } else {
            // Select all
            this.selectedItems.clear();
            mediaItems.forEach((item, index) => {
                this.selectedItems.add(index);
                item.classList.add('selected');
            });
        }

        this.updateDownloadButton();
    }

    updateDownloadButton() {
        const hasSelection = this.selectedItems.size > 0;
        this.downloadBtn.disabled = !hasSelection;

        if (hasSelection) {
            this.downloadBtn.textContent = `📥 Download Selected (${this.selectedItems.size})`;
        } else {
            this.downloadBtn.textContent = '📥 Download Selected';
        }
    }

    showDownloadOptions() {
        if (this.selectedItems.size === 0) return;
        this.downloadOptions.style.display = 'block';
        this.downloadOptions.scrollIntoView({ behavior: 'smooth' });
    }

    async confirmDownload() {
        if (!this.currentJobId || this.selectedItems.size === 0) return;

        const format = document.querySelector('input[name="downloadFormat"]:checked').value;
        const selectedItemsArray = Array.from(this.selectedItems);

        try {
            this.showLoading('Preparing download...');

            const response = await fetch(`/api/download/${this.currentJobId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    selected_items: selectedItemsArray,
                    format: format
                })
            });

            if (format === 'zip') {
                // Download ZIP file
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `tumblr_collection_${this.currentJobId}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } else {
                // Handle individual downloads
                const data = await response.json();
                if (data.download_urls) {
                    data.download_urls.forEach(item => {
                        const a = document.createElement('a');
                        a.href = item.url;
                        a.download = item.filename;
                        a.target = '_blank';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    });
                }
            }

            this.showNotification('Success', 'Download started', 'success');
            this.downloadOptions.style.display = 'none';

        } catch (error) {
            this.showNotification('Error', 'Download failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showSettingsModal() {
        this.settingsModal.style.display = 'flex';
    }

    hideSettingsModal() {
        this.settingsModal.style.display = 'none';
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/settings');
            const settings = await response.json();

            // Apply settings to form
            document.getElementById('downloadPath').value = settings.download_path || 'downloads';
            document.getElementById('maxConcurrent').value = settings.max_concurrent_downloads || 5;
            document.getElementById('filterImages').checked = settings.default_filters?.images !== false;
            document.getElementById('filterVideos').checked = settings.default_filters?.videos !== false;
            document.getElementById('filterGifs').checked = settings.default_filters?.gifs !== false;

        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }

    async saveSettings() {
        try {
            const settings = {
                download_path: document.getElementById('downloadPath').value,
                max_concurrent_downloads: parseInt(document.getElementById('maxConcurrent').value),
                default_filters: {
                    images: document.getElementById('filterImages').checked,
                    videos: document.getElementById('filterVideos').checked,
                    gifs: document.getElementById('filterGifs').checked
                }
            };

            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });

            if (response.ok) {
                this.showNotification('Success', 'Settings saved', 'success');
                this.hideSettingsModal();
            } else {
                throw new Error('Failed to save settings');
            }

        } catch (error) {
            this.showNotification('Error', error.message, 'error');
        }
    }

    showLoading(message = 'Loading...') {
        this.loadingMessage.textContent = message;
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showNotification(title, message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;

        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close">&times;</button>
        `;

        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });

        this.notifications.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    startJobPolling() {
        // Poll for job updates every 2 seconds
        setInterval(() => {
            if (this.activeJobs.size > 0) {
                this.updateJobsList();
            }
        }, 2000);
    }

    viewJobError(jobId) {
        // Show error details (implement based on job error data)
        this.showNotification('Error', 'Job failed. Check the application logs for details.', 'error');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tumblrWebInterface = new TumblrWebInterface();
});
