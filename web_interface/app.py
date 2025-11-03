#!/usr/bin/env python3
"""
Tumblr Image Collector Web Interface
Flask-based web application for Tumblr media collection
"""

import os
import sys
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import tempfile
import zipfile
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring_system import get_monitor, SystemHealthChecks, DiagnosticTool
import plotly.graph_objects as plotly_go
import plotly.utils

class WebInterface:
    def __init__(self):
        self.app = Flask(__name__,
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        CORS(self.app)

        # Configuration
        self.app.config['SECRET_KEY'] = 'tumblr-collector-web-secret-key'
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

        # Global state
        self.collector = None
        self.active_jobs = {}
        self.job_counter = 0

        # Monitoring system
        self.monitor = get_monitor()
        self.monitor.start_monitoring()

        self.setup_routes()
        self.setup_monitoring_routes()
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure required directories exist"""
        dirs = ['downloads', 'temp', 'logs']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/scan', methods=['POST'])
        def scan_blog():
            try:
                data = request.get_json()
                blog_name = data.get('blog_name', '').strip()
                tags = data.get('tags', [])
                date_range = data.get('date_range')
                include_likes = data.get('include_likes', False)

                if not blog_name:
                    return jsonify({'error': 'Blog name is required'}), 400

                # Initialize collector if needed
                if not self.collector:
                    self.collector = TumblrImageCollector()

                # Start scan in background
                job_id = self.start_scan_job(blog_name, tags, date_range, include_likes)

                return jsonify({
                    'job_id': job_id,
                    'status': 'started',
                    'message': f'Scan started for blog: {blog_name}'
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/job/<job_id>', methods=['GET'])
        def get_job_status(job_id):
            if job_id not in self.active_jobs:
                return jsonify({'error': 'Job not found'}), 404

            job = self.active_jobs[job_id]
            return jsonify({
                'job_id': job_id,
                'status': job['status'],
                'progress': job.get('progress', 0),
                'message': job.get('message', ''),
                'results': job.get('results', {}),
                'created_at': job['created_at'],
                'updated_at': job.get('updated_at')
            })

        @self.app.route('/api/jobs', methods=['GET'])
        def list_jobs():
            jobs = []
            for job_id, job in self.active_jobs.items():
                jobs.append({
                    'job_id': job_id,
                    'status': job['status'],
                    'progress': job.get('progress', 0),
                    'blog_name': job.get('blog_name', ''),
                    'created_at': job['created_at'],
                    'updated_at': job.get('updated_at')
                })
            return jsonify({'jobs': jobs})

        @self.app.route('/api/download/<job_id>', methods=['POST'])
        def download_media(job_id):
            try:
                if job_id not in self.active_jobs:
                    return jsonify({'error': 'Job not found'}), 404

                job = self.active_jobs[job_id]
                if job['status'] != 'completed':
                    return jsonify({'error': 'Job not completed yet'}), 400

                data = request.get_json()
                selected_items = data.get('selected_items', [])
                format_type = data.get('format', 'individual')  # individual or zip

                if format_type == 'zip':
                    # Create zip file
                    zip_path = self.create_zip_download(job_id, selected_items)
                    return send_file(zip_path, as_attachment=True,
                                   download_name=f'tumblr_collection_{job_id}.zip')
                else:
                    # Return download URLs for individual files
                    download_urls = self.get_download_urls(job_id, selected_items)
                    return jsonify({'download_urls': download_urls})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/settings', methods=['GET', 'POST'])
        def settings():
            if request.method == 'GET':
                # Return current settings
                settings = self.get_current_settings()
                return jsonify(settings)
            else:
                # Update settings
                try:
                    new_settings = request.get_json()
                    self.update_settings(new_settings)
                    return jsonify({'message': 'Settings updated successfully'})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500

        @self.app.route('/api/logs', methods=['GET'])
        def get_logs():
            try:
                # Return recent logs
                logs = self.get_recent_logs()
                return jsonify({'logs': logs})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'active_jobs': len(self.active_jobs)
            })

    def setup_monitoring_routes(self):
        """Setup monitoring and dashboard routes"""

        @self.app.route('/api/monitoring/health')
        def monitoring_health():
            """Get system health status"""
            results = self.monitor.run_all_checks()
            overall_status = self.monitor.get_overall_status()

            return jsonify({
                'overall_status': overall_status.value,
                'checks': {
                    name: {
                        'status': result.status.value,
                        'message': result.message,
                        'latency_ms': result.latency_ms,
                        'details': result.details
                    }
                    for name, result in results.items()
                },
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/monitoring/metrics')
        def monitoring_metrics():
            """Get system metrics"""
            try:
                import psutil

                metrics = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory': psutil.virtual_memory()._asdict(),
                    'disk': psutil.disk_usage('/')._asdict(),
                    'network': {
                        name: stats._asdict()
                        for name, stats in psutil.net_io_counters(pernic=True).items()
                    },
                    'timestamp': datetime.now().isoformat()
                }

                # Add custom metrics from collector
                metrics['jobs'] = {
                    'active': len(self.active_jobs),
                    'total': self.job_counter
                }

                return jsonify(metrics)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/monitoring/dashboard')
        def monitoring_dashboard():
            """Get dashboard data for charts"""
            try:
                # System info
                system_info = DiagnosticTool.system_info()

                # Health status
                health_results = self.monitor.run_all_checks()
                overall_status = self.monitor.get_overall_status()

                # Process info
                process_info = DiagnosticTool.process_info()

                # Dependency check
                dependencies = DiagnosticTool.dependency_check()

                return jsonify({
                    'system_info': system_info,
                    'health_status': {
                        'overall': overall_status.value,
                        'checks': {
                            name: {
                                'status': result.status.value,
                                'message': result.message,
                                'latency_ms': result.latency_ms
                            }
                            for name, result in health_results.items()
                        }
                    },
                    'process_info': process_info,
                    'dependencies': dependencies,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/monitoring/charts/cpu')
        def cpu_chart():
            """Get CPU usage chart data"""
            try:
                import psutil

                # Get CPU usage over time (simulate historical data)
                cpu_history = []
                for i in range(60):  # Last 60 seconds
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_history.append({
                        'time': (datetime.now() - timedelta(seconds=60-i)).isoformat(),
                        'value': cpu_percent
                    })

                return jsonify({
                    'data': cpu_history,
                    'current': psutil.cpu_percent(),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/monitoring/charts/memory')
        def memory_chart():
            """Get memory usage chart data"""
            try:
                import psutil

                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()

                return jsonify({
                    'memory_percent': memory.percent,
                    'memory_used': memory.used,
                    'memory_total': memory.total,
                    'memory_available': memory.available,
                    'swap_percent': swap.percent,
                    'swap_used': swap.used,
                    'swap_total': swap.total,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def start_scan_job(self, blog_name, tags=None, date_range=None, include_likes=False):
        """Start a scan job in background"""
        self.job_counter += 1
        job_id = f"job_{self.job_counter}"

        job = {
            'job_id': job_id,
            'blog_name': blog_name,
            'status': 'running',
            'progress': 0,
            'message': 'Initializing scan...',
            'results': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        self.active_jobs[job_id] = job

        # Start background thread
        thread = threading.Thread(
            target=self.run_scan_job,
            args=(job_id, blog_name, tags, date_range, include_likes),
            daemon=True
        )
        thread.start()

        return job_id

    def run_scan_job(self, job_id, blog_name, tags, date_range, include_likes):
        """Run the actual scan job"""
        try:
            job = self.active_jobs[job_id]

            # Update progress
            self.update_job_progress(job_id, 10, 'Connecting to Tumblr...')

            # Perform scan
            self.update_job_progress(job_id, 30, 'Scanning for posts...')

            # Get posts from Tumblr
            posts = self.collector.get_blog_posts(blog_name, limit=50)
            if posts is None:
                raise Exception("Rate limit exceeded or API error")

            self.update_job_progress(job_id, 50, f'Found {len(posts)} posts')

            # Process posts
            media_items = []
            for i, post in enumerate(posts):
                progress = 50 + (i / len(posts)) * 40
                self.update_job_progress(job_id, progress, f'Processing post {i+1}/{len(posts)}')

                # Extract media from post
                post_media = self.extract_media_from_post(post)
                media_items.extend(post_media)

            # Apply filters
            filtered_items = self.apply_web_filters(media_items, {
                'tags': tags,
                'date_range': date_range
            })

            self.update_job_progress(job_id, 90, f'Found {len(filtered_items)} media items')

            # Store results
            job['results'] = {
                'total_posts': len(posts),
                'media_items': filtered_items,
                'blog_name': blog_name
            }

            self.update_job_progress(job_id, 100, 'Scan completed')
            job['status'] = 'completed'

        except Exception as e:
            self.update_job_progress(job_id, 0, f'Error: {str(e)}')
            job['status'] = 'failed'
            job['error'] = str(e)

        job['updated_at'] = datetime.now().isoformat()

    def update_job_progress(self, job_id, progress, message):
        """Update job progress"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job['progress'] = progress
            job['message'] = message
            job['updated_at'] = datetime.now().isoformat()

    def extract_media_from_post(self, post):
        """Extract media items from a Tumblr post"""
        media_items = []

        post_type = post.get('type')
        post_data = {
            'post_id': post.get('id'),
            'timestamp': post.get('timestamp'),
            'tags': post.get('tags', []),
            'post_type': post_type
        }

        if post_type == 'photo':
            for photo in post.get('photos', []):
                original_size = photo.get('original_size', {})
                if original_size.get('url'):
                    media_items.append({
                        'type': 'image',
                        'url': original_size['url'],
                        'width': original_size.get('width'),
                        'height': original_size.get('height'),
                        'post_data': post_data
                    })

        elif post_type == 'video':
            video_url = post.get('video_url')
            if video_url:
                media_items.append({
                    'type': 'video',
                    'url': video_url,
                    'thumbnail': post.get('thumbnail_url'),
                    'post_data': post_data
                })

        return media_items

    def apply_web_filters(self, media_items, filters):
        """Apply filters to media items"""
        filtered = media_items

        # Apply tag filters if specified
        if filters.get('tags'):
            # Filter by tags (if post contains any of the specified tags)
            filtered = [
                item for item in filtered
                if any(tag.lower() in [t.lower() for t in item['post_data'].get('tags', [])]
                      for tag in filters['tags'])
            ]

        # Apply date range filters
        if filters.get('date_range'):
            date_range = filters['date_range']
            start_date = None
            end_date = None

            if date_range.get('start'):
                start_date = datetime.fromisoformat(date_range['start'].replace('Z', '+00:00'))
            if date_range.get('end'):
                end_date = datetime.fromisoformat(date_range['end'].replace('Z', '+00:00'))

            if start_date or end_date:
                filtered = [
                    item for item in filtered
                    if self.item_in_date_range(item, start_date, end_date)
                ]

        return filtered

    def item_in_date_range(self, item, start_date, end_date):
        """Check if media item is within date range"""
        timestamp = item['post_data'].get('timestamp')
        if not timestamp:
            return True  # Include items without timestamp

        item_date = datetime.fromtimestamp(timestamp)

        if start_date and item_date < start_date:
            return False
        if end_date and item_date > end_date:
            return False

        return True

    def create_zip_download(self, job_id, selected_items):
        """Create a zip file for download"""
        job = self.active_jobs[job_id]
        media_items = job['results']['media_items']

        # Create temporary zip file
        zip_path = Path('temp') / f'download_{job_id}.zip'

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, item in enumerate(media_items):
                if not selected_items or str(i) in selected_items:
                    # In a real implementation, you would download the file
                    # For now, we'll just add a placeholder
                    filename = f"{item['type']}_{i}.{self.get_extension_from_url(item['url'])}"
                    zip_file.writestr(filename, f"Placeholder for {item['url']}")

        return zip_path

    def get_download_urls(self, job_id, selected_items):
        """Get download URLs for individual files"""
        job = self.active_jobs[job_id]
        media_items = job['results']['media_items']

        urls = []
        for i, item in enumerate(media_items):
            if not selected_items or str(i) in selected_items:
                urls.append({
                    'url': item['url'],
                    'filename': f"{item['type']}_{i}.{self.get_extension_from_url(item['url'])}",
                    'type': item['type']
                })

        return urls

    def get_extension_from_url(self, url):
        """Extract file extension from URL"""
        path = url.split('?')[0]  # Remove query parameters
        if '.' in path:
            return path.split('.')[-1].lower()
        return 'jpg'  # Default

    def get_current_settings(self):
        """Get current application settings"""
        return {
            'download_path': 'downloads',
            'max_concurrent_downloads': 5,
            'default_filters': {
                'images': True,
                'videos': True,
                'gifs': True
            },
            'auto_cleanup': True,
            'log_level': 'INFO'
        }

    def update_settings(self, new_settings):
        """Update application settings"""
        # In a real implementation, save to config file
        pass

    def get_recent_logs(self):
        """Get recent application logs"""
        # In a real implementation, read from log files
        return [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': 'Web interface started'
            }
        ]

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.app.logger.info(f"Starting Tumblr Collector Web Interface on http://{host}:{port}")
        self.app.logger.info("Press Ctrl+C to stop")

        # Initialize collector
        try:
            self.collector = TumblrImageCollector()
        except Exception as e:
            self.app.logger.warning(f"Could not initialize Tumblr collector: {e}")
            self.app.logger.warning("Some features may not be available")

        self.app.run(host=host, port=port, debug=debug)

# Create web interface instance
web_interface = WebInterface()

if __name__ == '__main__':
    web_interface.run()
