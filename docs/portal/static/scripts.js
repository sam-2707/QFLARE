// QFLARE Documentation Portal JavaScript

class QFLAREDocs {
    constructor() {
        this.theme = localStorage.getItem('qflare-theme') || 'light';
        this.searchIndex = null;
        this.init();
    }

    init() {
        this.setupTheme();
        this.setupNavigation();
        this.setupSearch();
        this.setupInteractiveExamples();
        this.setupScrollSpy();
        this.setupCopyCode();
        this.setupTableOfContents();
    }

    // Theme Management
    setupTheme() {
        const themeToggle = document.getElementById('theme-toggle');
        const body = document.body;

        // Apply saved theme
        if (this.theme === 'dark') {
            body.setAttribute('data-theme', 'dark');
            if (themeToggle) themeToggle.textContent = '☀️';
        }

        // Theme toggle handler
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = body.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                
                body.setAttribute('data-theme', newTheme);
                localStorage.setItem('qflare-theme', newTheme);
                themeToggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';
                this.theme = newTheme;
            });
        }
    }

    // Navigation
    setupNavigation() {
        // Mobile menu toggle
        const menuToggle = document.querySelector('.menu-toggle');
        const navLinks = document.querySelector('.nav-links');

        if (menuToggle && navLinks) {
            menuToggle.addEventListener('click', () => {
                navLinks.classList.toggle('active');
            });
        }

        // Active page highlighting
        const currentPath = window.location.pathname;
        const navLinksElements = document.querySelectorAll('.nav-links a');
        
        navLinksElements.forEach(link => {
            if (link.getAttribute('href') === currentPath.split('/').pop()) {
                link.classList.add('active');
            }
        });

        // Smooth scrolling for anchor links
        document.addEventListener('click', (e) => {
            if (e.target.matches('a[href^="#"]')) {
                e.preventDefault();
                const target = document.querySelector(e.target.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    }

    // Search Functionality
    setupSearch() {
        const searchInput = document.querySelector('.search-input');
        const searchResults = document.querySelector('.search-results');
        
        if (!searchInput || !searchResults) return;

        // Load search index
        this.loadSearchIndex();

        // Search handler with debouncing
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.performSearch(e.target.value, searchResults);
            }, 300);
        });

        // Search on enter
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch(e.target.value, searchResults);
            }
        });
    }

    async loadSearchIndex() {
        try {
            const response = await fetch('search_index.json');
            this.searchIndex = await response.json();
        } catch (error) {
            console.warn('Failed to load search index:', error);
        }
    }

    performSearch(query, resultsContainer) {
        if (!query.trim() || !this.searchIndex) {
            resultsContainer.innerHTML = '';
            return;
        }

        const results = this.searchDocuments(query.toLowerCase());
        this.displaySearchResults(results, resultsContainer, query);
    }

    searchDocuments(query) {
        if (!this.searchIndex || !this.searchIndex.documents) return [];

        const results = [];
        const queryTerms = query.split(' ').filter(term => term.length > 0);

        this.searchIndex.documents.forEach(doc => {
            let score = 0;
            const content = `${doc.title} ${doc.content}`.toLowerCase();

            queryTerms.forEach(term => {
                const titleMatches = (doc.title.toLowerCase().match(new RegExp(term, 'g')) || []).length;
                const contentMatches = (doc.content.toLowerCase().match(new RegExp(term, 'g')) || []).length;
                const tagMatches = doc.tags.filter(tag => tag.toLowerCase().includes(term)).length;

                score += titleMatches * 3 + contentMatches + tagMatches * 2;
            });

            if (score > 0) {
                results.push({ ...doc, score });
            }
        });

        return results.sort((a, b) => b.score - a.score).slice(0, 10);
    }

    displaySearchResults(results, container, query) {
        if (results.length === 0) {
            container.innerHTML = `
                <div class="search-result">
                    <h3>No results found</h3>
                    <p>No documents match your search query "${query}".</p>
                </div>
            `;
            return;
        }

        const resultsHTML = results.map(result => `
            <div class="search-result">
                <h3><a href="${result.url}">${this.highlightQuery(result.title, query)}</a></h3>
                <p>${this.highlightQuery(this.truncateText(result.content, 150), query)}</p>
                <div class="search-tags">
                    <span class="search-tag">${result.category}</span>
                    ${result.tags.map(tag => `<span class="search-tag">${tag}</span>`).join('')}
                </div>
            </div>
        `).join('');

        container.innerHTML = resultsHTML;
    }

    highlightQuery(text, query) {
        const queryTerms = query.split(' ').filter(term => term.length > 0);
        let highlightedText = text;

        queryTerms.forEach(term => {
            const regex = new RegExp(`(${term})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
        });

        return highlightedText;
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }

    // Interactive Examples
    setupInteractiveExamples() {
        const runButtons = document.querySelectorAll('.run-button');
        
        runButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.runExample(e.target);
            });
        });
    }

    async runExample(button) {
        const exampleContainer = button.closest('.example-container');
        const codeElement = exampleContainer.querySelector('code');
        const code = codeElement.textContent;

        button.disabled = true;
        button.textContent = 'Running...';

        try {
            // Create output container
            let outputContainer = exampleContainer.querySelector('.example-output');
            if (!outputContainer) {
                outputContainer = document.createElement('div');
                outputContainer.className = 'example-output';
                outputContainer.innerHTML = '<h4>Output:</h4><pre class="output-content"></pre>';
                exampleContainer.appendChild(outputContainer);
            }

            const outputContent = outputContainer.querySelector('.output-content');
            outputContent.textContent = 'Executing code...';

            // Simulate code execution (in a real implementation, this would send to a backend)
            await this.simulateCodeExecution(code, outputContent);

        } catch (error) {
            console.error('Error running example:', error);
            const outputContent = exampleContainer.querySelector('.output-content');
            if (outputContent) {
                outputContent.textContent = `Error: ${error.message}`;
                outputContent.style.color = 'var(--danger-color)';
            }
        } finally {
            button.disabled = false;
            button.textContent = 'Run Code';
        }
    }

    async simulateCodeExecution(code, outputElement) {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Mock output based on code content
        let output = '';
        
        if (code.includes('print(')) {
            const printMatches = code.match(/print\((.*?)\)/g);
            if (printMatches) {
                output = printMatches.map(match => {
                    const content = match.replace(/print\(|\)/g, '');
                    return content.replace(/['"]/g, '');
                }).join('\n');
            }
        } else if (code.includes('QFLAREClient')) {
            output = `Client initialized successfully
Training started...
Epoch 1/5: Loss = 0.234
Epoch 2/5: Loss = 0.189
Epoch 3/5: Loss = 0.156
Training completed!`;
        } else if (code.includes('generate_keypair')) {
            output = `Key pair generated successfully
Public key length: 1568 bytes
Private key length: 3168 bytes
Algorithm: CRYSTALS-Kyber-1024`;
        } else if (code.includes('aggregate')) {
            output = `Aggregation completed
Participants: 10
Byzantine tolerance: 30%
Aggregated model accuracy: 94.2%`;
        } else {
            output = `Code executed successfully!
No explicit output statements found.`;
        }

        outputElement.textContent = output;
        outputElement.style.color = 'var(--success-color)';
    }

    // Scroll Spy
    setupScrollSpy() {
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        const tocLinks = document.querySelectorAll('.toc a');

        if (headings.length === 0 || tocLinks.length === 0) return;

        const observerOptions = {
            rootMargin: '-20% 0px -70% 0px',
            threshold: 0
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    tocLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, observerOptions);

        headings.forEach(heading => {
            if (heading.id) {
                observer.observe(heading);
            }
        });
    }

    // Copy Code
    setupCopyCode() {
        const codeBlocks = document.querySelectorAll('pre code');
        
        codeBlocks.forEach(codeBlock => {
            const pre = codeBlock.parentElement;
            
            // Create copy button
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.innerHTML = '📋';
            copyButton.title = 'Copy code';
            
            // Position button
            pre.style.position = 'relative';
            copyButton.style.position = 'absolute';
            copyButton.style.top = '10px';
            copyButton.style.right = '10px';
            copyButton.style.background = 'var(--surface-color)';
            copyButton.style.border = '1px solid var(--border-color)';
            copyButton.style.borderRadius = '4px';
            copyButton.style.padding = '5px 8px';
            copyButton.style.cursor = 'pointer';
            copyButton.style.fontSize = '14px';
            
            pre.appendChild(copyButton);
            
            // Copy functionality
            copyButton.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(codeBlock.textContent);
                    copyButton.innerHTML = '✅';
                    copyButton.title = 'Copied!';
                    
                    setTimeout(() => {
                        copyButton.innerHTML = '📋';
                        copyButton.title = 'Copy code';
                    }, 2000);
                } catch (error) {
                    console.error('Failed to copy code:', error);
                    copyButton.innerHTML = '❌';
                    setTimeout(() => {
                        copyButton.innerHTML = '📋';
                    }, 2000);
                }
            });
        });
    }

    // Table of Contents
    setupTableOfContents() {
        const tocContainer = document.querySelector('.toc');
        if (!tocContainer) return;

        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        if (headings.length === 0) {
            tocContainer.style.display = 'none';
            return;
        }

        const tocList = document.createElement('ul');
        tocList.className = 'toc-list';

        headings.forEach((heading, index) => {
            // Generate ID if not present
            if (!heading.id) {
                heading.id = `heading-${index}`;
            }

            const listItem = document.createElement('li');
            listItem.className = `toc-item toc-${heading.tagName.toLowerCase()}`;
            
            const link = document.createElement('a');
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent;
            
            listItem.appendChild(link);
            tocList.appendChild(listItem);
        });

        tocContainer.appendChild(tocList);
    }

    // Utility Methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Analytics and Performance
class DocsAnalytics {
    constructor() {
        this.startTime = Date.now();
        this.interactions = [];
        this.init();
    }

    init() {
        this.trackPageView();
        this.trackClicks();
        this.trackScrollDepth();
        this.trackTimeOnPage();
    }

    trackPageView() {
        const pageData = {
            type: 'pageview',
            url: window.location.href,
            title: document.title,
            timestamp: Date.now(),
            userAgent: navigator.userAgent,
            language: navigator.language
        };
        
        this.sendAnalytics(pageData);
    }

    trackClicks() {
        document.addEventListener('click', (e) => {
            const clickData = {
                type: 'click',
                element: e.target.tagName,
                className: e.target.className,
                text: e.target.textContent?.substring(0, 50),
                timestamp: Date.now()
            };
            
            this.interactions.push(clickData);
        });
    }

    trackScrollDepth() {
        let maxScroll = 0;
        
        window.addEventListener('scroll', this.throttle(() => {
            const scrollPercent = Math.round(
                (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
            );
            
            if (scrollPercent > maxScroll) {
                maxScroll = scrollPercent;
                
                const scrollData = {
                    type: 'scroll',
                    depth: scrollPercent,
                    timestamp: Date.now()
                };
                
                this.interactions.push(scrollData);
            }
        }, 1000));
    }

    trackTimeOnPage() {
        window.addEventListener('beforeunload', () => {
            const timeData = {
                type: 'time_on_page',
                duration: Date.now() - this.startTime,
                interactions: this.interactions.length,
                timestamp: Date.now()
            };
            
            this.sendAnalytics(timeData);
        });
    }

    sendAnalytics(data) {
        // In a real implementation, this would send to an analytics service
        console.log('Analytics:', data);
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QFLAREDocs();
    new DocsAnalytics();
    
    // Add fade-in animation to elements
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });
    
    document.querySelectorAll('.feature-card, .example-container, .search-result').forEach(el => {
        observer.observe(el);
    });
});

// Export for use in other scripts
window.QFLAREDocs = QFLAREDocs;