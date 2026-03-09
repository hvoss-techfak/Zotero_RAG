/**
 * SemTero Search UI - Client-side JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('search-form');
    const queryInput = document.getElementById('query-input');
    const searchButton = document.getElementById('search-button');
    const loadingDiv = document.getElementById('loading');
    const searchProgressBar = document.getElementById('search-progress-bar');
    const searchProgressLabel = document.getElementById('search-progress-label');
    const searchProgressDetail = document.getElementById('search-progress-detail');
    const searchProgressPercent = document.getElementById('search-progress-percent');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const resultCount = document.getElementById('result-count');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');
    const minRelevanceInput = document.getElementById('min-relevance');
    const relevanceValue = document.getElementById('relevance-value');
    const embedNowButton = document.getElementById('embed-now-button');
    const embedSummary = document.getElementById('embed-summary');
    const embedProgressBar = document.getElementById('embed-progress-bar');
    const embedProgressText = document.getElementById('embed-progress-text');
    const embedSentencesText = document.getElementById('embed-sentences-text');
    const embedNextRun = document.getElementById('embed-next-run');
    let activeSearchToken = 0;

    // Update relevance value display
    minRelevanceInput.addEventListener('input', function() {
        relevanceValue.textContent = this.value;
    });

    // Handle embed now button click
    embedNowButton.addEventListener('click', async function() {
        embedNowButton.disabled = true;
        embedNowButton.textContent = 'Starting…';
        hideError();

        try {
            const response = await fetch('/api/embed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || data.message || 'Failed to start embedding');
            }
            await refreshStatus();
        } catch (error) {
            showError(error.message);
        } finally {
            embedNowButton.textContent = 'Embed new documents now';
            embedNowButton.disabled = false;
        }
    });

    // Handle form submission
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;

        const searchToken = ++activeSearchToken;
        const searchId = generateSearchId();
        const stopPolling = startSearchProgressPolling(searchId, searchToken);

        // Show loading, hide results and errors
        showLoading(true);
        renderSearchProgress({
            percentage: 4,
            message: 'Preparing search',
            detail: 'Starting semantic search'
        });
        hideResults();
        hideError();

        try {
            const topSentences = document.getElementById('top-sentences').value;
            const minRelevance = minRelevanceInput.value;
            const requireCitedBibtex = document.getElementById('require-cited-bibtex').checked;

            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    top_sentences: parseInt(topSentences),
                    min_relevance: parseFloat(minRelevance),
                    require_cited_bibtex: requireCitedBibtex,
                    search_id: searchId
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Search failed');
            }

            stopPolling();
            if (searchToken === activeSearchToken) {
                displayResults(data.results, data.count);
            }

        } catch (error) {
            stopPolling();
            if (searchToken === activeSearchToken) {
                showError(error.message);
            }
        } finally {
            if (searchToken === activeSearchToken) {
                showLoading(false);
            }
        }
    });

    function showLoading(show) {
        loadingDiv.classList.toggle('hidden', !show);
        searchButton.disabled = show;
    }

    function renderSearchProgress(progress) {
        const pct = Math.max(0, Math.min(100, Number(progress.percentage || 0)));
        const message = progress.message || 'Searching';
        const detail = progress.detail || '';

        searchProgressBar.style.width = `${pct}%`;
        searchProgressLabel.textContent = message;
        searchProgressDetail.textContent = detail;
        searchProgressPercent.textContent = `${Math.round(pct)}%`;
    }

    function startSearchProgressPolling(searchId, searchToken) {
        let timeoutId = null;
        let stopped = false;

        async function poll() {
            if (stopped || searchToken !== activeSearchToken) {
                return;
            }

            try {
                const response = await fetch(`/api/search-progress/${encodeURIComponent(searchId)}`, {
                    cache: 'no-store'
                });

                if (response.ok) {
                    const progress = await response.json();
                    if (searchToken === activeSearchToken) {
                        renderSearchProgress(progress);
                    }
                    if (progress.finished) {
                        return;
                    }
                }
            } catch (error) {
                console.debug('Search progress polling failed', error);
            }

            if (!stopped && searchToken === activeSearchToken) {
                timeoutId = window.setTimeout(poll, 350);
            }
        }

        timeoutId = window.setTimeout(poll, 0);

        return function stop() {
            stopped = true;
            if (timeoutId !== null) {
                window.clearTimeout(timeoutId);
            }
        };
    }

    function generateSearchId() {
        if (window.crypto && typeof window.crypto.randomUUID === 'function') {
            return window.crypto.randomUUID();
        }
        return `search-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function hideResults() {
        resultsSection.classList.add('hidden');
        resultsContainer.innerHTML = '';
    }

    function hideError() {
        errorSection.classList.add('hidden');
    }

    function showError(message) {
        errorText.textContent = message;
        errorSection.classList.remove('hidden');
    }

    async function refreshStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch embedding status');
            }
            renderStatus(data);
        } catch (error) {
            embedSummary.textContent = error.message;
            embedProgressBar.style.width = '0%';
        }
    }

    function renderStatus(status) {
        const pct = Math.max(0, Math.min(100, Number(status.progress_percentage || 0)));
        embedProgressBar.style.width = `${pct}%`;

        const running = Boolean(status.is_running);
        const total = Number(status.total_documents || 0);
        const processed = Number(status.processed_documents || 0);
        const failed = Number(status.failed_documents || 0);
        const summary = status.last_run_summary || (running ? 'Embedding in progress' : 'Idle');

        embedSummary.textContent = running
            ? `${summary} • ${pct.toFixed(1)}% complete`
            : summary;
        embedProgressText.textContent = `${processed} / ${total} documents`;
        embedSentencesText.textContent = `${Number(status.embedded_sentences || 0)} sentences embedded${failed ? ` • ${failed} failed` : ''}`;
        embedNextRun.textContent = status.next_auto_reembed_at
            ? `Next auto re-embed: ${formatDateTime(status.next_auto_reembed_at)}`
            : 'No auto re-embed scheduled yet';
        embedNowButton.disabled = running;
    }

    function displayResults(results, count) {
        if (!results || results.length === 0) {
            resultCount.textContent = 'No results found';
            resultsContainer.innerHTML = '<p class="no-results">Try adjusting your search query or lowering the minimum relevance threshold.</p>';
            resultsSection.classList.remove('hidden');
            return;
        }

        resultCount.textContent = `${count} result${count !== 1 ? 's' : ''} found`;

        resultsContainer.innerHTML = results.map((result) => {
            const title = escapeHtml(result.document_title || result.section_title || 'Untitled Document');
            const text = escapeHtml(result.text || '');
            const authors = Array.isArray(result.authors) ? result.authors.join(', ') : (result.authors || '');
            const date = result.date || '';
            const relScore = parseFloat(result.relevance_score || 0).toFixed(2);
            const rerankScore = result.rerank_score ? parseFloat(result.rerank_score).toFixed(2) : null;
            const documentBibtex = result.bibtex ? `<pre class="bibtex-block">${escapeHtml(result.bibtex)}</pre>` : '<p class="muted">No document BibTeX available.</p>';

            let citedBibtexHtml = '';
            const citedBibtex = result.cited_bibtex;
            if (Array.isArray(citedBibtex) && citedBibtex.length > 0) {
                citedBibtexHtml = `
                    <div class="citations">
                        <div class="citations-title">Cited references (${citedBibtex.length})</div>
                        ${citedBibtex.map(bib => `<pre class="bibtex-block compact">${escapeHtml(bib)}</pre>`).join('')}
                    </div>
                `;
            }

            const metaParts = [];
            if (authors) metaParts.push(`Authors: ${escapeHtml(authors)}`);
            if (date) metaParts.push(`Date: ${escapeHtml(date)}`);
            if (result.item_type) metaParts.push(`Type: ${escapeHtml(result.item_type)}`);
            const metadataHtml = metaParts.length > 0 ? `<div class="metadata">${metaParts.join(' | ')}</div>` : '';

            return `
                <div class="result-card">
                    <div class="document-title">${title}</div>
                    ${metadataHtml}
                    <div class="result-text">${text}</div>
                    <div class="score-info">
                        Relevance: <span class="relevance-score">${relScore}</span>
                        ${rerankScore ? `<span class="relevance-score alt">Rerank: ${rerankScore}</span>` : ''}
                    </div>
                    <div class="document-bibtex">
                        <div class="citations-title">Document BibTeX</div>
                        ${documentBibtex}
                    </div>
                    ${citedBibtexHtml}
                </div>
            `;
        }).join('');

        resultsSection.classList.remove('hidden');
    }

    function formatDateTime(value) {
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleString();
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    refreshStatus();
    setInterval(refreshStatus, 3000);
});

