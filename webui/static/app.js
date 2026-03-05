/**
 * ZoteroRAG Search UI - Client-side JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('search-form');
    const queryInput = document.getElementById('query-input');
    const searchButton = document.getElementById('search-button');
    const loadingDiv = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const resultCount = document.getElementById('result-count');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');
    const minRelevanceInput = document.getElementById('min-relevance');
    const relevanceValue = document.getElementById('relevance-value');

    // Update relevance value display
    minRelevanceInput.addEventListener('input', function() {
        relevanceValue.textContent = this.value;
    });

    // Handle form submission
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;

        // Show loading, hide results and errors
        showLoading(true);
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
                    require_cited_bibtex: requireCitedBibtex
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Search failed');
            }

            displayResults(data.results, data.count);

        } catch (error) {
            showError(error.message);
        } finally {
            showLoading(false);
        }
    });

    function showLoading(show) {
        loadingDiv.classList.toggle('hidden', !show);
        searchButton.disabled = show;
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

    function displayResults(results, count) {
        if (!results || results.length === 0) {
            resultCount.textContent = 'No results found';
            resultsContainer.innerHTML = '<p class="no-results">Try adjusting your search query or lowering the minimum relevance threshold.</p>';
            resultsSection.classList.remove('hidden');
            return;
        }

        resultCount.textContent = `${count} result${count !== 1 ? 's' : ''} found`;
        
        resultsContainer.innerHTML = results.map((result, index) => {
            const title = escapeHtml(result.document_title || result.section_title || 'Untitled Document');
            const text = escapeHtml(result.text || '');
            const authors = Array.isArray(result.authors) ? result.authors.join(', ') : (result.authors || '');
            const date = result.date || '';
            
            // Format relevance score
            const relScore = parseFloat(result.relevance_score || 0).toFixed(2);
            const rerankScore = result.rerank_score ? parseFloat(result.rerank_score).toFixed(2) : null;

            // Build citations HTML if present
            let citationsHtml = '';
            const citedBibtex = result.cited_bibtex;
            if (citedBibtex && Array.isArray(citedBibtex) && citedBibtex.length > 0) {
                const citationItems = citedBibtex.map(bib => 
                    `<div class="citation-item">${escapeHtml(bib)}</div>`
                ).join('');
                
                citationsHtml = `
                    <div class="citations">
                        <div class="citations-title">Cited References (${citedBibtex.length})</div>
                        ${citationItems}
                    </div>
                `;
            }

            // Build metadata
            let metaParts = [];
            if (authors) metaParts.push(`Authors: ${escapeHtml(authors)}`);
            if (date) metaParts.push(`Date: ${escapeHtml(date)}`);
            
            const metadataHtml = metaParts.length > 0 
                ? `<div class="metadata">${metaParts.join(' | ')}</div>` 
                : '';

            return `
                <div class="result-card">
                    <div class="document-title">${title}</div>
                    ${metadataHtml}
                    <div class="result-text">${text}</div>
                    <div class="score-info">
                        Relevance: <span class="relevance-score">${relScore}</span>
                        ${rerankScore ? `<span class="relevance-score" style="background-color: #16a34a;">Rerank: ${rerankScore}</span>` : ''}
                    </div>
                    ${citationsHtml}
                </div>
            `;
        }).join('');

        resultsSection.classList.remove('hidden');
    }

    function escapeHtml(text) {
        if (!text) return '';
        
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});