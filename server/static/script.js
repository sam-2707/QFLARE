/**
 * QFLARE Interactive UI Script
 * Handles client-side interactions for the web dashboard.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Corrected ID to match the HTML in register_v2.html
    const generateKeysBtn = document.getElementById('generate-keys-btn');
    if (generateKeysBtn) {
        generateKeysBtn.addEventListener('click', handleGenerateKeys);
    }

    // Load status data if on the status page
    loadStatusData();
});

/**
 * Handles the "Generate Keys" button click.
 * Makes an API call to the server to get a new PQC key pair and displays it.
 */
async function handleGenerateKeys() {
    const resultBox = document.getElementById('keyGenResult');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    // Show loading state
    if (resultBox) resultBox.style.display = 'none';
    if (loadingSpinner) loadingSpinner.style.display = 'block';

    try {
        // Fetch new keys from the server API
        const response = await fetch('/api/request_qkey');

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();

        // Display the generated keys in the UI
        if (document.getElementById('kemKey')) {
            document.getElementById('kemKey').textContent = data.kem_public_key;
        }
        if (document.getElementById('sigKey')) {
            document.getElementById('sigKey').textContent = data.signature_public_key;
        }
        if (document.getElementById('deviceId')) {
            document.getElementById('deviceId').textContent = data.device_id;
        }
        
        if (resultBox) resultBox.style.display = 'block';

    } catch (error) {
        console.error('Error generating keys:', error);
        if (resultBox) {
            resultBox.innerHTML = `<div class="alert alert-error">Failed to generate keys. Please check the server logs and ensure it's running.</div>`;
            resultBox.style.display = 'block';
        }
    } finally {
        // Hide loading state
        if (loadingSpinner) loadingSpinner.style.display = 'none';
    }
}

/**
 * Fetches and displays system status on the status.html page.
 */
async function loadStatusData() {
    if (document.getElementById('health-status')) {
        try {
            // Fetch health data
            const healthResponse = await fetch('/health');
            const healthData = await healthResponse.json();
            const healthContainer = document.getElementById('health-status');
            healthContainer.innerHTML = `
                <p><strong>Overall Status:</strong> <span class="status-badge">${healthData.status}</span></p>
                <p><strong>Device Count:</strong> ${healthData.device_count}</p>
                <p><strong>Aggregator Status:</strong> ${healthData.components.aggregator}</p>
            `;

            // Fetch enclave data
            const enclaveResponse = await fetch('/api/enclave/status');
            const enclaveData = await enclaveResponse.json();
            const enclaveContainer = document.getElementById('enclave-status');
            enclaveContainer.innerHTML = `
                <p><strong>Type:</strong> ${enclaveData.enclave_type}</p>
                <p><strong>Status:</strong> ${enclaveData.status}</p>
                <p><strong>Poison Threshold:</strong> ${enclaveData.poison_threshold}</p>
                <p><strong>Total Aggregations:</strong> ${enclaveData.total_aggregations}</p>
            `;

        } catch (error) {
            console.error('Failed to load status data:', error);
            document.getElementById('health-status').innerHTML = '<p class="error">Could not load health data.</p>';
            document.getElementById('enclave-status').innerHTML = '<p class="error">Could not load enclave data.</p>';
        }
    }
}