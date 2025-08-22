/**
 * QFLARE Interactive UI Script
 * Handles client-side interactions for the web dashboard.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Attach event listeners to buttons
    const generateKeysBtn = document.getElementById('generateKeysBtn');
    if (generateKeysBtn) {
        generateKeysBtn.addEventListener('click', handleGenerateKeys);
    }
});

/**
 * Handles the "Generate Keys" button click.
 * Makes an API call to the server to get a new PQC key pair and displays it.
 */
async function handleGenerateKeys() {
    const resultBox = document.getElementById('keyGenResult');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    // Show loading state
    resultBox.style.display = 'none';
    loadingSpinner.style.display = 'block';

    try {
        // Fetch new keys from the server API
        const response = await fetch('/api/request_qkey', {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();

        // Display the generated keys in the UI
        document.getElementById('kemKey').textContent = data.kem_public_key;
        document.getElementById('sigKey').textContent = data.signature_public_key;
        document.getElementById('deviceId').textContent = data.device_id;
        
        resultBox.style.display = 'block';

    } catch (error) {
        console.error('Error generating keys:', error);
        resultBox.innerHTML = `<div class="alert alert-error">Failed to generate keys. Please check the server logs.</div>`;
        resultBox.style.display = 'block';
    } finally {
        // Hide loading state
        loadingSpinner.style.display = 'none';
    }
}