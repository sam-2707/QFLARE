document.addEventListener('DOMContentLoaded', () => {
    const generateKeysBtn = document.getElementById('generate-keys-btn');
    const keysContainer = document.getElementById('keys-container');
    const downloadKeysBtn = document.getElementById('download-keys-btn');
    const registerForm = document.getElementById('register-form');
    const messageContainer = document.getElementById('message-container');

    let privateKeys = {};

    generateKeysBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/request_qkey');
            const keys = await response.json();

            document.getElementById('kem-public-key').value = keys.kem_public_key;
            document.getElementById('signature-public-key').value = keys.signature_public_key;
            document.getElementById('kem-private-key').value = keys.kem_private_key;
            document.getElementById('signature-private-key').value = keys.signature_private_key;

            document.getElementById('kem-public-key-form').value = keys.kem_public_key;
            document.getElementById('signature-public-key-form').value = keys.signature_public_key;

            privateKeys = {
                kem_private_key: keys.kem_private_key,
                signature_private_key: keys.signature_private_key
            };

            keysContainer.style.display = 'block';
            downloadKeysBtn.style.display = 'block';
        } catch (error) {
            messageContainer.innerHTML = `<p class="error">Failed to generate keys: ${error}</p>`;
        }
    });

    downloadKeysBtn.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(privateKeys, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'private_keys.json';
        a.click();
        URL.revokeObjectURL(url);
    });

    registerForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const formData = new FormData(registerForm);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/api/enroll', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                messageContainer.innerHTML = `<p class="success">${result.message}</p>`;
            } else {
                messageContainer.innerHTML = `<p class="error">${result.detail}</p>`;
            }
        } catch (error) {
            messageContainer.innerHTML = `<p class="error">Failed to register device: ${error}</p>`;
        }
    });
});