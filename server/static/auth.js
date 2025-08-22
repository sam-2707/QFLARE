document.addEventListener('DOMContentLoaded', () => {
    const challengeForm = document.getElementById('challenge-form');
    const authenticateForm = document.getElementById('authenticate-form');
    const challengeContainer = document.getElementById('challenge-container');
    const messageContainer = document.getElementById('message-container');

    challengeForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const formData = new FormData(challengeForm);
        const deviceId = formData.get('device_id');

        try {
            const response = await fetch('/api/challenge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ device_id: deviceId })
            });

            const result = await response.json();

            if (response.ok) {
                document.getElementById('challenge').value = result.challenge;
                challengeContainer.style.display = 'block';
            } else {
                messageContainer.innerHTML = `<p class="error">${result.detail}</p>`;
            }
        } catch (error) {
            messageContainer.innerHTML = `<p class="error">Failed to get challenge: ${error}</p>`;
        }
    });

    authenticateForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const formData = new FormData(authenticateForm);
        const signature = formData.get('signature');
        const deviceId = document.getElementById('device-id').value;
        const challenge = document.getElementById('challenge').value;

        try {
            const response = await fetch('/api/verify_challenge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    device_id: deviceId, 
                    challenge_response: signature 
                })
            });

            const result = await response.json();

            if (response.ok) {
                messageContainer.innerHTML = `<p class="success">${result.message}</p>`;
            } else {
                messageContainer.innerHTML = `<p class="error">${result.detail}</p>`;
            }
        } catch (error) {
            messageContainer.innerHTML = `<p class="error">Failed to authenticate: ${error}</p>`;
        }
    });
});