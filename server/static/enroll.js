document.addEventListener('DOMContentLoaded', () => {
    const enrollForm = document.getElementById('enrollForm');
    const enrollResult = document.getElementById('enrollResult');

    enrollForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        enrollResult.textContent = '';

        const deviceId = document.getElementById('deviceId').value.trim();
        const kemKey = document.getElementById('kemKey').value.trim();
        const sigKey = document.getElementById('sigKey').value.trim();
        const enrollToken = document.getElementById('enrollToken').value.trim();

        enrollResult.innerHTML = '<span>Enrolling device...</span>';

        try {
            const response = await fetch('/api/enroll', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    device_id: deviceId,
                    kem_public_key: kemKey,
                    signature_public_key: sigKey,
                    enrollment_token: enrollToken
                })
            });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                enrollResult.innerHTML = `<div class="alert alert-success">Device enrolled successfully!<br>Message: ${data.message}</div>`;
            } else {
                enrollResult.innerHTML = `<div class="alert alert-error">Failed to enroll device.<br>${data.message || data.detail || 'Unknown error.'}</div>`;
            }
        } catch (err) {
            enrollResult.innerHTML = `<div class="alert alert-error">Error: ${err.message}</div>`;
        }
    });
});
