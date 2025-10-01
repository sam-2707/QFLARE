const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8080';

class ApiService {
  private async handleResponse(response: Response) {
    if (!response.ok) {
      let errorMessage = 'Request failed';
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorData.message || errorMessage;
      } catch {
        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return response.json();
    }
    return response.text();
  }

  async registerDevice(data: {
    device_id: string;
    device_type: string;
    organization: string;
    contact_email: string;
    phone_number: string;
    use_case: string;
    key_exchange_method: string;
  }) {
    try {
      console.log('🚀 API Call - Register Device:', { url: `${API_BASE}/api/secure_register`, data });
      
      const formData = new URLSearchParams();
      Object.entries(data).forEach(([key, value]) => {
        formData.append(key, value);
      });

      console.log('📤 Sending form data:', formData.toString());

      const response = await fetch(`${API_BASE}/api/secure_register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });

      console.log('📥 Response received:', { 
        status: response.status, 
        statusText: response.statusText,
        url: response.url 
      });

      return this.handleResponse(response);
    } catch (error) {
      console.error('❌ API Error in registerDevice:', error);
      throw error;
    }
  }

  async verifyOtp(data: { device_id: string; otp: string }) {
    const formData = new URLSearchParams();
    formData.append('otp', data.otp);

    const response = await fetch(`${API_BASE}/api/secure_verify/${data.device_id}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    return this.handleResponse(response);
  }

  async getDevices() {
    const response = await fetch(`${API_BASE}/api/devices`);
    return this.handleResponse(response);
  }

  async getDeviceDetails(deviceId: string) {
    const response = await fetch(`${API_BASE}/api/devices/${deviceId}`);
    return this.handleResponse(response);
  }

  async deleteDevice(deviceId: string) {
    const response = await fetch(`${API_BASE}/api/devices/${deviceId}`, {
      method: 'DELETE',
    });
    return this.handleResponse(response);
  }

  async getSystemMetrics() {
    const response = await fetch(`${API_BASE}/api/admin/metrics`);
    return this.handleResponse(response);
  }

  async getActivityLogs() {
    const response = await fetch(`${API_BASE}/api/admin/logs`);
    return this.handleResponse(response);
  }

  async updateDeviceStatus(deviceId: string, status: string) {
    const formData = new URLSearchParams();
    formData.append('status', status);

    const response = await fetch(`${API_BASE}/api/devices/${deviceId}/status`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    return this.handleResponse(response);
  }

  async generateSecurityReport() {
    const response = await fetch(`${API_BASE}/api/admin/security-report`);
    return this.handleResponse(response);
  }
}

const apiService = new ApiService();
export default apiService;
