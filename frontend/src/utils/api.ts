/**
 * API Utility Functions
 * Provides fetch with timeout, retry logic, and error handling
 */

const DEFAULT_TIMEOUT = 30000; // 30 seconds

export interface FetchOptions extends RequestInit {
  timeout?: number;
}

/**
 * Fetch with automatic timeout and abort capability
 */
export async function fetchWithTimeout(
  url: string,
  options: FetchOptions = {}
): Promise<Response> {
  const { timeout = DEFAULT_TIMEOUT, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. Please check your connection and try again.');
    }
    throw error;
  }
}

/**
 * Authenticated fetch with Bearer token
 */
export async function authenticatedFetch(
  url: string,
  options: FetchOptions = {}
): Promise<Response> {
  const token = localStorage.getItem('qflare-token');
  
  if (!token) {
    throw new Error('Authentication required. Please log in.');
  }

  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
    ...options.headers,
  };

  return fetchWithTimeout(url, { ...options, headers });
}

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  initialDelay: number = 1000
): Promise<T> {
  let lastError: Error;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      if (i < maxRetries - 1) {
        const delay = initialDelay * Math.pow(2, i);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError!;
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!localStorage.getItem('qflare-token');
}

/**
 * Get authentication token
 */
export function getAuthToken(): string | null {
  return localStorage.getItem('qflare-token');
}

/**
 * Parse error response
 */
export async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const data = await response.json();
    return data.detail || data.message || `Error: ${response.status} ${response.statusText}`;
  } catch {
    return `Error: ${response.status} ${response.statusText}`;
  }
}
