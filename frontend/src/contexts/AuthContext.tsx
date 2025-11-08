import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface User {
  id: string;
  username: string;
  role: 'admin' | 'user';
  email: string;
  full_name?: string;
  is_active: boolean;
  is_verified: boolean;
}

interface RegisterData {
  email: string;
  username: string;
  full_name?: string;
  password: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isUser: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (userData: RegisterData) => Promise<boolean>;
  logout: () => Promise<void>;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

// API configuration
const API_BASE_URL = 'http://localhost:8002/api';

interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Load user from localStorage on app start
  useEffect(() => {
    const savedUser = localStorage.getItem('qflare-user');
    const savedToken = localStorage.getItem('qflare-token');
    
    if (savedUser && savedToken) {
      try {
        const parsedUser = JSON.parse(savedUser);
        setUser(parsedUser);
        // Optionally validate token with backend
        validateToken(savedToken);
      } catch (error) {
        console.error('Error parsing saved user:', error);
        localStorage.removeItem('qflare-user');
        localStorage.removeItem('qflare-token');
        localStorage.removeItem('qflare-refresh-token');
      }
    }
    setLoading(false);
  }, []);

  const validateToken = async (token: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        // Token is invalid, clear stored data
        await logout();
      }
    } catch (error) {
      console.error('Token validation error:', error);
      // Network error, keep user logged in but they might need to re-login later
    }
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email,
          password: password,
        }),
      });

      if (response.ok) {
        const data: LoginResponse = await response.json();
        
        // Store tokens
        localStorage.setItem('qflare-token', data.access_token);
        localStorage.setItem('qflare-refresh-token', data.refresh_token);
        
        // Get user info from backend
        const userResponse = await fetch(`${API_BASE_URL}/auth/me`, {
          headers: {
            'Authorization': `Bearer ${data.access_token}`,
          },
        });
        
        if (userResponse.ok) {
          const userData = await userResponse.json();
          setUser(userData);
          localStorage.setItem('qflare-user', JSON.stringify(userData));
          
          setLoading(false);
          return true;
        } else {
          console.error('Failed to get user data');
          setLoading(false);
          return false;
        }
      } else {
        const errorData = await response.json();
        console.error('Login failed:', errorData);
        setLoading(false);
        return false;
      }
    } catch (error) {
      console.error('Login error:', error);
      setLoading(false);
      return false;
    }
  };

  const register = async (userData: RegisterData): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        return true;
      } else {
        const errorData = await response.json();
        console.error('Registration failed:', errorData);
        throw new Error(errorData.detail || 'Registration failed');
      }
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  };

  const logout = async (): Promise<void> => {
    const token = localStorage.getItem('qflare-token');
    
    // Call backend logout endpoint
    if (token) {
      try {
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      } catch (error) {
        console.error('Logout error:', error);
        // Continue with local logout even if backend call fails
      }
    }

    // Clear local storage and state
    setUser(null);
    localStorage.removeItem('qflare-user');
    localStorage.removeItem('qflare-token');
    localStorage.removeItem('qflare-refresh-token');
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    isUser: user?.role === 'user',
    login,
    register,
    logout,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};