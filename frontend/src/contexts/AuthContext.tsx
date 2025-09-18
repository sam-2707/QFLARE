import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface User {
  id: string;
  username: string;
  role: 'admin' | 'user';
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isUser: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

// API configuration
const API_BASE_URL = 'http://localhost:8000/api';

interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    username: string;
    role: string;
    last_login: string | null;
    is_active: boolean;
  };
  expires_in: number;
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
        setUser(JSON.parse(savedUser));
        // Optionally validate token with backend
        validateToken(savedToken);
      } catch (error) {
        console.error('Error parsing saved user:', error);
        localStorage.removeItem('qflare-user');
        localStorage.removeItem('qflare-token');
      }
    }
    setLoading(false);
  }, []);

  const validateToken = async (token: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/validate-token`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        // Token is invalid, clear stored data
        logout();
      }
    } catch (error) {
      console.error('Token validation error:', error);
      // Network error, keep user logged in but they might need to re-login later
    }
  };

  const login = async (username: string, password: string): Promise<boolean> => {
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: username.toLowerCase(),
          password: password,
        }),
      });

      if (response.ok) {
        const data: LoginResponse = await response.json();
        
        // Create User object from response
        const user: User = {
          id: `${data.user.role}-${Date.now()}`, // Generate ID
          username: data.user.username,
          role: data.user.role as 'admin' | 'user',
          email: `${data.user.username}@qflare.com`,
          name: data.user.role === 'admin' ? 'System Administrator' : 'Standard User'
        };

        setUser(user);
        localStorage.setItem('qflare-user', JSON.stringify(user));
        localStorage.setItem('qflare-token', data.access_token);
        
        setLoading(false);
        return true;
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

  const logout = async () => {
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
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    isUser: user?.role === 'user',
    login,
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