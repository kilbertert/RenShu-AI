import axios, { AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { convertKeysToSnake } from '../utils/camelToSnakeConverter';

// 根据环境选择API基础URL
const getBaseURL = (): string => {
  if (import.meta.env.PROD) {
    return import.meta.env.VITE_API_BASE_URL || 'https://your-production-server.com';
  }
  return 'http://localhost:8000';
};

// Token 刷新状态管理
let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];

// 添加等待刷新的请求到队列
const subscribeTokenRefresh = (callback: (token: string) => void) => {
  refreshSubscribers.push(callback);
};

// 刷新完成后，通知所有等待的请求
const onTokenRefreshed = (newToken: string) => {
  refreshSubscribers.forEach(callback => callback(newToken));
  refreshSubscribers = [];
};

// 根据当前路由判断应该使用哪个角色的 token
const getTokenPrefix = (): string => {
    const pathname = window.location.pathname;
    if (pathname.startsWith('/admin') || pathname.startsWith('/login/admin') || pathname.startsWith('/register/admin')) {
        return 'admin_';
    }
    if (pathname.startsWith('/professional') || pathname.startsWith('/login/professional') || pathname.startsWith('/register/professional')) {
        return 'professional_';
    }
    return 'user_';
};

// 获取当前 Token
export const getToken = (): string => {
    return localStorage.getItem(`${getTokenPrefix()}access_token`) || '';
};

// 清除认证数据并跳转到登录页
const handleAuthFailure = () => {
    const prefix = getTokenPrefix();
    const savedUser = localStorage.getItem(`${prefix}user`);
    let isAdmin = false;

    try {
        if (savedUser) {
            const user = JSON.parse(savedUser);
            isAdmin = user.role === 'ADMIN';
        }
    } catch {}

    // 清除当前角色的认证数据
    localStorage.removeItem(`${prefix}access_token`);
    localStorage.removeItem(`${prefix}refresh_token`);
    localStorage.removeItem(`${prefix}user_id`);
    localStorage.removeItem(`${prefix}user`);

    // 根据用户类型跳转到对应的登录页
    window.location.href = isAdmin ? '/login/admin' : '/';
};

// 创建axios实例
const request: AxiosInstance = axios.create({
  baseURL: getBaseURL(),
  timeout: 600000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
request.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = getToken();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器
request.interceptors.response.use(
  (response: AxiosResponse) => convertKeysToSnake(response.data),

  async (error) => {
    const originalRequest = error.config;

    // 如果不是 401 错误，直接返回
    if (error.response?.status !== 401) {
      return Promise.reject(error);
    }

    // 如果是刷新 token 的请求失败，直接跳转登录页
    if (originalRequest.url?.includes('/refresh')) {
      handleAuthFailure();
      return Promise.reject(error);
    }

    // 如果已经重试过，不再重试
    if (originalRequest._retry) {
      handleAuthFailure();
      return Promise.reject(error);
    }

    const refreshToken = localStorage.getItem(`${getTokenPrefix()}refresh_token`);
    if (!refreshToken) {
      handleAuthFailure();
      return Promise.reject(error);
    }

    // 标记该请求已重试
    originalRequest._retry = true;

    // 如果正在刷新 token，将请求加入队列等待
    if (isRefreshing) {
      return new Promise((resolve) => {
        subscribeTokenRefresh((newToken: string) => {
          originalRequest.headers.Authorization = `Bearer ${newToken}`;
          resolve(request(originalRequest));
        });
      });
    }

    // 开始刷新 token
    isRefreshing = true;

    try {
      const res = await axios.post(`${getBaseURL()}/api/v1/users/refresh`, {
        refresh_token: refreshToken,
      });

      const { access_token, refresh_token: newRefreshToken } = convertKeysToSnake(res.data).data;

      // 保存新的 token（使用角色前缀）
      const prefix = getTokenPrefix();
      localStorage.setItem(`${prefix}access_token`, access_token);
      localStorage.setItem(`${prefix}refresh_token`, newRefreshToken);

      // 通知所有等待的请求
      onTokenRefreshed(access_token);

      // 重试原请求
      originalRequest.headers.Authorization = `Bearer ${access_token}`;
      return request(originalRequest);

    } catch (refreshError) {
      // 刷新失败，跳转登录页
      handleAuthFailure();
      return Promise.reject(refreshError);
    } finally {
      isRefreshing = false;
    }
  }
);

// ============================================
// 🚀 原生 Fetch SSE 流式请求封装（无自动重连）
// ============================================

export interface StreamRequestConfig {
  url: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  data?: any;
  headers?: Record<string, string>;
  onMessage: (data: any) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
}

/**
 * 原生 Fetch SSE 流式请求封装
 * 
 * 特性：
 * - ✅ 无自动重连（切换窗口不会重发请求）
 * - ✅ 自动携带 Token
 * - ✅ 统一错误处理
 * - ✅ 支持请求取消
 * - ✅ 类型安全
 * - ✅ 长时间流式响应支持（无超时限制）
 */
export const streamRequest = async (config: StreamRequestConfig): Promise<() => void> => {
  const {
    url,
    method = 'POST',
    data,
    headers = {},
    onMessage,
    onError,
    onComplete
  } = config;

  const controller = new AbortController();
  const token = getToken();

  try {
    const response = await fetch(`${getBaseURL()}${url}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '',
        ...headers
      },
      body: data ? JSON.stringify(data) : undefined,
      signal: controller.signal,
      // ✅ 不设置 timeout，允许长时间流式响应
      // keepalive: true,  // 保持连接活跃（可选）
    });

    // 处理 HTTP 错误
    if (!response.ok) {
      if (response.status === 401) {
        handleAuthFailure();
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    // 读取流
    const processStream = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            onComplete?.();
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          
          // 处理 SSE 格式: "data: {...}\n\n"
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // 保留不完整的行

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith(':')) continue; // 跳过空行和注释
            
            if (trimmed.startsWith('data: ')) {
              const dataStr = trimmed.slice(6);
              
              // 结束信号
              if (dataStr === '[DONE]') {
                onComplete?.();
                return;
              }

              try {
                const parsed = JSON.parse(dataStr);
                onMessage(parsed);
              } catch (parseError) {
                console.warn('Failed to parse SSE data:', dataStr);
              }
            }
          }
        }
      } catch (err) {
        // 忽略手动取消的错误
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }
        throw err;
      }
    };

    // 启动流处理（不等待完成）
    processStream().catch((err) => {
      if (err instanceof Error && err.name !== 'AbortError') {
        onError?.(err);
      }
    });

  } catch (error) {
    // 忽略手动取消的错误
    if (error instanceof Error && error.name !== 'AbortError') {
      onError?.(error);
    }
  }

  // 返回取消函数
  return () => controller.abort();
};

export default request;
export { getBaseURL };
