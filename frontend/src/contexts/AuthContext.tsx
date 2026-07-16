import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { User, UserRole } from '../types';
import { authApi } from '../api';
import { adminAuthApi } from '../api/modules/auth';

// 认证上下文类型
interface AuthContextType {
    user: User | null;
    isAuthLoading: boolean;
    login: (role: UserRole, authData?: any) => Promise<void>;
    logout: () => Promise<void>;
}

// 创建上下文
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// 根据路由判断当前角色
const getRoleFromPath = (pathname: string): UserRole | null => {
    if (pathname.startsWith('/admin') || pathname.startsWith('/login/admin') || pathname.startsWith('/register/admin')) {
        return UserRole.ADMIN;
    }
    if (pathname.startsWith('/public') || pathname.startsWith('/login/public') || pathname.startsWith('/register/public')) {
        return UserRole.PUBLIC;
    }
    if (pathname.startsWith('/professional') || pathname.startsWith('/login/professional') || pathname.startsWith('/register/professional')) {
        return UserRole.PROFESSIONAL;
    }
    return null;
};

// 根据角色获取 localStorage key 前缀
const getStoragePrefix = (role: UserRole): string => {
    if (role === UserRole.ADMIN) return 'admin_';
    if (role === UserRole.PROFESSIONAL) return 'professional_';
    return 'user_';
};

// 从 localStorage 同步读取初始用户状态（根据当前路由）
const getInitialUser = (pathname: string): User | null => {
    try {
        const role = getRoleFromPath(pathname);
        if (!role) return null;

        const prefix = getStoragePrefix(role);
        const token = localStorage.getItem(`${prefix}access_token`);
        const savedUser = localStorage.getItem(`${prefix}user`);

        if (token && savedUser) {
            return JSON.parse(savedUser);
        }
    } catch {
        // 解析失败，返回 null
    }
    return null;
};

// Provider 组件
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const location = useLocation();
    const navigate = useNavigate();

    // 乐观渲染：立即从 localStorage 同步初始化用户状态
    const [user, setUser] = useState<User | null>(() => getInitialUser(location.pathname));
    const [isAuthLoading, setIsAuthLoading] = useState(false);

    // 当路由变化时，更新用户状态
    useEffect(() => {
        const newUser = getInitialUser(location.pathname);
        // Only update if ID or Role changed, preventing redundant updates
        setUser(prev => {
            if (prev?.id === newUser?.id && prev?.role === newUser?.role) return prev;
            return newUser;
        });
    }, [location.pathname]);

    // 后台静默验证 Token（不阻塞页面渲染）
    useEffect(() => {
        const verifyTokenSilently = async () => {
            // Using user.role directly from state is safer than re-deriving from path
            // if we trust user state is synced correctly by the effect above.
            // However, verifyTokenSilently logic relied on localStorage.
            // Let's keep it robust but prevent re-runs.
            
            const role = user?.role; 
            if (!role) return;

            const prefix = getStoragePrefix(role);
            const token = localStorage.getItem(`${prefix}access_token`);
            // We already have 'user' in state, no need to parse from local storage again just for checking existence
            
            if (token && user) {
                try {
                    // 根据用户角色调用对应的API验证token
                    if (role === UserRole.ADMIN) {
                        await adminAuthApi.me();
                    } else {
                        await authApi.getMe();
                    }
                    // Token 有效，无需操作
                } catch (error: any) {
                    // 只有在明确的 401 错误时才清除认证数据
                    if (error?.response?.status === 401) {
                        console.log('Token已过期或无效，清除登录状态');
                        clearAuthData(role);
                        navigate(role === UserRole.ADMIN ? '/login/admin' : '/', { replace: true });
                    } else {
                        // 网络错误或其他错误，保持登录状态
                        console.log('Token验证请求失败，但保持登录状态', error);
                    }
                }
            }
        };

        // 只有当用户状态存在时才进行后台验证
        if (user) {
            verifyTokenSilently();
        }
    }, [user?.id, user?.role, navigate]); // Removed location.pathname, depend on specific user fields

    // 清除认证数据（只清除指定角色的数据）
    const clearAuthData = (role: UserRole) => {
        const prefix = getStoragePrefix(role);
        localStorage.removeItem(`${prefix}access_token`);
        localStorage.removeItem(`${prefix}refresh_token`);
        localStorage.removeItem(`${prefix}user_id`);
        localStorage.removeItem(`${prefix}user`);

        // 只有当前用户是该角色时才清除状态
        if (user?.role === role) {
            setUser(null);
        }
    };

    // 登录处理：先基于 authData 立即登录 + 跳转,后台再异步补全 /me 详细信息。
    // 这样 /me 失败(网络/CORS/服务端错误)不再阻塞登录流程。
    const login = useCallback(async (role: UserRole, authData?: any) => {
        const routeMap: Record<UserRole, string> = {
            [UserRole.PUBLIC]: '/public',
            [UserRole.PROFESSIONAL]: '/professional',
            [UserRole.ADMIN]: '/admin'
        };
        const prefix = getStoragePrefix(role);

        // 1) 用 authData 中的最基本字段(目前 login 响应只保证 user_id)立即构造用户
        const fallbackUser: User = {
            id: authData?.user_id || 'unknown',
            name: authData?.username || authData?.email || 'User',
            role: role,
            avatar: role === UserRole.PUBLIC
                ? 'https://picsum.photos/id/64/200/200'
                : 'https://picsum.photos/id/55/200/200',
            base_profile: role === UserRole.PUBLIC ? {
                constitution_type: 'Unknown',
                taboo_items: [],
                medical_history: 'None recorded',
                family_history: 'None recorded',
                allergy_info: 'None recorded',
                merged_diseases: 'None recorded',
            } : undefined,
        };

        setUser(fallbackUser);
        localStorage.setItem(`${prefix}user`, JSON.stringify(fallbackUser));
        navigate(routeMap[role]);

        // 2) 后台异步拉取 /me 完整画像,失败时静默回退到 fallbackUser
        try {
            const response = role === UserRole.ADMIN
                ? await adminAuthApi.me()
                : await authApi.getMe();
            const data = response?.data;
            if (!data) return;

            const enrichedUser: User = {
                ...fallbackUser,
                id: data.id || fallbackUser.id,
                name: data.username || data.email || fallbackUser.name,
                avatar: data.avatar_url
                    || (role === UserRole.PROFESSIONAL
                        ? 'https://api.dicebear.com/7.x/avataaars/svg?seed=Felix'
                        : 'https://api.dicebear.com/7.x/avataaars/svg?seed=User'),
                base_profile: data.base_profile || fallbackUser.base_profile,
            };

            // 仅在值发生实质变化时更新,避免无意义的重渲染
            setUser(prev => {
                if (
                    prev?.id === enrichedUser.id &&
                    prev?.name === enrichedUser.name &&
                    prev?.avatar === enrichedUser.avatar
                ) {
                    return prev;
                }
                return enrichedUser;
            });
            localStorage.setItem(`${prefix}user`, JSON.stringify(enrichedUser));
        } catch (err) {
            // /me 失败不影响登录流程,fallbackUser 已就位
            console.warn('[Auth] 后台拉取 /me 失败,使用登录响应中的基础信息:', err);
        }
    }, [navigate]);

    // 登出处理
    const logout = useCallback(async () => {
        const isAdmin = user?.role === UserRole.ADMIN;

        try {
            if (isAdmin) {
                await adminAuthApi.logout();
            } else {
                await authApi.logout();
            }

        } catch {
            console.log('登出时出现问题，但仍已退出登录。');
        }

        if (user) {
            clearAuthData(user.role);
        }
        navigate(isAdmin ? '/login/admin' : '/', { replace: true });
    }, [user, navigate]);

    return (
        <AuthContext.Provider value={{ user, isAuthLoading,  login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

// 自定义 Hook
export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export default AuthContext;
