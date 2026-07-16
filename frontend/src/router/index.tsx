import React, { Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { UserRole } from '../types';
import { useAuth } from '../contexts/AuthContext';
import ProtectedRoute from './ProtectedRoute';

// 首屏必需页面：静态导入，避免首屏 loading 闪烁
import LandingPage from '../views/home/LandingPage';
import PublicLoginPage from '../views/public/PublicLoginPage';
import ProfessionalLoginPage from '../views/professional/ProfessionalLoginPage';
import PublicRegisterPage from '../views/public/PublicRegisterPage';
import ProfessionalRegisterPage from '../views/professional/ProfessionalRegisterPage';
import AdminLoginPage from '../views/admin/AdminLoginPage';
import AdminRegisterPage from '../views/admin/AdminRegisterPage';

// Portal 页面：eager 导入以避开 React 18.3.1 + Vite react-refresh 在
// lazy 边界 + useLayoutEffect 直写 documentElement 场景下的 removeChild bug
import PublicPortal from '../views/Public/PublicPortal';
import ProfessionalPortal from '../views/professional/ProfessionalPortal';
import AdminPortal from '../views/admin/AdminPortal';
import PublicModelManagementPage from '../views/public/PublicModelManagementPage';

// 路由切换 loading 占位（与品牌色协调）
const RouteLoading: React.FC = () => (
    <div className="h-screen w-full flex flex-col items-center justify-center bg-tcm-cream">
        <div className="w-12 h-12 rounded-full border-4 border-tcm-freshGreen border-t-tcm-lightGreen animate-spin mb-4" />
        <p className="text-tcm-darkGreen font-serif-sc text-sm tracking-widest opacity-70">加载中…</p>
    </div>
);


const AppContent: React.FC = () => {
    const { user, login, logout } = useAuth();
    const handleRegisterSuccess = () => {
        console.log('Registration successful');
    };

    return (
        <div className="h-screen w-full bg-tcm-cream text-tcm-charcoal overflow-hidden selection:bg-tcm-lightGreen selection:text-white relative">

            <Routes>
                {/* 公开路由 */}
                <Route path="/" element={<LandingPage />} />

                {/* 登录路由 */}
                <Route path="/login/public" element={<PublicLoginPage onLogin={login} />} />
                <Route path="/login/professional" element={<ProfessionalLoginPage onLogin={login} />} />
                <Route path="/login/admin" element={<AdminLoginPage onLogin={login} />} />
                <Route path="/login" element={<Navigate to="/login/public" />} />

                {/* 注册路由 */}
                <Route path="/register/public" element={<PublicRegisterPage onRegisterSuccess={handleRegisterSuccess} />} />
                <Route path="/register/professional" element={<ProfessionalRegisterPage onRegisterSuccess={handleRegisterSuccess} />} />
                <Route path="/register/admin" element={<AdminRegisterPage onRegister={handleRegisterSuccess} />} />

                {/* 受保护的路由 - Public */}
                <Route
                    path="/public"
                    element={
                        <ProtectedRoute allowedRoles={[UserRole.PUBLIC]} redirectTo="/">
                            <Suspense fallback={<RouteLoading />}>
                                <PublicPortal user={user!} onLogout={logout} />
                            </Suspense>
                        </ProtectedRoute>
                    }
                />
                  {/* 新增模型管理路由 */}
                <Route
                    path="/public/models"
                    element={
                        <ProtectedRoute allowedRoles={[UserRole.PUBLIC]} redirectTo="/">
                            <Suspense fallback={<RouteLoading />}>
                                <PublicModelManagementPage />
                            </Suspense>
                        </ProtectedRoute>
                    }
                />

                {/* 受保护的路由 - Professional */}
                <Route
                    path="/professional"
                    element={
                        <ProtectedRoute allowedRoles={[UserRole.PROFESSIONAL]} redirectTo="/">
                            <Suspense fallback={<RouteLoading />}>
                                <ProfessionalPortal user={user!} onLogout={logout} />
                            </Suspense>
                        </ProtectedRoute>
                    }
                />

                {/* 受保护的路由 - Admin */}
                <Route
                    path="/admin"
                    element={
                        <ProtectedRoute allowedRoles={[UserRole.ADMIN]} redirectTo="/login/admin">
                            <Suspense fallback={<RouteLoading />}>
                                <AdminPortal user={user!} onLogout={logout} />
                            </Suspense>
                        </ProtectedRoute>
                    }
                />
            </Routes>
        </div>
    );
};

export default AppContent;
