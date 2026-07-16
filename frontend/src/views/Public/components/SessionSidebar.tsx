import React, { useRef, useState } from 'react';
import { Icons } from '../../../components/common/Icons';
import type { User } from '../../../types';
import type { ChatSession } from './shared';
import { groupSessionsByDate } from './shared';

interface SessionSidebarProps {
  user: User;
  sessions: ChatSession[];
  activeSessionId: string;
  showLeftSidebar: boolean;
  isDarkMode: boolean;
  onCloseSidebar: () => void;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (sessionId: string, e: React.MouseEvent) => void;
  onToggleTheme: () => void;
  onEditProfile: () => void;
  onOpenModelManagement: () => void;
  onLogout: () => void;
}

const SessionSidebarInner: React.FC<SessionSidebarProps> = ({
  user,
  sessions,
  activeSessionId,
  showLeftSidebar,
  isDarkMode,
  onCloseSidebar,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onToggleTheme,
  onEditProfile,
  onOpenModelManagement,
  onLogout,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchActive, setIsSearchActive] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const filteredSessions = sessions.filter(s =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const groupSessions = groupSessionsByDate(filteredSessions);

  return (
    <aside
      className={`${
        showLeftSidebar ? 'w-[280px]' : 'w-0'
      } flex-shrink-0 bg-[#f0f4f9]/80 dark:bg-[#1e1e1e]/80 backdrop-blur-md border-r border-tcm-lightGreen/10 flex flex-col transition-all duration-300 overflow-hidden relative z-30`}
    >
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between p-4 pb-2">
          <button
            onClick={onCloseSidebar}
            className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-white/10 rounded-full transition-colors"
          >
            <Icons.Menu size={20} />
          </button>

          <button
            onClick={() => setIsSearchActive(!isSearchActive)}
            className={`p-2 rounded-full transition-colors ${
              isSearchActive
                ? 'bg-gray-200 dark:bg-white/10 text-gray-900 dark:text-white'
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-white/10'
            }`}
          >
            <Icons.Stethoscope className="rotate-90" size={20} />
          </button>
        </div>

        <div className="px-4 py-2">
          {isSearchActive ? (
            <div className="relative animate-in fade-in slide-in-from-top-1">
              <input
                ref={searchInputRef}
                type="text"
                placeholder="搜索历史会话..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full bg-white dark:bg-black/20 border border-transparent dark:border-white/10 rounded-full py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-gray-200 dark:focus:ring-gray-700 shadow-sm"
              />
              <div className="absolute left-3 top-2.5 text-gray-500">
                <Icons.Activity size={16} />
              </div>
            </div>
          ) : (
            <button
              onClick={onNewChat}
              className="flex items-center gap-3 px-4 py-3 bg-[#dde3ea] dark:bg-[#2a2a2a] text-gray-700 dark:text-gray-200 rounded-2xl hover:bg-[#d0d7de] dark:hover:bg-[#333] transition-colors w-full shadow-sm"
            >
              <Icons.Edit3 size={18} />
              <span className="text-sm font-medium">开启新诊疗</span>
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-2 custom-scrollbar space-y-4">
          {Object.entries(groupSessions).map(
            ([group, groupSessions]) =>
              groupSessions.length > 0 && (
                <div key={group} className="animate-in fade-in">
                  <div className="px-4 py-2 text-xs font-bold text-gray-500 dark:text-gray-400">
                    {group}
                  </div>
                  {groupSessions.map(session => (
                    <div
                      key={session.id}
                      onClick={() => onSelectSession(session.id)}
                      className={`group flex items-center gap-3 px-4 py-2 mx-2 rounded-full cursor-pointer transition-colors relative ${
                        activeSessionId === session.id
                          ? 'bg-tcm-lightGreen/20 dark:bg-tcm-lightGreen/10 text-tcm-darkGreen dark:text-tcm-freshGreen font-medium'
                          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-white/5'
                      }`}
                    >
                      <Icons.MessageSquare size={16} className="flex-shrink-0 opacity-70" />
                      <div className="flex-1 min-w-0 text-sm truncate pr-6">{session.title}</div>
                      <button
                        onClick={e => onDeleteSession(session.id, e)}
                        className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
                        title="删除会话"
                      >
                        <Icons.Trash2 size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              )
          )}
        </div>

        <div className="p-2 border-t border-gray-200 dark:border-white/5 relative bg-white/50 dark:bg-black/20">
          {showUserMenu && (
            <div className="absolute bottom-full left-2 right-2 mb-2 bg-[#f0f4f9] dark:bg-[#1e1e1e] rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 overflow-hidden animate-in slide-in-from-bottom-2 z-40">
              <button
                onClick={() => {
                  onEditProfile();
                  setShowUserMenu(false);
                }}
                className="w-full text-left px-4 py-3 hover:bg-gray-200 dark:hover:bg-white/10 flex items-center gap-3 text-sm text-gray-700 dark:text-gray-200 transition-colors"
              >
                <Icons.Edit3 size={16} className="text-tcm-lightGreen" />
                个人资料设置
              </button>
              <button
                onClick={() => {
                  onOpenModelManagement();
                  setShowUserMenu(false);
                }}
                className="w-full text-left px-4 py-3 hover:bg-gray-200 dark:hover:bg-white/10 flex items-center gap-3 text-sm text-gray-700 dark:text-gray-200 transition-colors"
              >
                <Icons.Settings size={16} className="text-tcm-gold" />
                模型管理配置
              </button>
              <button
                onClick={onToggleTheme}
                className="w-full text-left px-4 py-3 hover:bg-gray-200 dark:hover:bg-white/10 flex items-center gap-3 text-sm text-gray-700 dark:text-gray-200 transition-colors"
              >
                {isDarkMode ? (
                  <Icons.Sun size={16} className="text-yellow-500" />
                ) : (
                  <Icons.Moon size={16} className="text-indigo-400" />
                )}
                {isDarkMode ? '切换到浅色模式' : '切换到深色模式'}
              </button>
              <button
                onClick={() => {
                  onLogout();
                  setShowUserMenu(false);
                }}
                className="w-full text-left px-4 py-3 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-3 text-sm text-red-600 dark:text-red-400 border-t border-gray-200 dark:border-gray-700 transition-colors"
              >
                <Icons.LogOut size={16} />
                退出账号
              </button>
            </div>
          )}

          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className={`flex items-center gap-3 w-full p-3 rounded-full hover:bg-gray-200 dark:hover:bg-white/5 transition-colors ${
              showUserMenu ? 'bg-gray-200 dark:bg-white/5' : ''
            }`}
          >
            <div className="w-8 h-8 rounded-full bg-tcm-darkGreen text-white flex items-center justify-center text-xs font-bold shadow-inner">
              {user.name.charAt(0)}
            </div>
            <div className="flex-1 text-left text-sm font-medium text-gray-700 dark:text-gray-200 truncate">
              {user.name}
            </div>
            <Icons.Settings size={18} className="text-gray-500" />
          </button>
        </div>
      </div>
    </aside>
  );
};

export const SessionSidebar = React.memo(SessionSidebarInner);
SessionSidebar.displayName = 'SessionSidebar';
