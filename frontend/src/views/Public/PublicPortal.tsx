import React, {
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
  useMemo,
  useCallback,
} from 'react';
import { useNavigate } from 'react-router-dom';
import {
  User,
  ChatMessage,
  UserPersona,
  AIModelConfig,
  ProviderConfig,
  CustomModel,
} from '../../types';
import { AVAILABLE_MODELS, PROVIDERS } from '../../constants';
import { Icons } from '../../components/common/Icons';
import { LogoutConfirmModal } from '../../components/common/LogoutConfirmModal';
import { AlertModal } from '../../components/common/AlertModal';
import { v4 as uuidv4 } from 'uuid';
import { providerApi } from '../../api/modules/model';
import { chatApi } from '../../api/modules/chat';
import { conversationApi } from '../../api/modules/conversation';
import { ModelSelector, type RightPanelMode } from './components/ModelSelector';
import { SessionSidebar } from './components/SessionSidebar';
import { ChatPanel } from './components/ChatPanel';
import { SettingsPanel } from './components/SettingsPanel';
import HealthProfile from './components/HealthProfile';
import { type ChatSession, type Attachment, buildDefaultPersona } from './components/shared';

interface PublicPortalProps {
  user: User;
  onLogout: () => void;
}

const PublicPortal: React.FC<PublicPortalProps> = ({ user, onLogout }) => {
  const navigate = useNavigate();

  // ====== 跨子组件共享状态 (父管) ======
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>(
    () => localStorage.getItem('last_active_session_id') || ''
  );
  const [showLeftSidebar, setShowLeftSidebar] = useState(true);
  const [showRightSidebar, setShowRightSidebar] = useState(true);
  const [rightPanelMode, setRightPanelMode] = useState<RightPanelMode>('health');

  const [apiData, setApiData] = useState<any[]>([]);
  const [providerConfigs, setProviderConfigs] = useState<Record<string, ProviderConfig>>(() => {
    const saved = localStorage.getItem('user_provider_configs');
    return saved ? JSON.parse(saved) : {};
  });
  const [customModels, setCustomModels] = useState<CustomModel[]>(() => {
    const saved = localStorage.getItem('user_custom_models');
    return saved ? JSON.parse(saved) : [];
  });

  const [selectedProviderId, setSelectedProviderId] = useState<string>(
    () => localStorage.getItem('last_selected_provider') || 'google'
  );
  const [selectedModel, setSelectedModel] = useState<AIModelConfig>(
    AVAILABLE_MODELS[0]
  );
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [enableThinking, setEnableThinking] = useState(false);

  const [sessionLoadingStates, setSessionLoadingStates] = useState<Record<string, boolean>>({});

  // Modal 状态 (父管)
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [editPersonaForm, setEditPersonaForm] = useState<UserPersona>(
    buildDefaultPersona(user.base_profile)
  );
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [alertConfig, setAlertConfig] = useState({ title: '', description: '' });
  const [confirmAction, setConfirmAction] = useState<{
    isOpen: boolean;
    type: 'deleteSession' | 'deleteMessage' | null;
    id: string | null;
    title: string;
    description: string;
  }>({
    isOpen: false,
    type: null,
    id: null,
    title: '',
    description: '',
  });
  const [isQuickConfigOpen, setIsQuickConfigOpen] = useState(false);
  const [changedFields, setChangedFields] = useState<string[]>([]);

  // 主题
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme_public');
    return saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  // Refs (父管 - 跨组件可见)
  const isInitializedRef = useRef(false);
  const isProviderFetchedRef = useRef(false);
  const sessionsRef = useRef(sessions);
  const fetchedSessionIds = useRef<Set<string>>(new Set());
  const globalSendingLockRef = useRef(false);
  const isSendingRef = useRef<Record<string, boolean>>({});
  const lastSendTimeRef = useRef<number>(0);
  const cancelStreamRef = useRef<(() => void) | null>(null);

  // ====== 派生数据 ======
  const currentSession = useMemo(
    () => sessions.find(s => s.id === activeSessionId) || null,
    [sessions, activeSessionId]
  );

  const isCurrentSessionLoading = useMemo(
    () => (activeSessionId ? !!sessionLoadingStates[activeSessionId] : false),
    [activeSessionId, sessionLoadingStates]
  );

  const defaultPersona = useMemo(
    () => buildDefaultPersona(user.base_profile),
    [user.base_profile]
  );

  const persona = useMemo(
    () => (currentSession?.persona ? currentSession.persona : defaultPersona),
    [currentSession, defaultPersona]
  );

  const healthScore = useMemo(() => persona?.health_score || 85, [persona]);

  const allModels = useMemo<AIModelConfig[]>(() => {
    if (apiData.length > 0) {
      const models: AIModelConfig[] = [];
      apiData.forEach((p: any) => {
        if (p.models) {
          p.models.forEach((m: any) => {
            models.push({
              id: m.model_name,
              realId: m.id,
              modelName: m.model_name,
              providerId: p.id,
              name: m.label || m.model_name,
              description: m.description || '',
              supportsThinking: m.features?.includes('thinking') || false,
              supportsVision: m.features?.includes('vision') || false,
              supportsToolCall: m.features?.includes('tool_call') || false,
              provider: p.name as any,
              contextWindow: m.context_window ? `${Math.round(m.context_window / 1000)}K` : undefined,
              defaultTemperature: m.default_temperature,
              defaultTopP: m.default_top_p,
              defaultMaxTokens: m.default_max_tokens,
              isEnabled: m.is_enabled,
              isBuiltin: m.is_builtin,
            });
          });
        }
      });
      return models;
    }
    return [...AVAILABLE_MODELS, ...customModels];
  }, [apiData, customModels]);

  const currentProviders = useMemo(() => {
    if (apiData.length > 0) {
      return apiData
        .filter((p: any) => p.is_enabled !== false)
        .map((p: any) => ({
          id: p.name,
          name: p.label || p.name,
          icon: p.icon || '🤖',
          isBuiltin: p.is_builtin,
        }));
    }
    return PROVIDERS.map(p => ({ ...p, isBuiltin: true }));
  }, [apiData]);

  const filteredModels = useMemo(
    () => allModels.filter(m => m.provider === selectedProviderId && m.isEnabled !== false),
    [selectedProviderId, allModels]
  );

  // ====== Effects ======

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme_public', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme_public', 'light');
    }
  }, [isDarkMode]);

  useEffect(() => {
    if (selectedProviderId) {
      localStorage.setItem('last_selected_provider', selectedProviderId);
    }
  }, [selectedProviderId]);

  useEffect(() => {
    if (selectedModel && selectedModel.id) {
      localStorage.setItem('last_selected_model_id', selectedModel.id);
    }
  }, [selectedModel]);

  useEffect(() => {
    if (activeSessionId) {
      localStorage.setItem('last_active_session_id', activeSessionId);
    } else {
      localStorage.removeItem('last_active_session_id');
    }
  }, [activeSessionId]);

  // 拉取 providers + models
  useEffect(() => {
    if (isProviderFetchedRef.current) return;
    isProviderFetchedRef.current = true;

    const fetchData = async () => {
      try {
        const response = await providerApi.get_providers_filtered('all');
        if (response.success && response.data) {
          setApiData(response.data);
          setProviderConfigs(prev => {
            const next = { ...prev };
            let changed = false;
            response.data.forEach((p: any) => {
              if (p.api_key && (!next[p.name]?.apiKey)) {
                next[p.name] = { ...next[p.name], apiKey: 'CONFIGURED_IN_BACKEND', enabled: true };
                changed = true;
              }
            });
            return changed ? next : prev;
          });
          if (response.data.length > 0) {
            const currentExists = response.data.some((p: any) => p.name === selectedProviderId);
            if (!currentExists) {
              const defaultProvider =
                response.data.find((p: any) => p.name === 'google') || response.data[0];
              setSelectedProviderId(defaultProvider.name);
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch models config:', error);
      }
    };
    fetchData();
  }, []);

  // 拉取会话列表
  useEffect(() => {
    if (isInitializedRef.current) return;
    isInitializedRef.current = true;

    const fetchConversations = async () => {
      try {
        const res = await conversationApi.getConversations();
        if (res.success && res.data && res.data.length > 0) {
          const backendSessions: ChatSession[] = res.data.map((c: any) => ({
            id: c.id,
            title: c.title || '无标题对话',
            messages: [],
            persona: c.session_metadata,
            lastModified: (() => {
              const date = c.updated_at ? new Date(c.updated_at) : new Date();
              return isNaN(date.getTime()) ? new Date() : date;
            })(),
          }));
          setSessions(backendSessions);
          setActiveSessionId(prev => prev || '');
        } else {
          setSessions([]);
          setActiveSessionId(prev => prev || '');
        }
      } catch (e) {
        console.error('Failed to fetch conversations', e);
        setSessions([]);
        setActiveSessionId(prev => prev || '');
      }
    };
    fetchConversations();
  }, []);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  useEffect(() => {
    const handleStorageChange = () => {
      const savedProviderConfigs = localStorage.getItem('user_provider_configs');
      if (savedProviderConfigs) setProviderConfigs(JSON.parse(savedProviderConfigs));
      const savedCustomModels = localStorage.getItem('user_custom_models');
      if (savedCustomModels) setCustomModels(JSON.parse(savedCustomModels));
    };
    window.addEventListener('storage', handleStorageChange);
    handleStorageChange();
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // active session 变化时拉取消息
  useEffect(() => {
    const fetchMessages = async () => {
      const session = sessionsRef.current.find(s => s.id === activeSessionId);
      if (!session || session.messages.length > 0 || fetchedSessionIds.current.has(activeSessionId)) {
        return;
      }
      fetchedSessionIds.current.add(activeSessionId);

      try {
        const res = await conversationApi.getMessages(activeSessionId);
        if (res.success && res.data) {
          const messages: ChatMessage[] = res.data.map((m: any) => ({
            id: m.id,
            role: m.role === 'assistant' ? 'model' : 'user',
            text: m.content,
            timestamp: new Date(m.created_at),
            attachments: (() => {
              if (!m.message_metadata) return [];
              try {
                const meta =
                  typeof m.message_metadata === 'string' ? JSON.parse(m.message_metadata) : m.message_metadata;
                return meta.attachments || [];
              } catch (e) {
                return [];
              }
            })(),
          }));
          setSessions(prev => prev.map(s => (s.id === activeSessionId ? { ...s, messages } : s)));
        }
      } catch (e) {
        console.error('Failed to fetch messages for session', activeSessionId, e);
      }
    };
    if (activeSessionId) fetchMessages();
  }, [activeSessionId]);

  // 同步 selectedModel + filteredModels
  useEffect(() => {
    const isSelectedValid = filteredModels.some(m => m.id === selectedModel.id);
    if (isSelectedValid) {
      const currentInFiltered = filteredModels.find(m => m.id === selectedModel.id);
      if (
        currentInFiltered &&
        (currentInFiltered.realId !== selectedModel.realId ||
          currentInFiltered.providerId !== selectedModel.providerId)
      ) {
        setSelectedModel(currentInFiltered);
      }
    } else {
      const lastModelId = localStorage.getItem('last_selected_model_id');
      const lastModel = lastModelId ? filteredModels.find(m => m.id === lastModelId) : undefined;
      if (lastModel) {
        setSelectedModel(lastModel);
      } else if (filteredModels.length > 0) {
        setSelectedModel(filteredModels[0]);
      } else if (selectedModel.id !== '') {
        setSelectedModel({
          id: '',
          name: '无可用模型',
          description: '该提供商下暂无可用模型',
          provider: selectedProviderId,
          supportsThinking: false,
          supportsVision: false,
          supportsToolCall: false,
          isEnabled: false,
          isBuiltin: false,
        });
      }
    }
  }, [filteredModels, selectedProviderId]);

  // 同步 editPersonaForm 当 persona 变化
  useEffect(() => {
    setEditPersonaForm(persona);
  }, [persona]);

  // 页面可见性保护
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        console.log('📱 [页面状态] 页面不可见,用户切换到其他标签/应用');
      } else {
        console.log('📱 [页面状态] 页面重新可见,用户切换回来');
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  // ====== Handlers (useCallback 包裹以配合 memo) ======

  const handleToggleTheme = useCallback(() => setIsDarkMode(v => !v), []);

  const handleProviderChange = useCallback((providerId: string) => {
    setSelectedProviderId(providerId);
  }, []);

  const handleModelSelect = useCallback((model: AIModelConfig) => {
    setSelectedModel(model);
    setTemperature(model.defaultTemperature ?? 0.7);
    setTopP(model.defaultTopP ?? 1.0);
    setMaxTokens(model.defaultMaxTokens ?? 4096);
  }, []);

  const handleOpenModelSettings = useCallback(
    (model: AIModelConfig) => {
      if (selectedModel.id !== model.id) {
        setSelectedModel(model);
        setTemperature(model.defaultTemperature ?? 0.7);
        setTopP(model.defaultTopP ?? 1.0);
        setMaxTokens(model.defaultMaxTokens ?? 4096);
      }
      setRightPanelMode('settings');
      setShowRightSidebar(true);
    },
    [selectedModel.id]
  );

  const handleResetDefaults = useCallback(() => {
    setTemperature(selectedModel.defaultTemperature ?? 0.7);
    setTopP(selectedModel.defaultTopP ?? 1.0);
    setMaxTokens(selectedModel.defaultMaxTokens ?? 4096);
  }, [selectedModel]);

  const handleTemperatureChange = useCallback((v: number) => setTemperature(v), []);
  const handleTopPChange = useCallback((v: number) => setTopP(v), []);
  const handleMaxTokensChange = useCallback((v: number) => setMaxTokens(v), []);
  const handleToggleThinking = useCallback(() => setEnableThinking(v => !v), []);

  const handleSetRightPanelMode = useCallback((mode: RightPanelMode) => {
    setRightPanelMode(mode);
  }, []);

  const handleToggleRightSidebar = useCallback(() => setShowRightSidebar(v => !v), []);
  const handleCloseRightSidebar = useCallback(() => setShowRightSidebar(false), []);
  const handleCloseLeftSidebar = useCallback(() => setShowLeftSidebar(false), []);
  const handleOpenLeftSidebar = useCallback(() => setShowLeftSidebar(true), []);

  const handleOpenHistory = useCallback(() => {
    setRightPanelMode('history');
    setShowRightSidebar(true);
  }, []);

  const handleNewChat = useCallback(() => {
    setActiveSessionId('');
    if (window.innerWidth < 768) setShowLeftSidebar(false);
  }, []);

  const handleSelectSession = useCallback((id: string) => {
    setActiveSessionId(id);
  }, []);

  const handleDeleteSession = useCallback((sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmAction({
      isOpen: true,
      type: 'deleteSession',
      id: sessionId,
      title: '确认删除会话',
      description: '确定要删除这个会话吗?此操作不可恢复。',
    });
  }, []);

  const handleDeleteMessage = useCallback((messageId: string) => {
    setConfirmAction({
      isOpen: true,
      type: 'deleteMessage',
      id: messageId,
      title: '确认删除消息',
      description: '确定要删除这条消息吗?',
    });
  }, []);

  const handleConfirmAction = useCallback(async () => {
    const { type, id } = confirmAction;
    setConfirmAction(prev => ({ ...prev, isOpen: false }));
    if (type === 'deleteSession' && id) {
      try {
        const res = await conversationApi.deleteConversation(id);
        if (res.success) {
          const next = sessions.filter(s => s.id !== id);
          setSessions(next);
          if (next.length > 0) {
            if (activeSessionId === id) setActiveSessionId(next[0].id);
          } else {
            setActiveSessionId('');
          }
        }
      } catch (error) {
        console.error('Failed to delete session', error);
        setAlertConfig({ title: '删除失败', description: '删除会话时发生错误,请稍后重试。' });
        setShowAlertModal(true);
      }
    } else if (type === 'deleteMessage' && id) {
      try {
        const res = await conversationApi.deleteMessage(id);
        if (res.success) {
          const freshRes = await conversationApi.getMessages(activeSessionId);
          if (freshRes.success && freshRes.data) {
            const messages: ChatMessage[] = freshRes.data.map((m: any) => ({
              id: m.id,
              role: m.role === 'assistant' ? 'model' : 'user',
              text: m.content,
              timestamp: new Date(m.created_at),
              attachments: (() => {
                if (!m.message_metadata) return [];
                try {
                  const meta =
                    typeof m.message_metadata === 'string' ? JSON.parse(m.message_metadata) : m.message_metadata;
                  return meta.attachments || [];
                } catch (e) {
                  return [];
                }
              })(),
            }));
            setSessions(prev => prev.map(s => (s.id === activeSessionId ? { ...s, messages } : s)));
          }
        }
      } catch (error) {
        console.error('Failed to delete message', error);
        setAlertConfig({ title: '删除失败', description: '删除消息时发生错误,请稍后重试。' });
        setShowAlertModal(true);
      }
    }
  }, [confirmAction, sessions, activeSessionId]);

  const handleOpenEditProfile = useCallback(() => setIsEditingProfile(true), []);
  const handleCloseEditProfile = useCallback(() => setIsEditingProfile(false), []);
  const handleOpenModelManagement = useCallback(() => navigate('/public/models'), [navigate]);
  const handleLogoutClick = useCallback(() => setShowLogoutModal(true), []);
  const handleLogoutConfirm = useCallback(() => {
    setShowLogoutModal(false);
    onLogout();
  }, [onLogout]);

  const handleSaveProfile = useCallback(() => {
    if (activeSessionId) {
      setSessions(prev =>
        prev.map(s => (s.id === activeSessionId ? { ...s, persona: editPersonaForm } : s))
      );
    }
    setIsEditingProfile(false);
  }, [activeSessionId, editPersonaForm]);

  // 更新单个 session 的 persona (供 SSE 完成时使用)
  const updateSessionPersona = useCallback(
    (sessionId: string, newPersona: UserPersona, newScore?: number) => {
      setSessions(prev =>
        prev.map(s =>
          s.id === sessionId
            ? { ...s, persona: newPersona, healthScore: newScore !== undefined ? newScore : s.healthScore }
            : s
        )
      );
    },
    []
  );

  const updateSessionMessages = useCallback((sessionId: string, newMessages: ChatMessage[]) => {
    setSessions(prev =>
      prev.map(s => (s.id === sessionId ? { ...s, messages: newMessages, lastModified: new Date() } : s))
    );
  }, []);

  // SSE 取消
  const handleCancelGeneration = useCallback(() => {
    if (cancelStreamRef.current) {
      cancelStreamRef.current();
      cancelStreamRef.current = null;
      globalSendingLockRef.current = false;
      Object.keys(isSendingRef.current).forEach(key => {
        isSendingRef.current[key] = false;
      });
      setSessionLoadingStates({});
      if (activeSessionId) {
        setSessions(prev =>
          prev.map(s => {
            if (s.id === activeSessionId) {
              const lastMsg = s.messages[s.messages.length - 1];
              if (lastMsg && lastMsg.role === 'model' && !lastMsg.text) {
                const updatedMessages = s.messages.map((m, idx) =>
                  idx === s.messages.length - 1 ? { ...m, text: '*[对话已取消]*' } : m
                );
                return { ...s, messages: updatedMessages };
              }
            }
            return s;
          })
        );
      }
    }
  }, [activeSessionId]);

  // 实际发送 (内部使用,负责 SSE + persona 分析)
  const handleSendMessage = useCallback(
    async (overrideText?: string) => {
      console.log('🚀 [handleSendMessage] 被调用');
      if (globalSendingLockRef.current) {
        console.log('⚠️ [全局锁] 有请求正在进行中,忽略重复提交');
        return;
      }
      const now = Date.now();
      if (now - lastSendTimeRef.current < 2000) {
        console.log('⚠️ [时间戳] 请求间隔过短,忽略重复提交');
        return;
      }
      const currentSessionKey = activeSessionId || 'new';
      if (isSendingRef.current[currentSessionKey]) {
        console.log('⚠️ [会话锁] 该会话请求已在进行中,忽略重复提交');
        return;
      }
      globalSendingLockRef.current = true;
      lastSendTimeRef.current = now;
      isSendingRef.current[currentSessionKey] = true;

      // 中断恢复
      const currentSess = sessions.find(s => s.id === activeSessionId);
      if (currentSess?.isInterrupted && currentSess?.threadId) {
        const textForResume = overrideText ?? '';
        if (!textForResume.trim()) {
          globalSendingLockRef.current = false;
          isSendingRef.current[currentSessionKey] = false;
          return;
        }
        await handleResumeMessage(textForResume, currentSess);
        return;
      }

      if (!selectedModel.id) {
        globalSendingLockRef.current = false;
        isSendingRef.current[currentSessionKey] = false;
        setAlertConfig({
          title: '未选择有效模型',
          description: '当前未选择有效的模型配置。请先切换到有可用模型的提供商,或在模型管理中启用模型。',
        });
        setShowAlertModal(true);
        return;
      }

      const textToSend = overrideText ?? '';
      if (!textToSend.trim()) {
        globalSendingLockRef.current = false;
        isSendingRef.current[currentSessionKey] = false;
        return;
      }

      // 检查 API key
      const providerConfig = providerConfigs[selectedModel.provider];
      const isGoogle = selectedModel.provider === 'google';
      if (!isGoogle && !providerConfig?.apiKey) {
        globalSendingLockRef.current = false;
        isSendingRef.current[currentSessionKey] = false;
        setIsQuickConfigOpen(true);
        return;
      }

      let targetSessionId = activeSessionId;
      let isNewSession = false;

      if (!targetSessionId) {
        targetSessionId = uuidv4();
        isNewSession = true;
        const newSession: ChatSession = {
          id: targetSessionId,
          title: textToSend.slice(0, 20) || '新的对话',
          messages: [],
          lastModified: new Date(),
          persona: defaultPersona,
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
      }

      const userMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'user',
        text: textToSend,
        timestamp: new Date(),
      };

      // Optimistic UI update
      if (isNewSession) {
        updateSessionMessages(targetSessionId, [userMsg]);
      } else {
        const existing = sessions.find(s => s.id === targetSessionId)?.messages || [];
        updateSessionMessages(targetSessionId, [...existing, userMsg]);
      }

      if (isNewSession || (currentSession?.messages.length === 0)) {
        const newTitle = textToSend.slice(0, 20) + (textToSend.length > 20 ? '...' : '');
        setSessions(prev =>
          prev.map(s => (s.id === targetSessionId ? { ...s, title: newTitle } : s))
        );
      }

      setSessionLoadingStates(prev => ({ ...prev, [targetSessionId]: true }));

      const history = isNewSession
        ? []
        : (currentSession?.messages || []).map(m => ({
            role: m.role === 'model' ? 'assistant' : 'user',
            content: m.text,
          }));
      history.push({ role: 'user', content: textToSend });

      // === 流式生成 ===
      (async () => {
        const modelMsgId = (Date.now() + 1).toString();
        const modelMsg: ChatMessage = {
          id: modelMsgId,
          role: 'model',
          text: '',
          timestamp: new Date(),
        };
        setSessions(prev => {
          const session = prev.find(s => s.id === targetSessionId);
          if (session) {
            const newMsgs = [...session.messages, modelMsg];
            return prev.map(s =>
              s.id === targetSessionId ? { ...s, messages: newMsgs, lastModified: new Date() } : s
            );
          }
          return prev;
        });

        let accumulatedText = '';
        let currentSteps: string[] = [];

        try {
          const cancelStream = await chatApi.generateStream(
            {
              user_id: user.id,
              conversation_id: targetSessionId,
              query: textToSend,
              model_configuration: {
                provider_id: selectedModel.providerId || '',
                model_id: selectedModel.realId || '',
                model_name: selectedModel.modelName || selectedModel.id,
                temperature,
                top_p: topP,
                max_tokens: maxTokens,
              },
              stream: true,
              enable_thinking: enableThinking,
            },
            (data: any) => {
              if (data.type === 'thread_init' && data.thread_id) {
                setSessions(prev =>
                  prev.map(s =>
                    s.id === targetSessionId ? { ...s, threadId: data.thread_id } : s
                  )
                );
                return;
              }
              if (data.type === 'interrupt') {
                accumulatedText = data.question || '';
                setSessions(prev =>
                  prev.map(s =>
                    s.id === targetSessionId
                      ? { ...s, threadId: data.thread_id || s.threadId, isInterrupted: true }
                      : s
                  )
                );
                setSessions(prev => {
                  const session = prev.find(s => s.id === targetSessionId);
                  if (session) {
                    const updatedMessages = session.messages.map(m =>
                      m.id === modelMsgId
                        ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] }
                        : m
                    );
                    return prev.map(s =>
                      s.id === targetSessionId ? { ...s, messages: updatedMessages } : s
                    );
                  }
                  return prev;
                });
                globalSendingLockRef.current = false;
                isSendingRef.current[targetSessionId] = false;
                isSendingRef.current['new'] = false;
                cancelStreamRef.current = null;
                setSessionLoadingStates(prev => {
                  const next = { ...prev };
                  delete next[targetSessionId];
                  return next;
                });
                return;
              }
              if (data.type === 'done') {
                setSessions(prev => {
                  const session = prev.find(s => s.id === targetSessionId);
                  if (session) {
                    const updatedMessages = session.messages.map(m =>
                      m.id === modelMsgId ? { ...m, queryType: data.query_type } : m
                    );
                    return prev.map(s =>
                      s.id === targetSessionId ? { ...s, messages: updatedMessages } : s
                    );
                  }
                  return prev;
                });
                return;
              }
              if (data.type === 'content') {
                accumulatedText += data.content || '';
              } else if (data.type && data.type !== 'error') {
                currentSteps = [...currentSteps, data.type];
              }
              setSessions(prev => {
                const session = prev.find(s => s.id === targetSessionId);
                if (session) {
                  const updatedMessages = session.messages.map(m =>
                    m.id === modelMsgId
                      ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] }
                      : m
                  );
                  return prev.map(s =>
                    s.id === targetSessionId ? { ...s, messages: updatedMessages } : s
                  );
                }
                return prev;
              });
              if (accumulatedText.length > 0) {
                setSessionLoadingStates(prev => {
                  const next = { ...prev };
                  delete next[targetSessionId];
                  return next;
                });
              }
            },
            (error: Error) => {
              console.error('流式请求错误:', error);
              globalSendingLockRef.current = false;
              isSendingRef.current[targetSessionId] = false;
              isSendingRef.current['new'] = false;
              cancelStreamRef.current = null;
              setSessions(prev => {
                const session = prev.find(s => s.id === targetSessionId);
                if (session) {
                  const errorMsg = {
                    id: uuidv4(),
                    role: 'model' as const,
                    text: error.message || '抱歉,服务调用失败。',
                    timestamp: new Date(),
                  };
                  const newMsgs = [...session.messages, errorMsg];
                  return prev.map(s =>
                    s.id === targetSessionId ? { ...s, messages: newMsgs, lastModified: new Date() } : s
                  );
                }
                return prev;
              });
            },
            () => {
              console.log('✅ [流式] 完成,总长度=', accumulatedText.length);
              setSessionLoadingStates(prev => {
                const next = { ...prev };
                delete next[targetSessionId];
                return next;
              });
              globalSendingLockRef.current = false;
              isSendingRef.current[targetSessionId] = false;
              isSendingRef.current['new'] = false;
              cancelStreamRef.current = null;
            }
          );
          cancelStreamRef.current = cancelStream;
        } catch (e) {
          console.error(e);
          setSessionLoadingStates(prev => {
            const next = { ...prev };
            delete next[targetSessionId];
            return next;
          });
          globalSendingLockRef.current = false;
          isSendingRef.current[targetSessionId] = false;
          isSendingRef.current['new'] = false;
        }
      })();

      // === 用户画像分析 (并行) ===
      (async () => {
        try {
          const res = await chatApi.analyzePersona({
            user_id: user.id,
            text: textToSend,
            conversation_id: targetSessionId,
            model_configuration: {
              provider_id: selectedModel.providerId || '',
              model_id: selectedModel.realId || '',
              model_name: selectedModel.modelName || selectedModel.id,
              temperature: 0.1,
              top_p: 0.95,
              max_tokens: 512,
            },
          });
          if (res && res.success && res.data) {
            const newPersona = { ...persona, ...res.data };
            const changes: string[] = [];
            (Object.keys(newPersona) as Array<keyof UserPersona>).forEach(key => {
              if (newPersona[key] !== persona[key]) changes.push(key);
            });
            if (changes.length > 0) {
              updateSessionPersona(targetSessionId, newPersona);
              setChangedFields(changes);
              setTimeout(() => setChangedFields([]), 3000);
            }
          }
        } catch (e) {
          console.error('Persona analysis failed in parallel', e);
        }
      })();
    },
    [
      activeSessionId,
      sessions,
      selectedModel,
      providerConfigs,
      temperature,
      topP,
      maxTokens,
      enableThinking,
      defaultPersona,
      persona,
      user.id,
      currentSession?.messages,
      updateSessionMessages,
      updateSessionPersona,
    ]
  );

  // handleResumeMessage 单独定义,被 handleSendMessage 调用
  const handleResumeMessage = useCallback(
    async (textToSend: string, session: ChatSession) => {
      const targetSessionId = session.id;
      const userMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'user',
        text: textToSend,
        timestamp: new Date(),
      };
      setSessions(prev =>
        prev.map(x =>
          x.id === targetSessionId
            ? { ...x, messages: [...x.messages, userMsg], lastModified: new Date() }
            : x
        )
      );
      setSessionLoadingStates(prev => ({ ...prev, [targetSessionId]: true }));

      const modelMsgId = (Date.now() + 1).toString();
      const modelMsg: ChatMessage = {
        id: modelMsgId,
        role: 'model',
        text: '',
        timestamp: new Date(),
      };
      setSessions(prev =>
        prev.map(x =>
          x.id === targetSessionId
            ? { ...x, messages: [...x.messages, modelMsg], lastModified: new Date() }
            : x
        )
      );

      let accumulatedText = '';
      let currentSteps: string[] = [];

      try {
        const cancelStream = await chatApi.resumeStream(
          {
            conversation_id: targetSessionId,
            thread_id: session.threadId!,
            query: textToSend,
            model_configuration: {
              provider_id: selectedModel.providerId || '',
              model_id: selectedModel.realId || '',
              model_name: selectedModel.modelName || selectedModel.id,
              temperature,
              top_p: topP,
              max_tokens: maxTokens,
            },
          },
          (data: any) => {
            if (data.type === 'interrupt') {
              accumulatedText = data.question || '';
              setSessions(prev =>
                prev.map(s =>
                  s.id === targetSessionId
                    ? { ...s, threadId: data.thread_id || s.threadId, isInterrupted: true }
                    : s
                )
              );
              setSessions(prev => {
                const s = prev.find(s => s.id === targetSessionId);
                if (s) {
                  const updatedMessages = s.messages.map(m =>
                    m.id === modelMsgId
                      ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] }
                      : m
                  );
                  return prev.map(x =>
                    x.id === targetSessionId ? { ...x, messages: updatedMessages } : x
                  );
                }
                return prev;
              });
              globalSendingLockRef.current = false;
              isSendingRef.current[targetSessionId] = false;
              cancelStreamRef.current = null;
              setSessionLoadingStates(prev => {
                const next = { ...prev };
                delete next[targetSessionId];
                return next;
              });
              return;
            }
            if (data.type === 'done') {
              setSessions(prev =>
                prev.map(s =>
                  s.id === targetSessionId ? { ...s, isInterrupted: false } : s
                )
              );
              setSessions(prev => {
                const s = prev.find(s => s.id === targetSessionId);
                if (s) {
                  const updatedMessages = s.messages.map(m =>
                    m.id === modelMsgId ? { ...m, queryType: data.query_type } : m
                  );
                  return prev.map(x =>
                    x.id === targetSessionId ? { ...x, messages: updatedMessages } : x
                  );
                }
                return prev;
              });
              return;
            }
            if (data.type === 'content') {
              accumulatedText += data.content || '';
            } else if (data.type && data.type !== 'error') {
              currentSteps = [...currentSteps, data.type];
            }
            setSessions(prev => {
              const s = prev.find(s => s.id === targetSessionId);
              if (s) {
                const updatedMessages = s.messages.map(m =>
                  m.id === modelMsgId
                    ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] }
                    : m
                );
                return prev.map(x =>
                  x.id === targetSessionId ? { ...x, messages: updatedMessages } : x
                );
              }
              return prev;
            });
            if (accumulatedText.length > 0) {
              setSessionLoadingStates(prev => {
                const next = { ...prev };
                delete next[targetSessionId];
                return next;
              });
            }
          },
          (error: Error) => {
            console.error('Resume 流式请求错误:', error);
            globalSendingLockRef.current = false;
            isSendingRef.current[targetSessionId] = false;
            cancelStreamRef.current = null;
          },
          () => {
            setSessionLoadingStates(prev => {
              const next = { ...prev };
              delete next[targetSessionId];
              return next;
            });
            globalSendingLockRef.current = false;
            isSendingRef.current[targetSessionId] = false;
            cancelStreamRef.current = null;
          }
        );
        cancelStreamRef.current = cancelStream;
      } catch (e) {
        console.error('Resume error:', e);
        setSessionLoadingStates(prev => {
          const next = { ...prev };
          delete next[targetSessionId];
          return next;
        });
        globalSendingLockRef.current = false;
        isSendingRef.current[targetSessionId] = false;
      }
    },
    [selectedModel, temperature, topP, maxTokens]
  );

  // Save quick config (api key)
  const handleSaveQuickConfig = useCallback(
    async (apiKey: string) => {
      setProviderConfigs(prev => ({
        ...prev,
        [selectedProviderId]: {
          ...prev[selectedProviderId],
          apiKey: apiKey,
          enabled: true,
        },
      }));
      const current = JSON.parse(localStorage.getItem('user_provider_configs') || '{}');
      current[selectedProviderId] = {
        ...current[selectedProviderId],
        apiKey,
        enabled: true,
      };
      localStorage.setItem('user_provider_configs', JSON.stringify(current));

      const provider = apiData.find((p: any) => p.name === selectedProviderId);
      if (provider) {
        try {
          await providerApi.update({
            provider_id: provider.id,
            api_key: apiKey,
            is_enabled: true,
          });
        } catch (e) {
          console.error('Failed to sync API key to backend', e);
        }
      }
    },
    [selectedProviderId, apiData]
  );

  // ====== Render ======
  return (
    <div className="h-screen w-full flex bg-rice-paper overflow-hidden transition-colors duration-500">
      <LogoutConfirmModal
        isOpen={showLogoutModal}
        onConfirm={handleLogoutConfirm}
        onCancel={() => setShowLogoutModal(false)}
        variant="public"
      />

      <AlertModal
        isOpen={showAlertModal}
        onConfirm={() => setShowAlertModal(false)}
        title={alertConfig.title}
        description={alertConfig.description}
        variant="public"
        icon="AlertTriangle"
      />

      <AlertModal
        isOpen={confirmAction.isOpen}
        onConfirm={handleConfirmAction}
        onCancel={() => setConfirmAction(prev => ({ ...prev, isOpen: false }))}
        title={confirmAction.title}
        description={confirmAction.description}
        variant="public"
        icon="Trash2"
        confirmText="删除"
        cancelText="取消"
      />

      {/* 快速配置小窗 */}
      {isQuickConfigOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white dark:bg-tcm-charcoal w-full max-w-sm rounded-3xl shadow-2xl p-6 border border-tcm-gold/30 scale-in-center">
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="p-4 bg-tcm-gold/10 rounded-full text-tcm-gold">
                <Icons.Zap size={32} />
              </div>
              <h3 className="text-lg font-bold text-tcm-darkGreen dark:text-tcm-cream font-serif-sc">
                需配置服务商
              </h3>
              <p className="text-sm text-gray-500">
                您选中的模型属于{' '}
                <b>{currentProviders.find(p => p.id === selectedProviderId)?.name}</b>,需要配置
                API Key 才能使用。
              </p>

              <div className="w-full space-y-3 pt-2">
                <input
                  type="password"
                  placeholder="输入 Provider API Key"
                  autoFocus
                  className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-xl outline-none focus:ring-2 focus:ring-tcm-lightGreen transition-all text-sm"
                  onChange={e => handleSaveQuickConfig(e.target.value)}
                />
                <button
                  onClick={() => setIsQuickConfigOpen(false)}
                  className="w-full py-3 bg-tcm-darkGreen text-white rounded-xl font-bold hover:bg-tcm-lightGreen transition-all"
                >
                  保存并继续
                </button>
                <button
                  onClick={() => setIsQuickConfigOpen(false)}
                  className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                >
                  稍后再说
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <SessionSidebar
        user={user}
        sessions={sessions}
        activeSessionId={activeSessionId}
        showLeftSidebar={showLeftSidebar}
        isDarkMode={isDarkMode}
        onCloseSidebar={handleCloseLeftSidebar}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onToggleTheme={handleToggleTheme}
        onEditProfile={handleOpenEditProfile}
        onOpenModelManagement={handleOpenModelManagement}
        onLogout={handleLogoutClick}
      />

      <main className="flex-1 flex flex-col relative z-10 transition-colors duration-500 min-w-0 min-h-0 bg-transparent">
        <header className="h-16 flex items-center justify-end px-6 z-10 transition-colors bg-white/80 dark:bg-[#131314]/80 backdrop-blur-sm border-b border-tcm-lightGreen/5">
          <button
            onClick={handleOpenHistory}
            title="健康档案"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
              rightPanelMode === 'history' && showRightSidebar
                ? 'bg-tcm-lightGreen/10 text-tcm-lightGreen'
                : 'text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-white/5'
            }`}
          >
            <Icons.Stethoscope size={16} />
            <span>档案</span>
          </button>
          <ModelSelector
            selectedProviderId={selectedProviderId}
            selectedModel={selectedModel}
            providers={currentProviders}
            filteredModels={filteredModels}
            rightPanelMode={rightPanelMode}
            showRightSidebar={showRightSidebar}
            onProviderChange={handleProviderChange}
            onModelSelect={handleModelSelect}
            onOpenModelSettings={handleOpenModelSettings}
            onToggleRightSidebar={handleToggleRightSidebar}
          />
        </header>

        <ChatPanel
          user={user}
          currentSession={currentSession}
          selectedModel={selectedModel}
          enableThinking={enableThinking}
          isLoading={isCurrentSessionLoading}
          showLeftSidebar={showLeftSidebar}
          globalSendingLockRef={globalSendingLockRef}
          onOpenLeftSidebar={handleOpenLeftSidebar}
          onSendMessage={handleSendMessage}
          onCancelGeneration={handleCancelGeneration}
          onDeleteMessage={handleDeleteMessage}
          onToggleThinking={handleToggleThinking}
        />
      </main>

      <SettingsPanel
        showRightSidebar={showRightSidebar}
        rightPanelMode={rightPanelMode}
        isDarkMode={isDarkMode}
        persona={persona}
        healthScore={healthScore}
        changedFields={changedFields}
        temperature={temperature}
        topP={topP}
        maxTokens={maxTokens}
        selectedModel={selectedModel}
        onChangeMode={handleSetRightPanelMode}
        onCloseSidebar={handleCloseRightSidebar}
        onTemperatureChange={handleTemperatureChange}
        onTopPChange={handleTopPChange}
        onMaxTokensChange={handleMaxTokensChange}
        onResetDefaults={handleResetDefaults}
      />

      {rightPanelMode === 'history' && showRightSidebar && (
        <HealthProfile onBack={handleCloseRightSidebar} />
      )}

      {/* 编辑个人资料 Modal */}
      {isEditingProfile && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-tcm-charcoal rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden animate-fade-in-up border border-tcm-lightGreen/20">
            <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
              <h2 className="text-xl font-bold text-tcm-darkGreen dark:text-tcm-cream font-serif-sc">
                修改健康档案
              </h2>
              <button
                onClick={handleCloseEditProfile}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
              >
                <Icons.X />
              </button>
            </div>
            <div className="p-6 space-y-4 max-h-[60vh] overflow-y-auto">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase mb-1">
                    年龄
                  </label>
                  <input
                    type="text"
                    value={editPersonaForm.age}
                    onChange={e => setEditPersonaForm({ ...editPersonaForm, age: e.target.value })}
                    className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg dark:text-white focus:ring-1 focus:ring-tcm-lightGreen outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase mb-1">
                    性别
                  </label>
                  <input
                    type="text"
                    value={editPersonaForm.gender}
                    onChange={e => setEditPersonaForm({ ...editPersonaForm, gender: e.target.value })}
                    className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg dark:text-white focus:ring-1 focus:ring-tcm-lightGreen outline-none"
                  />
                </div>
              </div>
              <div>
                <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase mb-1">
                  既往病史
                </label>
                <textarea
                  value={editPersonaForm.base_profile?.medical_history || ''}
                  onChange={e =>
                    setEditPersonaForm({
                      ...editPersonaForm,
                      base_profile: {
                        ...(editPersonaForm.base_profile || {}),
                        medical_history: e.target.value,
                      },
                    })
                  }
                  className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg h-24 dark:text-white focus:ring-1 focus:ring-tcm-lightGreen outline-none"
                />
              </div>
            </div>
            <div className="p-6 bg-gray-50 dark:bg-black/10 flex justify-end gap-3">
              <button
                onClick={handleCloseEditProfile}
                className="px-6 py-2 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-white/5 rounded-lg"
              >
                取消
              </button>
              <button
                onClick={handleSaveProfile}
                className="px-6 py-2 bg-tcm-darkGreen text-white rounded-lg hover:bg-tcm-lightGreen shadow-lg"
              >
                保存更改
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PublicPortal;
