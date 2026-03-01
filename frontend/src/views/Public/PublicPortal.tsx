import React, { useState, useEffect, useLayoutEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, ChatMessage, UserPersona, AIModelConfig, ProviderConfig, CustomModel } from '../../types';
import { AVAILABLE_MODELS, PROVIDERS } from '../../constants';
import { Icons } from '../../components/common/Icons';
import { BrandLogo } from '../../components/common/BrandLogo';
import { LogoutConfirmModal } from '../../components/common/LogoutConfirmModal';
import { AlertModal } from '../../components/common/AlertModal';
import { v4 as uuidv4 } from 'uuid';
import { providerApi } from '../../api/modules/model';
import { chatApi } from '../../api/modules/chat';
import { conversationApi } from '../../api/modules/conversation';
import { getProviderIconPath, getModelIconPath, isDarkInvert } from '../../utils/iconMap';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface PublicPortalProps {
  user: User;
  onLogout: () => void;
}

interface Attachment {
  file: File;
  previewUrl: string;
  base64: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  lastModified: Date;
  persona?: UserPersona; // Add persona to session
  healthScore?: number;  // Add health score to session
  threadId?: string;       // LangGraph thread_id for resume
  isInterrupted?: boolean; // Whether waiting for user follow-up answer
}

// Helper to group sessions by date
const groupSessionsByDate = (sessions: ChatSession[]) => {
  const groups: { [key: string]: ChatSession[] } = {
    '今天': [],
    '昨天': [],
    '最近7天': [],
    '更早': []
  };

  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  sessions.forEach(session => {
    const date = new Date(session.lastModified);
    const sessionDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());

    if (sessionDate.getTime() === today.getTime()) {
      groups['今天'].push(session);
    } else if (sessionDate.getTime() === yesterday.getTime()) {
      groups['昨天'].push(session);
    } else if (sessionDate > lastWeek) {
      groups['最近7天'].push(session);
    } else {
      groups['更早'].push(session);
    }
  });

  return groups;
};

const PublicPortal: React.FC<PublicPortalProps> = ({ user, onLogout }) => {
  const navigate = useNavigate();
  
  // --- State ---

  // Sessions Management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  
  const [activeSessionId, setActiveSessionId] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchActive, setIsSearchActive] = useState(false);
  
  // UI Toggles
  const [showLeftSidebar, setShowLeftSidebar] = useState(true);
  const [showRightSidebar, setShowRightSidebar] = useState(true);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
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
    description: ''
  });
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [showProviderSelector, setShowProviderSelector] = useState(false);
  const [isQuickConfigOpen, setIsQuickConfigOpen] = useState(false);

  // Persistence: Provider Configs & Custom Models
  const [providerConfigs, setProviderConfigs] = useState<Record<string, ProviderConfig>>(() => {
    const saved = localStorage.getItem('user_provider_configs');
    return saved ? JSON.parse(saved) : {};
  });

  const [customModels, setCustomModels] = useState<CustomModel[]>(() => {
    const saved = localStorage.getItem('user_custom_models');
    return saved ? JSON.parse(saved) : [];
  });



  // Model & Provider State
  const [selectedProviderId, setSelectedProviderId] = useState<string>(() => {
      return localStorage.getItem('last_selected_provider') || 'google';
  });
  
  // Persist provider selection
  useEffect(() => {
      if (selectedProviderId) {
          localStorage.setItem('last_selected_provider', selectedProviderId);
      }
  }, [selectedProviderId]);

  // Removed providerType state as we now show all in one list
  const [apiData, setApiData] = useState<any[]>([]);

  // 🛡️ 防止 StrictMode 或页面切换导致的重复初始化（放在所有 useEffect 之前）
  const isInitializedRef = useRef(false);
  const isProviderFetchedRef = useRef(false);

  // Fetch Providers and Models from API
  useEffect(() => {
    // 防止 StrictMode 双重挂载导致的重复请求
    if (isProviderFetchedRef.current) {
        console.log('⚠️ 跳过重复的 Provider 初始化请求');
        return;
    }
    isProviderFetchedRef.current = true;

    const fetchData = async () => {
      try {
        // Use filtered API with 'all' to get everything
        const response = await providerApi.get_providers_filtered('all');

        if (response.success && response.data) {
          setApiData(response.data);

          // Sync backend config existence to local state
          setProviderConfigs(prev => {
              const next = { ...prev };
              let changed = false;
              response.data.forEach((p: any) => {
                  // If backend has key (encrypted string) and local doesn't, mark as configured
                  if (p.api_key && (!next[p.name]?.apiKey)) {
                      next[p.name] = {
                          ...next[p.name],
                          apiKey: 'CONFIGURED_IN_BACKEND',
                          enabled: true
                      };
                      changed = true;
                  }
              });
              return changed ? next : prev;
          });

          // If current selected provider is not in the list (and list is not empty), select the first one
          if (response.data.length > 0) {
             const currentExists = response.data.some((p: any) => p.name === selectedProviderId);
             if (!currentExists) {
                // Try to find google or just take the first one
                const defaultProvider = response.data.find((p: any) => p.name === 'google') || response.data[0];
                setSelectedProviderId(defaultProvider.name);
             }
          } else {
             // No providers found
             // setSelectedProviderId(''); // Optional: clear selection
          }
        }
      } catch (error) {
        console.error("Failed to fetch models config:", error);
      }
    };
    fetchData();
  }, []); // Fetch once on mount

  // Fetch Conversations on mount
  useEffect(() => {
    // 防止 StrictMode 双重挂载导致的重复请求
    if (isInitializedRef.current) {
        console.log('⚠️ 跳过重复的初始化请求');
        return;
    }
    isInitializedRef.current = true;

    const fetchConversations = async () => {
        try {
            const res = await conversationApi.getConversations();
            console.log("Fetched conversations:", res);
            if (res.success && res.data && res.data.length > 0) {
                const backendSessions: ChatSession[] = res.data.map((c: any) => ({
                    id: c.id,
                    title: c.title || '无标题对话',
                    messages: [], // Load on demand
                    persona: c.session_metadata, // Map backend session_metadata to frontend persona
                    lastModified: (() => {
                        const date = c.updated_at ? new Date(c.updated_at) : new Date();
                        return isNaN(date.getTime()) ? new Date() : date;
                    })()
                }));

                setSessions(backendSessions);
                // 🛡️ 只在没有 activeSessionId 时才设置为空
                // 避免覆盖正在进行的会话
                setActiveSessionId(prev => prev || '');
            } else {
                setSessions([]);
                setActiveSessionId(prev => prev || '');
            }
        } catch (e) {
            console.error("Failed to fetch conversations", e);
            setSessions([]);
            setActiveSessionId(prev => prev || '');
        }
    };
    fetchConversations();
  }, []);

  // Ref to track sessions without triggering effect dependencies
  const sessionsRef = useRef(sessions);
  useEffect(() => { sessionsRef.current = sessions; }, [sessions]);

  // Ref to track which sessions have been fetched to prevent infinite loops
  const fetchedSessionIds = useRef<Set<string>>(new Set());

  // Fetch Messages when active session changes
  useEffect(() => {
    const fetchMessages = async () => {
        // Find current session in ref
        const session = sessionsRef.current.find(s => s.id === activeSessionId);
        
        // Skip if:
        // 1. Session not found
        // 2. Session already has messages (e.g. local new chat with welcome msg)
        // 3. Already fetched this session (prevents loop if backend returns empty)
        if (!session || session.messages.length > 0 || fetchedSessionIds.current.has(activeSessionId)) {
            return;
        }

        // Mark as fetched immediately to prevent race conditions or loops
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
                            const meta = typeof m.message_metadata === 'string' ? JSON.parse(m.message_metadata) : m.message_metadata;
                            return meta.attachments || [];
                        } catch (e) {
                            return [];
                        }
                    })()
                }));
                
                // Update the session with fetched messages
                setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, messages } : s));
            }
        } catch (e) {
            console.error("Failed to fetch messages for session", activeSessionId, e);
        }
    };

    if (activeSessionId) {
        fetchMessages();
    }
  }, [activeSessionId]); // Only depend on activeSessionId change

  // Computed Available Models (Filtered by Provider AND Enabled status)
  const allModels = useMemo(() => {
    if (apiData.length > 0) {
        // Map API data to AIModelConfig structure
        const models: AIModelConfig[] = [];
        apiData.forEach((p: any) => {
            if (p.models) {
                p.models.forEach((m: any) => {
                    models.push({
                        id: m.model_name, // Use model_name as ID for UI logic (e.g. icons)
                        realId: m.id,     // The UUID for backend API calls
                        modelName: m.model_name,
                        providerId: p.id, // The UUID of the provider
                        name: m.label || m.model_name,
                        description: m.description || '',
                        supportsThinking: m.features?.includes('thinking') || false,
                        supportsVision: m.features?.includes('vision') || false,
                        supportsToolCall: m.features?.includes('tool_call') || false,
                        provider: p.name as any,
                        contextWindow: m.context_window ? `${Math.round(m.context_window/1000)}K` : undefined,
                        defaultTemperature: m.default_temperature,
                        defaultTopP: m.default_top_p,
                        defaultMaxTokens: m.default_max_tokens,
                        isEnabled: m.is_enabled,
                        isBuiltin: m.is_builtin // This flag comes from backend now
                    });
                });
            }
        });
        // Merge with custom models if any (local storage ones)
        // Note: Ideally backend handles everything now, but keeping for compatibility if needed
        return models;
    }
    // Fallback to constants if API fails or empty
    return [...AVAILABLE_MODELS, ...customModels];
  }, [apiData, customModels]);

  const currentProviders = useMemo(() => {
      if (apiData.length > 0) {
          return apiData
            .filter((p: any) => p.is_enabled !== false)
            .map((p: any) => ({
              id: p.name, // Use name as ID to match model provider field
              name: p.label || p.name,
              icon: p.icon || '🤖',
              isBuiltin: p.is_builtin
          }));
      }
      return PROVIDERS.map(p => ({...p, isBuiltin: true}));
  }, [apiData]);

  const filteredModels = useMemo(() => 
    allModels.filter(m => m.provider === selectedProviderId && m.isEnabled !== false), 
  [selectedProviderId, allModels]);
  
  const [selectedModel, setSelectedModel] = useState<AIModelConfig>(filteredModels[0] || AVAILABLE_MODELS[0]);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(4096);  // 从 2000 提升到 4096，避免长文本截断
  const [enableThinking, setEnableThinking] = useState(false);  // 是否启用 thinking 模式

  const [rightPanelMode, setRightPanelMode] = useState<'health' | 'settings'>('health');

  // Sync when data changes externally (e.g. from management page)
  useEffect(() => {
    const handleStorageChange = () => {
        const savedProviderConfigs = localStorage.getItem('user_provider_configs');
        if (savedProviderConfigs) setProviderConfigs(JSON.parse(savedProviderConfigs));
        
        const savedCustomModels = localStorage.getItem('user_custom_models');
        if (savedCustomModels) setCustomModels(JSON.parse(savedCustomModels));


    };
    
    window.addEventListener('storage', handleStorageChange);
    handleStorageChange(); // Init
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  useEffect(() => {
    // Check if the current selected model is valid for the current filtered list
    const isSelectedValid = filteredModels.some(m => m.id === selectedModel.id);
    
    if (isSelectedValid) {
        // Update the object reference to ensure we have the latest fields (like realId from API)
        const currentInFiltered = filteredModels.find(m => m.id === selectedModel.id);
        if (currentInFiltered && currentInFiltered !== selectedModel) {
            // Check if they are actually different in content that matters (e.g. realId)
            // Or just always update to keep fresh. Since filteredModels is memoized, 
            // the object ref only changes when dependencies change.
            if (currentInFiltered.realId !== selectedModel.realId || currentInFiltered.providerId !== selectedModel.providerId) {
                setSelectedModel(currentInFiltered);
            }
        }
    } else {
        // Try to restore from localStorage first
        const lastModelId = localStorage.getItem('last_selected_model_id');
        const lastModel = lastModelId ? filteredModels.find(m => m.id === lastModelId) : undefined;

        if (lastModel) {
            setSelectedModel(lastModel);
        } else if (filteredModels.length > 0) {
            setSelectedModel(filteredModels[0]);
        } else if (selectedModel.id !== '') {
            // No models available for this provider (empty provider)
            // Set a dummy model to reset the UI state, but only if not already reset
            setSelectedModel({
                id: '', 
                name: '无可用模型',
                description: '该提供商下暂无可用模型',
                provider: selectedProviderId,
                supportsThinking: false,
                supportsVision: false,
                supportsToolCall: false,
                isEnabled: false,
                isBuiltin: false
            });
        }
    }
  }, [filteredModels, selectedModel, selectedProviderId]);

  // Persist model selection
  useEffect(() => {
      if (selectedModel && selectedModel.id) {
          localStorage.setItem('last_selected_model_id', selectedModel.id);
      }
  }, [selectedModel]);

  // Chat State
  const currentSession = sessions.find(s => s.id === activeSessionId) || null;
  const [inputValue, setInputValue] = useState('');
  
  // Use a map to track loading state for EACH session independently
  // Key: sessionId, Value: boolean
  const [sessionLoadingStates, setSessionLoadingStates] = useState<Record<string, boolean>>({});

  // Helper to check if current session is loading
  const isCurrentSessionLoading = useMemo(() => {
      return activeSessionId ? !!sessionLoadingStates[activeSessionId] : false;
  }, [activeSessionId, sessionLoadingStates]);

  // Settings
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  // Theme & Persona
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme_public');
    return saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });
  
  // Field label mapping
  const PERSONA_FIELD_LABELS: Record<string, string> = {
    age: '年龄',
    gender: '性别',
    chief_complaint: '主诉症状',
    suspected_diagnosis: '疑似诊断',
    recommended_treatment: '调理建议'
  };

  // Default Persona Template
  const defaultPersona: UserPersona = {
    age: '',
    gender: '',
    chief_complaint: '待分析...',
    suspected_diagnosis: '分析待定',
    recommended_treatment: ' wellness 建议',
    health_score: 100,
    base_profile: user.base_profile
  };

  // Current Persona is derived from the active session
  // If active session has persona, use it.
  // Else (e.g. landing page), use Base Profile + defaults
  const persona = useMemo(() => {
      if (currentSession?.persona) {
          return currentSession.persona;
      }
      
      // Merge base profile with defaults for landing page
      // Use defaultPersona which already includes baseProfile data
      return defaultPersona;
  }, [currentSession, user.base_profile]);

  const healthScore = useMemo(() => {
      return persona?.health_score  || 85;
  }, [persona, currentSession]);

  const [editPersonaForm, setEditPersonaForm] = useState<UserPersona>(defaultPersona);
  const [changedFields, setChangedFields] = useState<string[]>([]);


  // Sync edit form when persona changes (e.g. switching sessions)
  useEffect(() => {
      setEditPersonaForm(persona);
  }, [persona]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  
  // 防重复提交：跟踪每个会话的发送状态 + 时间戳防护
  const isSendingRef = useRef<Record<string, boolean>>({});
  const lastSendTimeRef = useRef<number>(0);

  // 🛑 取消流式请求的函数引用
  const cancelStreamRef = useRef<(() => void) | null>(null);

  // 取消当前对话的处理函数
  const handleCancelGeneration = () => {
    if (cancelStreamRef.current) {
      console.log('🛑 用户取消了对话生成');
      cancelStreamRef.current();
      cancelStreamRef.current = null;

      // 解除所有锁定
      globalSendingLockRef.current = false;
      Object.keys(isSendingRef.current).forEach(key => {
        isSendingRef.current[key] = false;
      });

      // 清除 loading 状态
      setSessionLoadingStates({});

      // 更新当前消息，标记为已取消
      if (activeSessionId) {
        setSessions(prev => prev.map(s => {
          if (s.id === activeSessionId) {
            const lastMsg = s.messages[s.messages.length - 1];
            if (lastMsg && lastMsg.role === 'model' && !lastMsg.text) {
              // 如果最后一条消息是空的 AI 消息，添加取消提示
              const updatedMessages = s.messages.map((m, idx) =>
                idx === s.messages.length - 1
                  ? { ...m, text: '*[对话已取消]*' }
                  : m
              );
              return { ...s, messages: updatedMessages };
            }
          }
          return s;
        }));
      }
    }
  };

  // 🛡️ 页面可见性变化保护：当用户切换回页面时，记录日志并防止意外触发
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        console.log('📱 [页面状态] 页面不可见，用户切换到其他标签/应用');
      } else {
        console.log('📱 [页面状态] 页面重新可见，用户切换回来');
        // 重置时间戳防护，防止可能的意外触发
        // 但保持发送锁状态，防止正在进行的请求被干扰
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  // --- Effects ---

  useLayoutEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme_public', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme_public', 'light');
    }
  }, [isDarkMode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages]);

  // --- Handlers ---
  


  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files: File[] = Array.from(e.target.files);
      const newAttachments: Attachment[] = await Promise.all(
        files.map(async (file: File) => {
          const base64 = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onload = (ev) => resolve(ev.target?.result as string);
            reader.readAsDataURL(file);
          });
          return { file, previewUrl: URL.createObjectURL(file), base64 };
        })
      );
      setAttachments(prev => [...prev, ...newAttachments]);
    }
  };

  // 语音识别逻辑 (来自 a.ts)
  const handleVoiceInput = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'zh-CN';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onstart = () => setInputValue(prev => prev + " (正在倾听...)");
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(prev => prev.replace(" (正在倾听...)", "") + transcript);
      };
      recognition.onerror = () => setInputValue(prev => prev.replace(" (正在倾听...)", ""));
      recognition.start();
    } else {
      alert("您的浏览器不支持语音识别功能。");
    }
  };

  const handleNewChat = () => {
    setActiveSessionId('');
    setInputValue('');
    if (window.innerWidth < 768) setShowLeftSidebar(false);
  };

  const handleConfirmAction = async () => {
    const { type, id } = confirmAction;
    setConfirmAction(prev => ({ ...prev, isOpen: false })); // Close modal immediately

    if (type === 'deleteSession' && id) {
        try {
            const res = await conversationApi.deleteConversation(id);
            if (res.success) {
                setSessions(prev => {
                    const next = prev.filter(s => s.id !== id);
                    if (next.length === 0) {
                        // All sessions deleted, create a new one via API
                        // We use a self-executing async function here or handle it in effect
                        // But since state update is sync, we can't await here directly in filter
                        // Better to handle the creation outside
                        return next; 
                    }
                    return next;
                });
                
                // Determine next session ID
                const nextSessions = sessions.filter(s => s.id !== id);
                if (nextSessions.length > 0) {
                    if (activeSessionId === id) {
                        setActiveSessionId(nextSessions[0].id);
                    }
                } else {
                    // No sessions left, go to Landing Page
                    setActiveSessionId('');
                }
            }
        } catch (error) {
            console.error("Failed to delete session", error);
            setAlertConfig({ title: '删除失败', description: '删除会话时发生错误，请稍后重试。' });
            setShowAlertModal(true);
        }
    } else if (type === 'deleteMessage' && id) {
        try {
            const res = await conversationApi.deleteMessage(id);
            if (res.success) {
                // Fetch fresh messages to ensure we get the correct state after paired deletion
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
                                const meta = typeof m.message_metadata === 'string' ? JSON.parse(m.message_metadata) : m.message_metadata;
                                return meta.attachments || [];
                            } catch (e) {
                                return [];
                            }
                        })()
                    }));
                    setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, messages } : s));
                }
            }
        } catch (error) {
            console.error("Failed to delete message", error);
            setAlertConfig({ title: '删除失败', description: '删除消息时发生错误，请稍后重试。' });
            setShowAlertModal(true);
        }
    }
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setConfirmAction({
          isOpen: true,
          type: 'deleteSession',
          id: sessionId,
          title: '确认删除会话',
          description: '确定要删除这个会话吗？此操作不可恢复。'
      });
  };

  const handleDeleteMessage = async (messageId: string) => {
      setConfirmAction({
          isOpen: true,
          type: 'deleteMessage',
          id: messageId,
          title: '确认删除消息',
          description: '确定要删除这条消息吗？'
      });
  };

  // 🛡️ 全局发送锁 - 防止任何情况下的重复发送
  const globalSendingLockRef = useRef(false);

  const handleResumeMessage = async (textToSend: string, session: ChatSession) => {
    const targetSessionId = session.id;

    // 添加用户消息到 UI
    const userMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'user',
        text: textToSend,
        timestamp: new Date()
    };
    setSessions(prev => {
        const s = prev.find(s => s.id === targetSessionId);
        if (s) {
            return prev.map(x => x.id === targetSessionId
                ? { ...x, messages: [...x.messages, userMsg], lastModified: new Date() }
                : x
            );
        }
        return prev;
    });

    setInputValue('');
    setSessionLoadingStates(prev => ({ ...prev, [targetSessionId]: true }));

    // 创建空 AI 消息
    const modelMsgId = (Date.now() + 1).toString();
    const modelMsg: ChatMessage = {
        id: modelMsgId,
        role: 'model',
        text: '',
        timestamp: new Date()
    };
    setSessions(prev => {
        const s = prev.find(s => s.id === targetSessionId);
        if (s) {
            return prev.map(x => x.id === targetSessionId
                ? { ...x, messages: [...x.messages, modelMsg], lastModified: new Date() }
                : x
            );
        }
        return prev;
    });

    let accumulatedText = '';
    let currentSteps: string[] = [];

    try {
        const cancelStream = await chatApi.resumeStream(
            {
                conversation_id: targetSessionId,
                thread_id: session.threadId!,
                query: textToSend,
                model_configuration: {
                    provider_id: selectedModel.providerId || "",
                    model_id: selectedModel.realId || "",
                    model_name: selectedModel.modelName || selectedModel.id,
                    temperature: temperature,
                    top_p: topP,
                    max_tokens: maxTokens
                }
            },
            (data) => {
                // interrupt：再次追问
                if (data.type === 'interrupt') {
                    accumulatedText = data.question || '';
                    setSessions(prev => prev.map(s =>
                        s.id === targetSessionId
                            ? { ...s, threadId: data.thread_id || s.threadId, isInterrupted: true }
                            : s
                    ));
                    setSessions(prev => {
                        const s = prev.find(s => s.id === targetSessionId);
                        if (s) {
                            const updatedMessages = s.messages.map(m =>
                                m.id === modelMsgId ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] } : m
                            );
                            return prev.map(x => x.id === targetSessionId ? { ...x, messages: updatedMessages } : x);
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
                    // 完成：清除 interrupted 状态
                    setSessions(prev => prev.map(s =>
                        s.id === targetSessionId
                            ? { ...s, isInterrupted: false }
                            : s
                    ));
                    setSessions(prev => {
                        const s = prev.find(s => s.id === targetSessionId);
                        if (s) {
                            const updatedMessages = s.messages.map(m =>
                                m.id === modelMsgId ? { ...m, queryType: data.query_type } : m
                            );
                            return prev.map(x => x.id === targetSessionId ? { ...x, messages: updatedMessages } : x);
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
                            m.id === modelMsgId ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] } : m
                        );
                        return prev.map(x => x.id === targetSessionId ? { ...x, messages: updatedMessages } : x);
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
            (error) => {
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
  };

  const handleSendMessage = async (overrideText?: string) => {
    // 🔍 详细日志：记录每次调用的来源和状态
    console.log('🚀 [handleSendMessage] 被调用', {
      overrideText: overrideText?.slice(0, 30),
      inputValue: inputValue?.slice(0, 30),
      activeSessionId,
      globalLock: globalSendingLockRef.current,
      timestamp: new Date().toISOString(),
      callStack: new Error().stack?.split('\n').slice(1, 4).join(' <- ')
    });

    // 🛡️ 防重复提交：第一道防线 - 全局锁检查
    if (globalSendingLockRef.current) {
      console.log('⚠️ [全局锁] 有请求正在进行中，忽略重复提交');
      return;
    }

    // 🛡️ 防重复提交：第二道防线 - 时间戳检查（防止2秒内重复调用）
    const now = Date.now();
    if (now - lastSendTimeRef.current < 2000) {
      console.log('⚠️ [时间戳] 请求间隔过短，忽略重复提交');
      return;
    }

    // 🛡️ 防重复提交：第三道防线 - 会话锁
    const currentSessionKey = activeSessionId || 'new';
    if (isSendingRef.current[currentSessionKey]) {
      console.log('⚠️ [会话锁] 该会话请求已在进行中，忽略重复提交');
      return;
    }

    // 立即设置所有锁
    globalSendingLockRef.current = true;
    lastSendTimeRef.current = now;
    isSendingRef.current[currentSessionKey] = true;

    // 检查是否处于 interrupted 状态（追问等待用户回答）
    const currentSess = sessions.find(s => s.id === activeSessionId);
    if (currentSess?.isInterrupted && currentSess?.threadId) {
        const textForResume = overrideText || inputValue;
        if (!textForResume.trim()) {
            globalSendingLockRef.current = false;
            isSendingRef.current[currentSessionKey] = false;
            return;
        }
        // 走 resume 流程
        await handleResumeMessage(textForResume, currentSess);
        return;
    }

    if (!selectedModel.id) {
      globalSendingLockRef.current = false;
      isSendingRef.current[currentSessionKey] = false;
      setAlertConfig({
        title: '未选择有效模型',
        description: '当前未选择有效的模型配置。请先切换到有可用模型的提供商，或在模型管理中启用模型。'
      });
      setShowAlertModal(true);
      return;
    }

    const textToSend = overrideText || inputValue;
    if ((!textToSend.trim() && attachments.length === 0)) {
      globalSendingLockRef.current = false;
      isSendingRef.current[currentSessionKey] = false;
      return;
    }

    // Check configuration
    // Look at Provider Level Config first
    const providerConfig = providerConfigs[selectedModel.provider];
    const isGoogle = selectedModel.provider === 'google';

    // Logic: If not google (built-in assumed), check if provider API key exists
    if (!isGoogle && !providerConfig?.apiKey) {
      globalSendingLockRef.current = false;
      isSendingRef.current[currentSessionKey] = false;
      setIsQuickConfigOpen(true);
      return;
    }

    // Determine target Session ID
    let targetSessionId = activeSessionId;
    let isNewSession = false;

    // If no active session (Landing Page mode), generate ID locally
    if (!targetSessionId) {
        targetSessionId = uuidv4();
        isNewSession = true;
        
        // Optimistically create session in UI state
        // The backend will create the record automatically when receiving the first message
        const newSession: ChatSession = {
            id: targetSessionId,
            title: textToSend.slice(0, 20) || '新的对话',
            messages: [],
            lastModified: new Date(),
            // Initialize with Base Profile + Defaults
            // Use defaultPersona which already includes baseProfile data
            persona: defaultPersona
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
    }

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: textToSend,
      timestamp: new Date(),
      attachments: attachments.map(a => ({
        type: a.file.type.startsWith('image') ? 'image' : 'file',
        url: a.previewUrl,
        name: a.file.name
      }))
    };

    // Optimistically update UI
    // If it was a new session, we already added it to state above.
    // We need to add the message to the session.
    updateSessionMessages(targetSessionId, isNewSession ? [userMsg] : [...(sessions.find(s => s.id === targetSessionId)?.messages || []), userMsg]);

    // Update title if it's the first message
    if (isNewSession || (currentSession?.messages.length === 0)) {
       const newTitle = textToSend.slice(0, 20) + (textToSend.length > 20 ? '...' : '');
       setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, title: newTitle } : s));
    }

    setInputValue('');
    const currentAttachments = [...attachments];
    setAttachments([]);
    
    // Set loading for this specific session
    setSessionLoadingStates(prev => ({ ...prev, [targetSessionId]: true }));

    // ... (rest of the logic uses targetSessionId)

    // 构造历史消息 (Need to fetch latest state or use what we have)
    // Since we just updated state, we might not have it in 'currentSession' var yet if it's stale closure
    // But we can construct history from what we know.
    const history = isNewSession ? [] : (currentSession?.messages || []).map(m => ({
        role: m.role === 'model' ? 'assistant' : 'user',
        content: m.text
    }));
    
    // 添加当前消息
    const currentContent = userMsg.text || (currentAttachments.length > 0 ? "Analyzed attachment" : "");
    history.push({
        role: 'user',
        content: currentContent
    });

    // 并行调用：1. 聊天生成（流式） 2. 用户画像分析
    await (async () => {
        // 创建一个空的 AI 消息，逐渐填充内容
        const modelMsgId = (Date.now() + 1).toString();
        const modelMsg: ChatMessage = {
            id: modelMsgId,
            role: 'model',
            text: '',
            timestamp: new Date()
        };
        
        // 添加到 UI
        setSessions(prev => {
            const session = prev.find(s => s.id === targetSessionId);
            if (session) {
                const newMsgs = [...session.messages, modelMsg];
                return prev.map(s => s.id === targetSessionId ? { ...s, messages: newMsgs, lastModified: new Date() } : s);
            }
            return prev;
        });

        let accumulatedText = '';
        let currentSteps: string[] = [];

        try {
            // 🚀 使用企业级 SSE 封装
            const cancelStream = await chatApi.generateStream(
                {
                    user_id: user.id,
                    conversation_id: targetSessionId,
                    query: userMsg.text || (currentAttachments.length > 0 ? "Analyzed attachment" : ""),
                    model_configuration: {
                        provider_id: selectedModel.providerId || "",
                        model_id: selectedModel.realId || "",
                        model_name: selectedModel.modelName || selectedModel.id,
                        temperature: temperature,
                        top_p: topP,
                        max_tokens: maxTokens
                    },
                    stream: true,
                    enable_thinking: enableThinking  // 传递 thinking 参数
                },
                // onMessage: 每次接收到数据时调用（完整解析对象）
                (data: { type: string; content?: string; query_type?: string; steps?: string[]; thread_id?: string; question?: string; action?: string }) => {
                    // 保存 thread_id（首条消息）
                    if (data.type === 'thread_init' && data.thread_id) {
                        setSessions(prev => prev.map(s =>
                            s.id === targetSessionId ? { ...s, threadId: data.thread_id } : s
                        ));
                        return;
                    }

                    // interrupt：追问暂停，显示问题并解锁输入
                    if (data.type === 'interrupt') {
                        accumulatedText = data.question || '';
                        // 记录 thread_id 和 interrupted 状态
                        setSessions(prev => prev.map(s =>
                            s.id === targetSessionId
                                ? { ...s, threadId: data.thread_id || s.threadId, isInterrupted: true }
                                : s
                        ));
                        // 更新 AI 消息为追问问题
                        setSessions(prev => {
                            const session = prev.find(s => s.id === targetSessionId);
                            if (session) {
                                const updatedMessages = session.messages.map(m =>
                                    m.id === modelMsgId ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] } : m
                                );
                                return prev.map(s => s.id === targetSessionId ? { ...s, messages: updatedMessages } : s);
                            }
                            return prev;
                        });
                        // 解锁发送，允许用户继续输入
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
                        // 完成消息：存储 queryType
                        setSessions(prev => {
                            const session = prev.find(s => s.id === targetSessionId);
                            if (session) {
                                const updatedMessages = session.messages.map(m =>
                                    m.id === modelMsgId ? { ...m, queryType: data.query_type } : m
                                );
                                return prev.map(s => s.id === targetSessionId ? { ...s, messages: updatedMessages } : s);
                            }
                            return prev;
                        });
                        return;
                    }

                    if (data.type === 'content') {
                        // LLM token：累积文本
                        accumulatedText += data.content || '';
                    } else if (data.type && data.type !== 'error') {
                        // 状态消息（中文步骤名）：追加到步骤列表
                        currentSteps = [...currentSteps, data.type];
                    }

                    // 更新 UI：同时更新 text 和 agentSteps
                    setSessions(prev => {
                        const session = prev.find(s => s.id === targetSessionId);
                        if (session) {
                            const updatedMessages = session.messages.map(m =>
                                m.id === modelMsgId ? { ...m, text: accumulatedText, agentSteps: [...currentSteps] } : m
                            );
                            return prev.map(s => s.id === targetSessionId ? { ...s, messages: updatedMessages } : s);
                        }
                        return prev;
                    });

                    // 收到第一个 content 时清除 loading
                    if (accumulatedText.length > 0) {
                        setSessionLoadingStates(prev => {
                            const next = { ...prev };
                            delete next[targetSessionId];
                            return next;
                        });
                    }
                },
                // onError: 错误处理
                (error: Error) => {
                    console.error('流式请求错误:', error);
                    // 🛡️ 错误时也要解除锁定
                    globalSendingLockRef.current = false;
                    isSendingRef.current[targetSessionId] = false;
                    isSendingRef.current['new'] = false;
                    // 清除取消函数引用
                    cancelStreamRef.current = null;

                    setSessions(prev => {
                        const session = prev.find(s => s.id === targetSessionId);
                        if (session) {
                            const errorMsg = {
                                id: uuidv4(),
                                role: 'model' as const,
                                text: error.message || "抱歉，服务调用失败。",
                                timestamp: new Date()
                            };
                            const newMsgs = [...session.messages, errorMsg];
                            return prev.map(s => s.id === targetSessionId ? { ...s, messages: newMsgs, lastModified: new Date() } : s);
                        }
                        return prev;
                    });
                },
                // onComplete: 流式结束
                () => {
                    console.log('✅ [流式] 完成，总长度=', accumulatedText.length);
                    // Clear loading
                    setSessionLoadingStates(prev => {
                        const next = { ...prev };
                        delete next[targetSessionId];
                        return next;
                    });
                    // 🛡️ 解除所有锁定
                    globalSendingLockRef.current = false;
                    isSendingRef.current[targetSessionId] = false;
                    isSendingRef.current['new'] = false;
                    // 清除取消函数引用
                    cancelStreamRef.current = null;
                }
            );

            // 🛑 保存取消函数引用，以便用户可以取消请求
            cancelStreamRef.current = cancelStream;

        } catch (e: any) {
            console.error(e);
            // 错误已经在 onError 中处理
            setSessionLoadingStates(prev => {
                const next = { ...prev };
                delete next[targetSessionId];
                return next;
            });
            // 🛡️ 解除所有锁定
            globalSendingLockRef.current = false;
            isSendingRef.current[targetSessionId] = false;
            isSendingRef.current['new'] = false;
        }
    })();

    // Helper to update persona for a specific session
    const updateSessionPersona = (sessionId: string, newPersona: UserPersona, newScore?: number) => {
        setSessions(prev => prev.map(s => {
            if (s.id === sessionId) {
                return { 
                    ...s, 
                    persona: newPersona,
                    healthScore: newScore !== undefined ? newScore : s.healthScore
                };
            }
            return s;
        }));
    };

    await (async () => {
        try {
             // Use conversation_id in the request to enable backend persistence
             const res = await chatApi.analyzePersona({
                  user_id: user.id,
                  text: userMsg.text,
                  // Removed current_persona: backend now fetches this from DB via conversation_id
                  conversation_id: targetSessionId, 
                  model_configuration: {
                      provider_id: selectedModel.providerId || "",
                      model_id: selectedModel.realId || "",
                      model_name: selectedModel.modelName || selectedModel.id,
                      temperature: 0.1,
                      top_p: 0.95,
                      max_tokens: 512
                  }
              });

             if (res && res.success && res.data) {
                 console.log('Raw backend response:', res.data);
                 const newPersona = { ...persona, ...res.data };
                 
                 const changes: string[] = [];
                 (Object.keys(newPersona) as Array<keyof UserPersona>).forEach(key => {
                     if (newPersona[key] !== persona[key]) changes.push(key);
                 });
                
                if (changes.length > 0) {
                    // Update Session State
                    updateSessionPersona(targetSessionId, newPersona);
                    
                    // UI feedback
                    setChangedFields(changes);
                    setTimeout(() => setChangedFields([]), 3000);
                }
             }
        } catch (e) {
            console.error("Persona analysis failed in parallel", e);
        } finally {
        }
    })();
  };

  const updateSessionMessages = (sessionId: string, newMessages: ChatMessage[]) => {
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: newMessages, lastModified: new Date() } : s));
  };

  // Only used for Quick Config Modal - Saves to Provider Config now
  const saveQuickConfig = async (apiKey: string) => {
    // 1. Update Local State
    setProviderConfigs(prev => ({
        ...prev,
        [selectedProviderId]: {
            ...prev[selectedProviderId],
            apiKey: apiKey,
            enabled: true
        }
    }));
    // Persist to local storage
    const current = JSON.parse(localStorage.getItem('user_provider_configs') || '{}');
    current[selectedProviderId] = { ...current[selectedProviderId], apiKey, enabled: true };
    localStorage.setItem('user_provider_configs', JSON.stringify(current));

    // 2. Sync to Backend
    const provider = apiData.find((p: any) => p.name === selectedProviderId);
    if (provider) {
        try {
            await providerApi.update({
                provider_id: provider.id,
                api_key: apiKey,
                is_enabled: true
            });
        } catch (e) {
            console.error("Failed to sync API key to backend", e);
        }
    }
  };

  const groupSessions = groupSessionsByDate(sessions.filter(s => s.title.toLowerCase().includes(searchQuery.toLowerCase())));

  const toggleTheme = () => {
      setIsDarkMode(!isDarkMode);
  };

  return (
    <div className="h-screen w-full flex bg-rice-paper overflow-hidden transition-colors duration-500">
      <LogoutConfirmModal 
        isOpen={showLogoutModal} 
        onConfirm={() => { setShowLogoutModal(false); onLogout(); }} 
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
      
      {/* 快速配置小窗 (仅在对话拦截时显示) */}
      {isQuickConfigOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white dark:bg-tcm-charcoal w-full max-w-sm rounded-3xl shadow-2xl p-6 border border-tcm-gold/30 scale-in-center">
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="p-4 bg-tcm-gold/10 rounded-full text-tcm-gold">
                <Icons.Zap size={32} />
              </div>
              <h3 className="text-lg font-bold text-tcm-darkGreen dark:text-tcm-cream font-serif-sc">需配置服务商</h3>
              <p className="text-sm text-gray-500">您选中的模型属于 <b>{currentProviders.find(p => p.id === selectedProviderId)?.name}</b>，需要配置 API Key 才能使用。</p>
              
              <div className="w-full space-y-3 pt-2">
                <input 
                  type="password" 
                  placeholder="输入 Provider API Key" 
                  autoFocus
                  className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-xl outline-none focus:ring-2 focus:ring-tcm-lightGreen transition-all text-sm"
                  onChange={(e) => saveQuickConfig(e.target.value)}
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

      {/* 1. LEFT SIDEBAR */}
      <aside
        className={`${showLeftSidebar ? 'w-[280px]' : 'w-0'} flex-shrink-0 bg-[#f0f4f9]/80 dark:bg-[#1e1e1e]/80 backdrop-blur-md border-r border-tcm-lightGreen/10 flex flex-col transition-all duration-300 overflow-hidden relative z-30`}
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between p-4 pb-2">
            <button
               onClick={() => setShowLeftSidebar(false)}
               className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-white/10 rounded-full transition-colors"
            >
              <Icons.Menu size={20} />
            </button>

            <button
               onClick={() => setIsSearchActive(!isSearchActive)}
               className={`p-2 rounded-full transition-colors ${isSearchActive ? 'bg-gray-200 dark:bg-white/10 text-gray-900 dark:text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-white/10'}`}
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
                   onChange={(e) => setSearchQuery(e.target.value)}
                   className="w-full bg-white dark:bg-black/20 border border-transparent dark:border-white/10 rounded-full py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-gray-200 dark:focus:ring-gray-700 shadow-sm"
                 />
                 <div className="absolute left-3 top-2.5 text-gray-500">
                   <Icons.Activity size={16} />
                 </div>
              </div>
            ) : (
              <button
                onClick={handleNewChat}
                className="flex items-center gap-3 px-4 py-3 bg-[#dde3ea] dark:bg-[#2a2a2a] text-gray-700 dark:text-gray-200 rounded-2xl hover:bg-[#d0d7de] dark:hover:bg-[#333] transition-colors w-full shadow-sm"
              >
                <Icons.Edit3 size={18} />
                <span className="text-sm font-medium">开启新诊疗</span>
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto px-2 py-2 custom-scrollbar space-y-4">
             {Object.entries(groupSessions).map(([group, groupSessions]) => (
               groupSessions.length > 0 && (
                 <div key={group} className="animate-in fade-in">
                   <div className="px-4 py-2 text-xs font-bold text-gray-500 dark:text-gray-400">{group}</div>
                   {groupSessions.map(session => (
                      <div
                        key={session.id}
                        onClick={() => setActiveSessionId(session.id)}
                        className={`group flex items-center gap-3 px-4 py-2 mx-2 rounded-full cursor-pointer transition-colors relative ${
                          activeSessionId === session.id 
                            ? 'bg-tcm-lightGreen/20 dark:bg-tcm-lightGreen/10 text-tcm-darkGreen dark:text-tcm-freshGreen font-medium' 
                            : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-white/5'
                        }`}
                      >
                        <Icons.MessageSquare size={16} className="flex-shrink-0 opacity-70" />
                        <div className="flex-1 min-w-0 text-sm truncate pr-6">{session.title}</div>
                        <button
                            onClick={(e) => handleDeleteSession(session.id, e)}
                            className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-all"
                            title="删除会话"
                        >
                            <Icons.Trash2 size={14} />
                        </button>
                      </div>
                   ))}
                 </div>
               )
             ))}
          </div>
          {/* Bottom: Settings / User */}
          <div className="p-2 border-t border-gray-200 dark:border-white/5 relative bg-white/50 dark:bg-black/20">
             {showUserMenu && (
                <div className="absolute bottom-full left-2 right-2 mb-2 bg-[#f0f4f9] dark:bg-[#1e1e1e] rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 overflow-hidden animate-in slide-in-from-bottom-2 z-40">
                  <button onClick={() => { setIsEditingProfile(true); setShowUserMenu(false); }} className="w-full text-left px-4 py-3 hover:bg-gray-200 dark:hover:bg-white/10 flex items-center gap-3 text-sm text-gray-700 dark:text-gray-200 transition-colors">
                    <Icons.Edit3 size={16} className="text-tcm-lightGreen"/> 个人资料设置
                  </button>
                  {/* 跳转到新的模型管理页面 */}
                  <button onClick={() => { navigate('/public/models'); setShowUserMenu(false); }} className="w-full text-left px-4 py-3 hover:bg-gray-200 dark:hover:bg-white/10 flex items-center gap-3 text-sm text-gray-700 dark:text-gray-200 transition-colors">
                    <Icons.Settings size={16} className="text-tcm-gold"/> 模型管理配置
                  </button>
                  <button onClick={toggleTheme} className="w-full text-left px-4 py-3 hover:bg-gray-200 dark:hover:bg-white/10 flex items-center gap-3 text-sm text-gray-700 dark:text-gray-200 transition-colors">
                    {isDarkMode ? <Icons.Sun size={16} className="text-yellow-500" /> : <Icons.Moon size={16} className="text-indigo-400" />}
                    {isDarkMode ? '切换到浅色模式' : '切换到深色模式'}
                  </button>
                  <button onClick={() => { setShowLogoutModal(true); setShowUserMenu(false); }}  className="w-full text-left px-4 py-3 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-3 text-sm text-red-600 dark:text-red-400 border-t border-gray-200 dark:border-gray-700 transition-colors">
                    <Icons.LogOut size={16}/> 退出账号
                  </button>
                </div>
             )}

             <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className={`flex items-center gap-3 w-full p-3 rounded-full hover:bg-gray-200 dark:hover:bg-white/5 transition-colors ${showUserMenu ? 'bg-gray-200 dark:bg-white/5' : ''}`}
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

      {/* 2. MIDDLE AREA */}
      <main className="flex-1 flex flex-col relative z-10 transition-colors duration-500 min-w-0 bg-transparent">
        <header className="h-16 flex items-center justify-between px-6 z-10 transition-colors bg-white/80 dark:bg-[#131314]/80 backdrop-blur-sm border-b border-tcm-lightGreen/5">
          <div className="flex items-center gap-3">
             {!showLeftSidebar && (
               <button
                 onClick={() => setShowLeftSidebar(true)}
                 className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-white/5 rounded-full transition-colors"
               >
                 <Icons.Menu size={20} />
               </button>
             )}

             {/* 仁术 Logo Dropdown 实现供应商切换 */}
             <div className="relative">
               <div 
                  onClick={() => setShowProviderSelector(!showProviderSelector)}
                  className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-white/5 px-3 py-1.5 rounded-lg transition-colors group"
                >
                  <BrandLogo size="sm" showText={true} />
                  <Icons.ChevronDown size={14} className={`text-gray-400 transition-transform duration-300 ${showProviderSelector ? 'rotate-180' : ''}`} />
               </div>

               {showProviderSelector && (
                 <>
                   <div className="fixed inset-0 z-40" onClick={() => setShowProviderSelector(false)}></div>
                   <div className="absolute top-full left-0 mt-2 w-64 bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 origin-top-left">
                     <div className="p-1.5 space-y-0.5 max-h-64 overflow-y-auto custom-scrollbar">
                        <div className="px-2 py-1.5 text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider flex justify-between">
                            <span>选择模型提供商</span>
                            <span></span>
                        </div>
                        {currentProviders.map((p: any) => (
                          <button
                            key={p.id}
                            onClick={() => {
                              setSelectedProviderId(p.id);
                              setShowProviderSelector(false);
                            }}
                           className={`w-full text-left p-2 rounded-lg flex items-center gap-3 transition-colors ${
                             selectedProviderId === p.id 
                               ? 'bg-tcm-lightGreen/10 border border-tcm-lightGreen/20 text-tcm-darkGreen dark:text-emerald-300' 
                               : 'hover:bg-gray-50 dark:hover:bg-white/5 text-gray-700 dark:text-gray-200'
                           }`}
                         >
                           <span className="text-lg w-5 h-5 flex items-center justify-center">
                             {(() => {
                               const iconPath = getProviderIconPath(p.id);
                               const shouldInvert = isDarkInvert(p.id);
                               return iconPath ? <img src={iconPath} alt={p.name} className={`w-full h-full object-contain ${shouldInvert ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} /> : p.icon;
                             })()}
                           </span>
                           <div className="flex flex-col">
                               <span className="text-xs font-bold">{p.name}</span>
                               {!p.isBuiltin && <span className="text-[9px] text-tcm-darkGreen dark:text-tcm-lightGreen bg-tcm-lightGreen/10 px-1 py-0.5 rounded w-fit mt-0.5">我的服务</span>}
                           </div>
                           {selectedProviderId === p.id && <Icons.Check size={14} className="ml-auto" />}
                         </button>
                       ))}
                       {currentProviders.length === 0 && (
                          <div className="p-4 text-center text-xs text-gray-400">
                             暂无提供商
                          </div>
                       )}
                     </div>
                   </div>
                 </>
               )}
             </div>
          </div>

          <div className="flex items-center gap-2">
             <div className="relative">
                <button
                  onClick={() => setShowModelSelector(!showModelSelector)}
                  className={`flex items-center justify-between gap-2 w-52 bg-white dark:bg-white/5 border px-3 py-1.5 rounded-lg text-xs font-bold text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-white/10 transition-colors shadow-sm ${showModelSelector ? 'border-tcm-lightGreen ring-2 ring-tcm-lightGreen/20' : 'border-gray-200 dark:border-white/10'}`}
                >
                    <div className="flex items-center gap-2 truncate">
                      {(() => {
                          const modelIcon = getModelIconPath(selectedModel.id);
                          if (modelIcon) return <img src={modelIcon} alt={selectedModel.name} className={`w-4 h-4 object-contain ${isDarkInvert(selectedModel.id) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} />;
                          return <Icons.Zap size={14} className="text-tcm-gold flex-shrink-0" />;
                      })()}
                      <span className="truncate">{selectedModel.name}</span>
                      {!selectedModel.isBuiltin && <span className="text-[9px] text-tcm-darkGreen dark:text-emerald-300 bg-tcm-lightGreen/10 px-1 py-0.5 rounded w-fit flex-shrink-0">me</span>}
                    </div>
                    <Icons.ChevronDown size={12} className={`text-gray-400 flex-shrink-0 transition-transform duration-300 ${showModelSelector ? 'rotate-180' : ''}`} />
                </button>

                {/* 参数设置按钮已移除，统一在下拉列表中通过滑块图标进行配置 */}

                {showModelSelector && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setShowModelSelector(false)}></div>
                    <div className="absolute top-full right-0 mt-2 w-full bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 origin-top-right">
                      <div className="p-1.5 space-y-0.5">
                        <div className="px-2 py-1.5 text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider">选择已适配模型</div>
                        {filteredModels.length === 0 ? (
                            <div className="p-4 text-center text-xs text-gray-400">
                                暂无已启用模型。<br/>请前往 <span className="text-tcm-lightGreen cursor-pointer" onClick={() => navigate('/public/models')}>模型管理</span> 启用。
                            </div>
                        ) : filteredModels.map(model => (
                          <div
                            key={model.id}
                            className={`w-full rounded-lg transition-colors flex flex-col ${
                              selectedModel.id === model.id 
                                ? 'bg-tcm-lightGreen/10 border border-tcm-lightGreen/20' 
                                : 'hover:bg-gray-50 dark:hover:bg-white/5 border border-transparent'
                            }`}
                          >
                            <div className="flex items-center w-full">
                                <button
                                    onClick={() => {
                                    setSelectedModel(model);
                                    setTemperature(model.defaultTemperature ?? 0.7);
                                    setTopP(model.defaultTopP ?? 1.0);
                                    setMaxTokens(model.defaultMaxTokens ?? 4096);  // 从 2000 提升到 4096
                                    setShowModelSelector(false);
                                    }}
                                    className="flex-1 text-left p-2 flex items-start gap-2 min-w-0"
                                >
                                    <div className="flex flex-col items-center gap-1 min-w-[24px]">
                                        {(() => {
                                            const modelIcon = getModelIconPath(model.id);
                                            if (modelIcon) {
                                                return (
                                                    <div className="mt-0.5 w-6 h-6 flex items-center justify-center flex-shrink-0">
                                                        <img src={modelIcon} alt={model.name} className={`w-5 h-5 object-contain ${isDarkInvert(model.id) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} />
                                                    </div>
                                                );
                                            }
                                            return (
                                                <div className={`mt-0.5 p-1 rounded-md flex-shrink-0 w-6 h-6 flex items-center justify-center ${
                                                    selectedModel.id === model.id 
                                                        ? 'bg-tcm-lightGreen text-white' 
                                                        : 'bg-gray-100 dark:bg-white/10 text-gray-500 dark:text-gray-400'
                                                }`}>
                                                    <Icons.Zap size={12} />
                                                </div>
                                            );
                                        })()}
                                        {!model.isBuiltin && (
                                            <span className="text-[10px] font-bold text-tcm-darkGreen dark:text-tcm-lightGreen bg-tcm-lightGreen/10 px-1 rounded leading-none scale-90 origin-top">
                                                me
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex flex-col">
                                            <div className="flex items-center gap-2">
                                                <div className={`text-xs font-bold truncate ${
                                                    selectedModel.id === model.id 
                                                    ? 'text-tcm-darkGreen dark:text-tcm-lightGreen' 
                                                    : 'text-gray-700 dark:text-gray-200'
                                                }`}>
                                                    {model.name}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="text-[10px] text-gray-400 truncate mt-0.5">
                                            {model.description}
                                        </div>
                                    </div>
                                    {(model as CustomModel).isCustom && <span className="hidden">Custom</span>}
                                </button>
                                
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            if (selectedModel.id !== model.id) {
                                                setSelectedModel(model);
                                                setTemperature(model.defaultTemperature ?? 0.7);
                                                setTopP(model.defaultTopP ?? 1.0);
                                                setMaxTokens(model.defaultMaxTokens ?? 4096);  // 从 2000 提升到 4096
                                            }
                                            // Close model selector and open right sidebar settings
                                            setShowModelSelector(false);
                                            setRightPanelMode('settings');
                                            setShowRightSidebar(true);
                                        }}
                                        className={`p-2 m-1 rounded-md hover:bg-gray-200 dark:hover:bg-white/10 transition-colors flex-shrink-0 ${
                                            rightPanelMode === 'settings' && selectedModel.id === model.id 
                                            ? 'text-tcm-darkGreen bg-tcm-lightGreen/10' 
                                            : 'text-gray-400'
                                        }`}
                                    >
                                        <Icons.Sliders size={14} />
                                    </button>
                                </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  </>
                )}
             </div>

             <button
               onClick={() => setShowRightSidebar(!showRightSidebar)}
               className={`p-2 rounded-full transition-colors ${showRightSidebar ? 'bg-tcm-lightGreen/10 text-tcm-lightGreen' : 'text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5'}`}
             >
               <Icons.Activity size={20} />
             </button>
          </div>
        </header>

        {/* Chat Stream */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scroll-smooth">
          <div className="max-w-4xl mx-auto w-full space-y-8">
            {currentSession ? (
            // Show messages if there is a session
            currentSession.messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in-up group w-full relative`}>
                  {msg.role === 'model' && (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-tcm-lightGreen to-tcm-darkGreen text-white flex items-center justify-center shadow-lg mr-4 flex-shrink-0 mt-1 overflow-hidden">
                      <Icons.Zap size={14} />
                    </div>
                  )}
                  <div className={`space-y-2 max-w-[85%] relative group-hover:z-10`}>
                    {msg.role === 'user' ? (
                       <div className="px-5 py-3 text-gray-800 dark:text-white leading-relaxed whitespace-pre-wrap">
                          {msg.attachments?.map((att, idx) => (
                            <img key={idx} src={att.url} alt="att" className="h-32 rounded-lg border mb-2" />
                          ))}
                          {msg.text}
                       </div>
                    ) : (
                       <div className="text-gray-800 dark:text-gray-200 leading-relaxed">
                          {/* Agent 步骤指示器 */}
                          {msg.agentSteps && msg.agentSteps.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mb-3">
                              {msg.agentSteps.map((step, idx) => {
                                const isLast = idx === msg.agentSteps!.length - 1;
                                const isActive = isLast && !msg.text;
                                return (
                                  <span key={idx} className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                                    ${isActive
                                      ? 'bg-tcm-lightGreen/20 text-tcm-darkGreen dark:text-tcm-lightGreen animate-pulse'
                                      : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
                                    }`}>
                                    {isActive ? '⟳' : '✓'} {step}
                                  </span>
                                );
                              })}
                            </div>
                          )}
                          {msg.text ? (
                            <div className="prose prose-sm md:prose-base max-w-none dark:prose-invert
                              prose-p:my-2 prose-p:leading-relaxed
                              prose-headings:text-tcm-darkGreen dark:prose-headings:text-tcm-lightGreen
                              prose-strong:text-tcm-darkGreen dark:prose-strong:text-tcm-lightGreen
                              prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
                              prose-pre:bg-gray-900 prose-pre:text-gray-100
                              prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                              prose-blockquote:border-tcm-lightGreen prose-blockquote:bg-tcm-lightGreen/5
                              prose-a:text-tcm-darkGreen hover:prose-a:text-tcm-lightGreen">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {msg.text}
                              </ReactMarkdown>
                            </div>
                          ) : (
                            // 正在生成中：显示 loading 动画
                            <div className="flex items-center gap-1.5 py-1">
                               <div className="w-2 h-2 bg-tcm-lightGreen rounded-full animate-bounce"></div>
                               <div className="w-2 h-2 bg-tcm-gold rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                               <div className="w-2 h-2 bg-tcm-darkGreen rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                          )}
                       </div>
                    )}
                    
                    {/* Message Actions */}
                    <div className={`absolute top-0 ${msg.role === 'user' ? '-left-8' : '-right-8'} opacity-0 group-hover:opacity-100 transition-opacity flex flex-col gap-1`}>
                        <button
                            onClick={() => handleDeleteMessage(msg.id)}
                            className="p-1 text-gray-400 hover:text-red-500 bg-white dark:bg-black/20 rounded shadow-sm"
                            title="删除消息"
                        >
                            <Icons.Trash2 size={12} />
                        </button>
                    </div>
                  </div>
                  {msg.role === 'user' && (
                    <img src={user.avatar} className="w-8 h-8 rounded-full border border-gray-200 dark:border-gray-600 ml-3 shadow-sm object-cover flex-shrink-0 mt-1" alt="Me" />
                  )}
                </div>
            ))
            ) : (
                // Show Landing Page Content when no session is active
                <div className="flex flex-col items-center justify-center min-h-[50vh] animate-in fade-in slide-in-from-bottom-4">
                    <div className="mb-8 p-4 bg-white/50 dark:bg-white/5 rounded-full shadow-sm backdrop-blur-sm">
                        <BrandLogo size="lg" showText={false} />
                    </div>
                    <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-8 font-serif-sc">有什么我能帮你的吗？</h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full max-w-3xl">
                        {[
                            { icon: <Icons.Thermometer size={18} />, text: "我最近感觉有点上火，喉咙痛怎么办？" },
                            { icon: <Icons.Activity size={18} />, text: "分析一下我的体质健康状况" },
                            { icon: <Icons.BookOpen size={18} />, text: "解释一下'气虚'是什么意思？" },
                            { icon: <Icons.Coffee size={18} />, text: "推荐一些适合春季的养生茶饮" },
                            { icon: <Icons.Moon size={18} />, text: "最近失眠多梦，有什么调理建议？" },
                            { icon: <Icons.FileText size={18} />, text: "帮我解读一下这个体检报告" }
                        ].map((item, idx) => (
                            <button
                                key={idx}
                                onClick={() => {
                                    // 🛡️ 额外的点击保护：检查是否正在发送
                                    if (globalSendingLockRef.current) {
                                        console.log('⚠️ [Landing按钮] 请求进行中，忽略点击');
                                        return;
                                    }
                                    handleSendMessage(item.text);
                                }}
                                className="flex items-center gap-3 p-4 bg-white dark:bg-white/5 hover:bg-gray-50 dark:hover:bg-white/10 border border-gray-100 dark:border-white/5 rounded-2xl shadow-sm hover:shadow-md transition-all text-left group"
                            >
                                <div className="p-2 bg-tcm-lightGreen/10 text-tcm-darkGreen dark:text-tcm-lightGreen rounded-lg group-hover:bg-tcm-lightGreen/20 transition-colors">
                                    {item.icon}
                                </div>
                                <span className="text-sm text-gray-600 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white transition-colors">{item.text}</span>
                            </button>
                        ))}
                    </div>
                </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-2 z-20">
          <div className="max-w-4xl mx-auto">
             {attachments.length > 0 && (
                <div className="flex gap-3 px-4 py-2 overflow-x-auto custom-scrollbar">
                  {attachments.map((att, idx) => (
                      <div key={idx} className="relative group flex-shrink-0">
                        <img src={att.previewUrl} alt="preview" className="h-12 w-12 rounded-lg object-cover border border-gray-200 dark:border-gray-600 shadow-sm" />
                        <button
                          onClick={() => setAttachments(attachments.filter((_, i) => i !== idx))}
                          className="absolute -top-1 -right-1 bg-gray-800 text-white rounded-full p-0.5 shadow-md opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Icons.X size={10} />
                        </button>
                      </div>
                  ))}
                </div>
             )}

             <div className="bg-[#f0f4f9] dark:bg-[#1e1e1e] rounded-full p-3 flex items-center gap-2 transition-all duration-300 shadow-sm border border-gray-200 dark:border-gray-700 ring-1 ring-transparent focus-within:ring-tcm-lightGreen/30 focus-within:bg-white dark:focus-within:bg-[#252525]">
                <button
                    onClick={() => fileInputRef.current?.click()}
                    className="p-1.5 text-gray-400 hover:text-tcm-darkGreen dark:text-gray-500 dark:hover:text-tcm-lightGreen transition-colors rounded-full hover:bg-gray-200/50 dark:hover:bg-white/10"
                    title="Upload"
                >
                  <input type="file" multiple ref={fileInputRef} className="hidden" onChange={handleFileUpload} />
                  <Icons.Paperclip size={20} />
                </button>

                <button
                    onClick={handleVoiceInput}
                    className="p-1.5 text-gray-400 hover:text-tcm-darkGreen dark:text-gray-500 dark:hover:text-tcm-lightGreen transition-colors rounded-full hover:bg-gray-200/50 dark:hover:bg-white/10"
                    title="语音输入"
                >
                  <Icons.Mic size={20} />
                </button>

                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder={selectedModel.id ? "输入健康咨询问题..." : "请先选择你要使用的模型..."}
                  className="flex-1 bg-transparent border-none focus:ring-0 focus:outline-none text-gray-800 dark:text-gray-100 placeholder-gray-400 resize-none py-1.5 text-sm max-h-32"
                  rows={1}
                  style={{ minHeight: '32px' }}
                />

                <div className="flex items-center gap-1 border-l border-gray-300 dark:border-gray-700 pl-2">
                    {/* Deep Thinking 按钮 - 仅当模型支持 thinking 时显示 */}
                    {selectedModel.supportsThinking && (
                      <button
                        onClick={() => setEnableThinking(!enableThinking)}
                        className={`p-1.5 rounded-full transition-all ${
                          enableThinking 
                            ? 'text-tcm-darkGreen bg-tcm-lightGreen/20 dark:text-tcm-lightGreen' 
                            : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
                        }`}
                        title={enableThinking ? "关闭深度思考" : "开启深度思考"}
                      >
                         <Icons.BrainCircuit size={18} />
                      </button>
                    )}
                </div>
                
                {/* 发送/取消按钮 - 根据状态切换 */}
                {isCurrentSessionLoading ? (
                  <button
                      onClick={handleCancelGeneration}
                      className="p-1.5 rounded-full transition-all duration-300 bg-red-500 text-white shadow-md hover:bg-red-600 transform hover:scale-105"
                      title="取消生成"
                  >
                      <Icons.X size={18} />
                  </button>
                ) : (
                  <button
                      onClick={() => handleSendMessage()}
                      disabled={!inputValue.trim() && attachments.length === 0}
                      className={`p-1.5 rounded-full transition-all duration-300 ${
                        inputValue.trim() || attachments.length > 0
                          ? 'bg-tcm-darkGreen text-white shadow-md hover:bg-tcm-lightGreen transform hover:scale-105'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                      }`}
                      title="发送消息"
                  >
                      <Icons.Send size={18} />
                  </button>
                )}
             </div>
          </div>
        </div>
      </main>

      {/* 3. RIGHT SIDEBAR */}
      <aside className={`${showRightSidebar ? 'w-80' : 'w-0'} flex-shrink-0 flex flex-col ${isDarkMode ? 'glass-panel-dark' : 'glass-panel'} border-l border-white/50 dark:border-white/10 z-20 shadow-xl transition-all duration-500 overflow-hidden`}>
        <div className="h-16 flex-shrink-0 flex items-center justify-between px-6 border-b border-gray-200/50 dark:border-white/10 bg-white/30 dark:bg-black/10">
           <div className="flex items-center">
             {rightPanelMode === 'settings' ? <Icons.Sliders className="text-tcm-darkGreen mr-3" size={20} /> : <Icons.Leaf className="text-tcm-lightGreen mr-3" size={24} />}
             <h1 className="text-lg font-bold text-tcm-darkGreen dark:text-tcm-cream font-serif-sc tracking-wide whitespace-nowrap">
               {rightPanelMode === 'settings' ? '模型参数配置' : '智能健康画像'}
             </h1>
           </div>
           {rightPanelMode === 'settings' && (
             <button onClick={() => setRightPanelMode('health')} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
               <Icons.X size={18} />
             </button>
           )}
        </div>

        <div className="flex-1 overflow-y-auto p-6 relative w-80">
           <div className="space-y-4 relative z-10">
              {rightPanelMode === 'settings' ? (
                 <div className="bg-white/60 dark:bg-white/5 p-4 rounded-xl border border-white dark:border-white/10 shadow-sm backdrop-blur-sm space-y-6 animate-in fade-in slide-in-from-right-4">
                    <div>
                        <div className="flex justify-between text-xs font-bold mb-2">
                            <span className="text-gray-600 dark:text-gray-300">随机性 (Temperature)</span>
                            <span className="text-tcm-darkGreen dark:text-tcm-lightGreen font-mono">{temperature}</span>
                        </div>
                        <input 
                            type="range" 
                            min="0" 
                            max="2" 
                            step="0.1" 
                            value={temperature} 
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-darkGreen"
                        />
                        <p className="text-[10px] text-gray-400 mt-1">值越高，回复越具有创造性；值越低，回复越保守准确。</p>
                    </div>

                    <div>
                        <div className="flex justify-between text-xs font-bold mb-2">
                            <span className="text-gray-600 dark:text-gray-300">核采样 (Top P)</span>
                            <span className="text-tcm-darkGreen dark:text-tcm-lightGreen font-mono">{topP}</span>
                        </div>
                        <input 
                            type="range" 
                            min="0" 
                            max="1" 
                            step="0.05" 
                            value={topP} 
                            onChange={(e) => setTopP(parseFloat(e.target.value))}
                            className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-darkGreen"
                        />
                        <p className="text-[10px] text-gray-400 mt-1">控制模型选择候选词的范围，较低的值会使回复更加专注。</p>
                    </div>

                    <div>
                        <div className="flex justify-between text-xs font-bold mb-2">
                            <span className="text-gray-600 dark:text-gray-300">最大Token数</span>
                            <span className="text-tcm-darkGreen dark:text-tcm-lightGreen font-mono">{maxTokens}</span>
                        </div>
                        <input 
                            type="range" 
                            min="100" 
                            max="8000" 
                            step="100" 
                            value={maxTokens} 
                            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                            className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-darkGreen"
                        />
                        <p className="text-[10px] text-gray-400 mt-1">限制模型单次回复生成的最大长度。</p>
                    </div>
                    
                    <div className="pt-4 border-t border-gray-200 dark:border-white/10">
                        <button 
                          onClick={() => {
                            // Reset to defaults
                            setTemperature(selectedModel.defaultTemperature ?? 0.7);
                            setTopP(selectedModel.defaultTopP ?? 1.0);
                            setMaxTokens(selectedModel.defaultMaxTokens ?? 4096);  // 从 2000 提升到 4096
                          }}
                          className="w-full py-2 text-xs text-gray-500 hover:text-tcm-darkGreen dark:text-gray-400 dark:hover:text-tcm-lightGreen transition-colors flex items-center justify-center gap-2"
                        >
                            <Icons.RotateCcw size={12} /> 恢复默认设置
                        </button>
                    </div>
                 </div>
              ) : (
                <>
                  <div className="bg-white/60 dark:bg-white/5 p-4 rounded-xl border border-white dark:border-white/10 shadow-sm backdrop-blur-sm">
                <div className="flex justify-between items-end mb-2">
                  <span className="text-sm font-bold text-gray-600 dark:text-gray-300">体质健康分</span>
                  <span className="text-3xl font-serif-sc font-bold text-tcm-darkGreen dark:text-tcm-lightGreen">{healthScore}</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-tcm-lightGreen to-tcm-gold transition-all duration-1000" style={{width: `${healthScore}%`}}></div>
                </div>
              </div>

              {(Object.keys(persona) as Array<keyof UserPersona>).map((key) => {
                // Skip displaying certain keys if needed, or format them
                if (key === 'base_profile' || key === 'health_score') return null;
                return (
                <div
                  key={key}
                  className={`p-3 rounded-lg border transition-all duration-700 ${
                    changedFields.includes(key) 
                      ? 'bg-tcm-gold/20 border-tcm-gold transform scale-105' 
                      : 'bg-white/40 dark:bg-white/5 border-transparent hover:border-tcm-lightGreen/30'
                  }`}
                >
                  <div className="text-[10px] text-gray-500 dark:text-gray-400 uppercase mb-1 flex items-center gap-1">
                    {PERSONA_FIELD_LABELS[key] || key.replace(/([A-Z])/g, ' $1').trim()}
                    {changedFields.includes(key) && <span className="w-2 h-2 bg-tcm-accent rounded-full animate-ping"></span>}
                  </div>
                  <div className="font-serif-sc text-tcm-darkGreen dark:text-tcm-cream text-sm font-medium leading-relaxed">
                    {persona[key] as string || "未录入"}
                  </div>
                </div>
              )})}
              
              {/* Base Profile Display */}
              <div className="bg-white/40 dark:bg-white/5 p-3 rounded-lg border border-transparent hover:border-tcm-lightGreen/30 transition-all">
                  <div className="text-[10px] text-gray-500 dark:text-gray-400 uppercase mb-2 border-b border-gray-200 dark:border-white/10 pb-1">
                      基础健康画像
                  </div>
                  <div className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                      {persona.base_profile && (persona.base_profile.constitution_type || persona.base_profile.medical_history || persona.base_profile.allergy_info || (persona.base_profile.taboo_items && persona.base_profile.taboo_items.length > 0)) ? (
                        <>
                          {persona.base_profile.constitution_type && <div><span className="opacity-70">体质:</span> {persona.base_profile.constitution_type}</div>}
                          {persona.base_profile.medical_history && <div><span className="opacity-70">病史:</span> {persona.base_profile.medical_history}</div>}
                          {persona.base_profile.allergy_info && <div><span className="opacity-70">过敏:</span> {persona.base_profile.allergy_info}</div>}
                          {persona.base_profile.taboo_items && persona.base_profile.taboo_items.length > 0 && <div><span className="opacity-70">禁忌:</span> {persona.base_profile.taboo_items.join(', ')}</div>}
                        </>
                      ) : (
                        <div className="text-gray-400 italic">暂无基础数据，请在个人资料中完善</div>
                      )}
                  </div>
              </div>
           </>
        )}
        </div>
      </div>
      </aside>

      {/* Edit Profile Modal */}
      {isEditingProfile && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
           <div className="bg-white dark:bg-tcm-charcoal rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden animate-fade-in-up border border-tcm-lightGreen/20">
              <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
                 <h2 className="text-xl font-bold text-tcm-darkGreen dark:text-tcm-cream font-serif-sc">修改健康档案</h2>
                 <button onClick={() => setIsEditingProfile(false)} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"><Icons.X /></button>
              </div>
              <div className="p-6 space-y-4 max-h-[60vh] overflow-y-auto">
                 <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase mb-1">年龄</label>
                      <input type="text" value={editPersonaForm.age} onChange={e => setEditPersonaForm({...editPersonaForm, age: e.target.value})} className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg dark:text-white focus:ring-1 focus:ring-tcm-lightGreen outline-none" />
                    </div>
                    <div>
                      <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase mb-1">性别</label>
                      <input type="text" value={editPersonaForm.gender} onChange={e => setEditPersonaForm({...editPersonaForm, gender: e.target.value})} className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg dark:text-white focus:ring-1 focus:ring-tcm-lightGreen outline-none" />
                    </div>
                 </div>
                 <div>
                    <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase mb-1">既往病史</label>
                    <textarea 
                      value={editPersonaForm.base_profile?.medical_history || ''} 
                      onChange={e => setEditPersonaForm({
                        ...editPersonaForm, 
                        base_profile: {
                          ...(editPersonaForm.base_profile || {}),
                          medical_history: e.target.value
                        }
                      })} 
                      className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg h-24 dark:text-white focus:ring-1 focus:ring-tcm-lightGreen outline-none" 
                    />
                 </div>
              </div>
              <div className="p-6 bg-gray-50 dark:bg-black/10 flex justify-end gap-3">
                 <button onClick={() => setIsEditingProfile(false)} className="px-6 py-2 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-white/5 rounded-lg">取消</button>
                 <button onClick={() => { 
                     // Update current session persona
                     if (activeSessionId) {
                         setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, persona: editPersonaForm } : s));
                     }
                     setIsEditingProfile(false); 
                 }} className="px-6 py-2 bg-tcm-darkGreen text-white rounded-lg hover:bg-tcm-lightGreen shadow-lg">保存更改</button>
              </div>
           </div>
        </div>
      )}
    </div>
  );
};



export default PublicPortal;
