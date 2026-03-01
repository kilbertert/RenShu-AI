
import React, { useState, useEffect, useMemo, useLayoutEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Icons } from '../../components/common/Icons';
import { ProviderConfig, CustomModel } from '../../types';
import { DeleteConfirmToast } from '../../components/common/DeleteConfirmToast';
import { Toast } from '../../components/common/Toast';
import { providerApi, modelApi } from '../../api/modules/model';
import { ModelProviderCreate, ModelProviderUpdate, ModelConfigDelete, ModelProviderDelete, ModelConfigCreate, ModelConfigUpdate, ProviderApiKeyVerify } from '../../api/types';
import { MODEL_TYPE_CATEGORIES, SUPPORTED_MODEL_TYPES, MODEL_CONFIG_TYPES, MODEL_FEATURES, FEATURE_COLORS } from '../../constants/models';
import { getProviderIconPath, getModelIconPath, isDarkInvert } from '../../utils/iconMap';

const PublicModelManagementPage: React.FC = () => {
  const navigate = useNavigate();

  // ==================== Theme Initialization ====================
  const [isDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme_public');
    return saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  useLayoutEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // ==================== 核心数据状态 ====================
  const [apiProviders, setApiProviders] = useState<any[]>([]);
  const [selectedProviderId, setSelectedProviderId] = useState<string>('');
  const [selectedModelType, setSelectedModelType] = useState<string>('all');

  // ==================== 持久化状态（localStorage） ====================
  const [providerConfigs, setProviderConfigs] = useState<Record<string, ProviderConfig>>(() => {
    const saved = localStorage.getItem('user_provider_configs');
    return saved ? JSON.parse(saved) : {};
  });

  const [enabledModels, setEnabledModels] = useState<string[]>(() => {
    const saved = localStorage.getItem('user_enabled_models');
    return saved ? JSON.parse(saved) : [];
  });

  const [customModels, setCustomModels] = useState<CustomModel[]>(() => {
    const saved = localStorage.getItem('user_custom_models');
    return saved ? JSON.parse(saved) : [];
  });

  const [customProviders, setCustomProviders] = useState<any[]>(() => {
    const saved = localStorage.getItem('user_custom_providers');
    return saved ? JSON.parse(saved) : [];
  });

  // ==================== UI交互状态 ====================
  const [searchProvider, setSearchProvider] = useState('');
  const [searchModel, setSearchModel] = useState('');
  const [showKey, setShowKey] = useState(false);

  // API Key验证状态
  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [verifyMessage, setVerifyMessage] = useState('');
  const [checkModelId, setCheckModelId] = useState<string>('');

  // ==================== 模态框状态 ====================
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<{ type: 'provider' | 'model', id: string } | null>(null);
  const [showAddProviderModal, setShowAddProviderModal] = useState(false);
  const [showAddModelModal, setShowAddModelModal] = useState(false);
  const [editingModelId, setEditingModelId] = useState<string | null>(null);
  
  // Refs for debounce (Map of field -> timeout)
  const updateTimeoutsRef = React.useRef<Map<string, NodeJS.Timeout>>(new Map());

  // ==================== 表单状态 ====================
  const [newProviderForm, setNewProviderForm] = useState({
    nameId: '',
    label: '',
    description: '',
    defaultBaseUrl: '',
    supportedTypes: [] as string[],
    isEnabled: true
  });

  const [modelForm, setModelForm] = useState<{
    id: string;
    label: string;
    type: string;
    description: string;
    features: string[];
    contextWindow: number | undefined;
    maxTokens: number | undefined;
    temperature: number;
    topP: number;
    enabled: boolean;
  }>({
    id: '',
    label: '',
    type: 'llm',
    description: '',
    features: [] as string[],
    contextWindow: undefined,
    maxTokens: undefined,
    temperature: 0.7,
    topP: 1.0,
    enabled: true
  });

  // ==================== Effects ====================

  // 持久化到localStorage
  useEffect(() => {
    localStorage.setItem('user_provider_configs', JSON.stringify(providerConfigs));
  }, [providerConfigs]);

  useEffect(() => {
    localStorage.setItem('user_enabled_models', JSON.stringify(enabledModels));
  }, [enabledModels]);

  useEffect(() => {
    localStorage.setItem('user_custom_models', JSON.stringify(customModels));
  }, [customModels]);

  useEffect(() => {
    localStorage.setItem('user_custom_providers', JSON.stringify(customProviders));
  }, [customProviders]);

  // 获取供应商和模型（用户自己的配置）
  useEffect(() => {
    const fetchBuiltinProviders = async () => {
      try {
        // 修改：使用 get_providers_with_models() 获取用户自己的配置
        // 后端会自动处理：首次访问时从管理员模板复制配置到用户账户
        const res = await providerApi.get_providers_with_models();
        if (res.success === true && Array.isArray(res.data)) {
          setApiProviders(res.data);

          // Sync backend config existence to local state
          setProviderConfigs(prev => {
              const next = { ...prev };
              let changed = false;
              res.data.forEach((p: any) => {
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

          // 始终从 API 同步启用状态 (因为后端已经处理了用户偏好持久化)
          const enabledFromApi = res.data.flatMap((p: any) =>
            (p.models || []).filter((m: any) => m.is_enabled === true).map((m: any) => m.model_name)
          );
          setEnabledModels(enabledFromApi);

          // 设置默认选中的供应商
          if (res.data.length > 0 && !selectedProviderId) {
            setSelectedProviderId(res.data[0].name);
          } else if (res.data.length > 0 && !res.data.some((p: any) => p.name === selectedProviderId)) {
            setSelectedProviderId(res.data[0].name);
          }
        }
      } catch (error) {
        console.error('Failed to fetch builtin providers:', error);
        setApiProviders([]);
      }
    };
    fetchBuiltinProviders();
  }, [selectedProviderId]);

  // 切换供应商时重置验证状态
  useEffect(() => {
    setVerifyStatus('idle');
    setVerifyMessage('');
    setShowKey(false);
  }, [selectedProviderId]);

  // ==================== 计算属性（useMemo） ====================

  // 合并内置和自定义供应商
  const allProviders = useMemo(() => {
    if (apiProviders.length > 0) {
      return apiProviders.map((p: any) => ({
        id: p.name,
        name: p.label || p.name,
        icon: p.icon || '🤖',
        description: p.description,
        defaultBaseUrl: p.default_base_url || p.base_url || '',
        supportedTypes: p.supported_model_types || [],
        helpUrl: p.help_url,
        providerId: p.id,
        isCustom: p.is_builtin === false,
        isEnabled: p.is_enabled !== false
      }));
    }
    return [];
  }, [apiProviders]);

  // 当前选中的供应商
  const currentProvider = allProviders.find(p => p.id === selectedProviderId);
  const currentProviderMeta = apiProviders.find((p: any) => p.name === selectedProviderId);
  const isCustomProvider = (currentProvider as any)?.isCustom || customProviders.some(p => p.id === selectedProviderId);
  const currentConfig = providerConfigs[selectedProviderId] || { apiKey: '', baseUrl: '', enabled: false };

  // 当前供应商的原始模型列表
  const rawModelsForProvider = useMemo(() => {
    if (currentProviderMeta?.models?.length) {
      return currentProviderMeta.models.map((m: any) => ({
        id: m.model_name,
        name: m.label || m.model_name,
        description: m.description || '',
        provider: currentProviderMeta.name,
        modelType: m.model_type || 'llm',
        supportsThinking: m.features?.includes('thinking') || m.features?.includes('agent_thought'),
        supportsVision: m.features?.includes('image_input') || m.features?.includes('vision'),
        contextWindow: m.context_window ? `${Math.round(m.context_window / 1000)}K` : undefined,
        isCustom: customModels.some(cm => cm.id === m.model_name),
        modelConfigId: m.id,
        isEnabled: m.is_enabled !== false,
        rawFeatures: m.features || [],
        default_max_tokens: m.default_max_tokens,
        default_temperature: m.default_temperature,
        default_top_p: m.default_top_p,
      }));
    }
    return [
      ...customModels.filter(m => m.provider === selectedProviderId)
    ];
  }, [currentProviderMeta, selectedProviderId, customModels]);

  // 根据分类和搜索过滤模型列表
  const displayModels = useMemo(() => {
    let filtered = rawModelsForProvider;

    // 按模型类型过滤
    if (selectedModelType !== 'all') {
      filtered = filtered.filter((m: any) => m.modelType === selectedModelType);
    }

    // 按搜索关键词过滤
    if (searchModel) {
      filtered = filtered.filter((m: any) =>
        m.name.toLowerCase().includes(searchModel.toLowerCase()) ||
        m.id.toLowerCase().includes(searchModel.toLowerCase())
      );
    }

    return filtered;
  }, [rawModelsForProvider, selectedModelType, searchModel]);

  // 模型元数据映射
  const modelMetaMap = useMemo(() => {
    return new Map<string, any>(rawModelsForProvider.map((m: any) => [m.id, m]));
  }, [rawModelsForProvider]);

  // ==================== 辅助函数 ====================

  // 刷新供应商和模型列表
  const fetchProvidersWithModels = async (nextSelectedProviderId?: string) => {
    try {
      const res = await providerApi.get_providers_with_models();
      if (res.success === true && Array.isArray(res.data)) {
        setApiProviders(res.data);
        const target = nextSelectedProviderId || selectedProviderId;
        if (res.data.length > 0 && !res.data.some((p: any) => p.name === target)) {
          setSelectedProviderId(res.data[0].name);
        } else if (nextSelectedProviderId) {
          setSelectedProviderId(nextSelectedProviderId);
        }
      }
    } catch (error) {
      setApiProviders([]);
    }
  };

  // 映射模型类型
  const mapModelType = (value: string) => {
    const lower = value.toLowerCase();
    if (lower.includes('multimodal')) return 'multimodal';
    if (lower.includes('embedding')) return 'embedding';
    if (lower.includes('image')) return 'image';
    if (lower.includes('code')) return 'code';
    return 'llm';
  };

  // 映射模型特性
  const mapModelFeatures = (features: string[]) => {
    return features.map(f => {
      if (f === 'structured') return 'structured_output';
      if (f === 'tools') return 'tool_call';
      if (f === 'thinking') return 'agent_thought';
      if (f === 'vision') return 'image_input';
      return f;
    });
  };

  // 自动选择第一个可用模型用于验证
  useEffect(() => {
    if (rawModelsForProvider.length > 0) {
      setCheckModelId(prev => {
        const exists = rawModelsForProvider.find((m: any) => m.id === prev);
        return exists ? prev : rawModelsForProvider[0].id;
      });
    } else {
      setCheckModelId('');
    }
  }, [rawModelsForProvider]);

  // ==================== 事件处理函数 ====================

  // 切换供应商启用状态
  const toggleProviderEnabled = async (e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (!currentProviderMeta?.id) return;

    const newEnabled = !currentProviderMeta.is_enabled;

    // 乐观更新本地状态
    setApiProviders(prev => prev.map(p =>
      p.id === currentProviderMeta.id ? { ...p, is_enabled: newEnabled } : p
    ));

    try {
      const updateData: ModelProviderUpdate = {
        provider_id: currentProviderMeta.id,
        is_enabled: newEnabled
      };
      const res = await providerApi.update(updateData);
      if (res.success !== true) {
        throw new Error(res.message || '更新供应商状态失败');
      }
    } catch (error: any) {
      console.error('Error updating provider enabled status:', error);
      alert('更新供应商状态失败：' + (error.message || '网络错误'));
      // 回滚状态
      setApiProviders(prev => prev.map(p =>
        p.id === currentProviderMeta.id ? { ...p, is_enabled: !newEnabled } : p
      ));
    }
  };

  // 更新供应商配置（API Key或Base URL）
  const updateConfig = (field: keyof ProviderConfig, value: string) => {
    // 1. Update local state immediately for UI responsiveness
    setProviderConfigs(prev => ({
      ...prev,
      [selectedProviderId]: {
        ...prev[selectedProviderId],
        [field]: value,
        enabled: true
      }
    }));

    // 2. Debounce update to backend (wait 2 seconds)
    // Clear existing timeout for this specific field
    const timeouts = updateTimeoutsRef.current;
    if (timeouts.has(field)) {
        clearTimeout(timeouts.get(field)!);
    }

    // Capture current context
    const targetProviderId = selectedProviderId;
    const targetProviderMeta = currentProviderMeta;

    if (!targetProviderMeta?.id) return;

    // Set new timeout
    const timeoutId = setTimeout(async () => {
        // Prepare partial update data
        const updateData: ModelProviderUpdate = {
            provider_id: targetProviderMeta.id,
            is_enabled: true
        };
        
        if (field === 'apiKey') updateData.api_key = value;
        if (field === 'baseUrl') updateData.base_url = value;

        try {
            const res = await providerApi.update(updateData);
            if (res.success !== true) {
                console.error('Failed to sync config:', res.message);
            } else {
                // Update apiProviders with new data if backend returns it
                setApiProviders(prev => prev.map(p => 
                    p.id === targetProviderMeta.id ? { ...p, ...updateData } : p
                ));
            }
        } catch (error: any) {
            console.error('Error updating provider config:', error);
        } finally {
            // Cleanup timeout from map
            timeouts.delete(field);
        }
    }, 2000);

    timeouts.set(field, timeoutId);
  };

  // 验证API Key连通性
  const handleVerifyApiKey = async () => {
    if (!currentConfig.apiKey || !checkModelId) return;

    setIsVerifying(true);
    setVerifyStatus('idle');
    setVerifyMessage('');

    try {
      const verifyData: ProviderApiKeyVerify = {
        provider_id: currentProviderMeta?.id,
        api_key: currentConfig.apiKey,
        base_url: currentConfig.baseUrl || currentProvider?.defaultBaseUrl,
        model_name: checkModelId
      };

      const res = await providerApi.verifyApiKey(verifyData);

      if (res.success === true && res.data?.valid) {
        setVerifyStatus('success');
        setVerifyMessage(res.data.message || 'API Key验证成功');
      } else {
        setVerifyStatus('error');
        setVerifyMessage(res.data?.message || res.message || 'API Key验证失败');
      }
    } catch (error: any) {
      setVerifyStatus('error');
      setVerifyMessage(error.message || '验证过程中发生错误');
    } finally {
      setIsVerifying(false);
    }
  };

  // 切换模型启用状态
  const toggleModelEnabled = async (modelId: string) => {
    const newEnabled = !enabledModels.includes(modelId);
    setEnabledModels(prev => {
      if (prev.includes(modelId)) return prev.filter(id => id !== modelId);
      return [...prev, modelId];
    });
    const modelMeta = modelMetaMap.get(modelId);
    if (modelMeta?.modelConfigId) {
      const updateData: ModelConfigUpdate = {
        model_config_id: modelMeta.modelConfigId,
        is_enabled: newEnabled
      };
      try {
        const res = await modelApi.update(updateData);
        if (res.success !== true) {
          throw new Error(res.message || '更新模型配置失败');
        }
      } catch (error: any) {
        console.error('Error updating model config:', error);
        alert('更新模型配置失败：' + (error.message || '网络错误'));
        // 恢复之前的启用状态
        setEnabledModels(prev => {
          if (newEnabled) return prev.filter(id => id !== modelId);
          return [...prev, modelId];
        });
      }
    }
  };

  // const handleCheckConnectivity = async () => {
  //   if (!currentConfig.apiKey) return;
  //   setIsChecking(true);
  //   setCheckStatus('idle');
  //   // Simulate check
  //   setTimeout(() => {
  //     setIsChecking(false);
  //     setCheckStatus('success'); // In a real app, actually fetch models
  //   }, 1500);
  // };

  const handleCreateCustomModel = async () => {
    if (!modelForm.id) {
      alert('请填写模型名称');
      return;
    }
    
    if (!currentProviderMeta?.id) {
      alert('无法获取当前供应商信息，请刷新页面重试');
      return;
    }
    
    // Validate model name format
    const modelNameRegex = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;
    if (!modelNameRegex.test(modelForm.id)) {
      alert('模型名称只能包含字母、数字、点号、下划线和横线，且必须以字母或数字开头');
      return;
    }
    
    const newModel: CustomModel = {
      id: modelForm.id,
      name: modelForm.id,
      description: modelForm.description || 'Custom Model Configuration',
      provider: selectedProviderId as any,
      supportsThinking: modelForm.features.includes('thinking'),
      supportsVision: modelForm.features.includes('vision') || modelForm.type.includes('Multimodal'),
      supportsToolCall: modelForm.features.includes('tool_call'),
      contextWindow: modelForm.contextWindow.toString() + (modelForm.contextWindow > 1000 ? 'K' : ''),
      isCustom: true
    };
    
    try {
      const createData: ModelConfigCreate = {
        provider_id: currentProviderMeta.id,
        model_name: modelForm.id,
        label: modelForm.label || modelForm.id, // Add label
        description: modelForm.description,
        model_type: modelForm.type,
        features: mapModelFeatures(modelForm.features),
        context_window: modelForm.contextWindow,
        default_max_tokens: modelForm.maxTokens,
        default_parameters: { // Use default_parameters dict
            temperature: modelForm.temperature,
            top_p: modelForm.topP
        }
      };
      const res = await modelApi.create(createData);
      if (res.success === true) {
        setCustomModels(prev => prev.some(m => m.id === newModel.id) ? prev : [...prev, newModel]);
        if (modelForm.enabled) {
          setEnabledModels(prev => prev.includes(newModel.id) ? prev : [...prev, newModel.id]);
        }
        setShowAddModelModal(false);
        setModelForm({
          id: '',
          label: '',
          type: 'llm',
          description: '',
          features: [],
          contextWindow: undefined,
          maxTokens: undefined,
          temperature: 0.7,
          topP: 1.0,
          enabled: true
        });
        fetchProvidersWithModels(selectedProviderId);
      } else {
        throw new Error(res.message || '创建模型失败');
      }
    } catch (error: any) {
      console.error('Error creating custom model:', error);
      alert('创建模型时发生错误：' + (error.message || '网络错误'));
      setCustomModels(prev => prev);
    }
  };

  // Delete Request Handlers
  const requestDeleteCustomModel = (id: string) => {
    setDeleteTarget({ type: 'model', id });
    setShowDeleteConfirm(true);
  };

  const requestDeleteCustomProvider = (id: string) => {
    setDeleteTarget({ type: 'provider', id });
    setShowDeleteConfirm(true);
  };

  // Confirm Delete Logic
  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;

    if (deleteTarget.type === 'model') {
      const modelMeta = modelMetaMap.get(deleteTarget.id);
      if (modelMeta?.modelConfigId) {
        const deleteData: ModelConfigDelete = { model_config_id: modelMeta.modelConfigId };
        try {
          const res = await modelApi.delete(deleteData);
          if (res.success !== true) {
            throw new Error(res.message || '删除模型失败');
          }
          fetchProvidersWithModels(selectedProviderId);
        } catch (error: any) {
          console.error('Error deleting model:', error);
          alert('删除模型失败：' + (error.message || '网络错误'));
        }
      }
      setCustomModels(prev => prev.filter(m => m.id !== deleteTarget.id));
      setEnabledModels(prev => prev.filter(mid => mid !== deleteTarget.id));
    } else if (deleteTarget.type === 'provider') {
      const providerMeta = apiProviders.find((p: any) => p.name === deleteTarget.id);
      if (providerMeta?.id) {
        const deleteData: ModelProviderDelete = { provider_id: providerMeta.id };
        try {
          const res = await providerApi.delete(deleteData);
          if (res.success !== true) {
            throw new Error(res.message || '删除供应商失败');
          }
          fetchProvidersWithModels(apiProviders.length > 0 ? apiProviders[0].name : '');
        } catch (error: any) {
          console.error('Error deleting provider:', error);
          alert('删除供应商失败：' + (error.message || '网络错误'));
        }
      }
      setCustomProviders(prev => prev.filter(p => p.id !== deleteTarget.id));
      setProviderConfigs(prev => {
        const next = { ...prev };
        delete next[deleteTarget.id];
        return next;
      });
      const modelsToDelete = customModels.filter(m => m.provider === deleteTarget.id).map(m => m.id);
      setCustomModels(prev => prev.filter(m => m.provider !== deleteTarget.id));
      setEnabledModels(prev => prev.filter(mid => !modelsToDelete.includes(mid)));
      if (selectedProviderId === deleteTarget.id) {
        setSelectedProviderId(apiProviders.length > 0 ? apiProviders[0].name : '');
      }
    }

    setShowDeleteConfirm(false);
    setDeleteTarget(null);
  };

  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);

  // ...

  // Open Edit Provider Modal
  const handleEditProvider = () => {
      if (!currentProviderMeta) return;
      setEditingProviderId(currentProviderMeta.id);
      setNewProviderForm({
          nameId: currentProviderMeta.name,
          label: currentProviderMeta.label || currentProviderMeta.name,
          description: currentProviderMeta.description || '',
          defaultBaseUrl: currentProviderMeta.default_base_url || '',
          supportedTypes: currentProviderMeta.supported_model_types || [],
          isEnabled: currentProviderMeta.is_enabled !== false
      });
      setShowAddProviderModal(true);
  };

  // Add or Update Provider Logic
  const handleSaveProvider = async () => {
    if (!newProviderForm.nameId) {
      alert('请填写供应商名称');
      return;
    }
    
    // Validate provider ID format (only for new providers)
    if (!editingProviderId) {
        const providerIdRegex = /^[a-zA-Z][a-zA-Z0-9_-]*$/;
        if (!providerIdRegex.test(newProviderForm.nameId)) {
        alert('供应商名称只能包含字母、数字、下划线和横线，且必须以字母开头');
        return;
        }
        
        // Check ID conflict
        if (allProviders.some(p => p.id === newProviderForm.nameId)) {
            alert("供应商ID已存在！");
            return;
        }
    }

    try {
      if (editingProviderId) {
          // Update existing provider
          const updateData: ModelProviderUpdate = {
              provider_id: editingProviderId,
              label: newProviderForm.label,
              description: newProviderForm.description,
              base_url: newProviderForm.defaultBaseUrl,
              supported_model_types: newProviderForm.supportedTypes,
              is_enabled: newProviderForm.isEnabled
          };
          const res = await providerApi.update(updateData);
          if (res.success === true) {
             // Update local state if needed (though fetchProvidersWithModels usually handles it)
             setShowAddProviderModal(false);
             setEditingProviderId(null);
             setNewProviderForm({ nameId: '', label: '', description: '', defaultBaseUrl: '', supportedTypes: [], isEnabled: true });
             fetchProvidersWithModels(newProviderForm.nameId);
          } else {
              throw new Error(res.message || '更新供应商失败');
          }
      } else {
          // Create new provider
          const createData: ModelProviderCreate = {
            name: newProviderForm.nameId,
            label: newProviderForm.label,
            description: newProviderForm.description,
            base_url: newProviderForm.defaultBaseUrl,
            supported_model_types: newProviderForm.supportedTypes,
            position: allProviders.length + 1
          };
          const res = await providerApi.create(createData);
          if (res.success === true) {
            if (!newProviderForm.isEnabled && res.data?.id) {
              const updateData: ModelProviderUpdate = { provider_id: res.data.id, is_enabled: false };
              await providerApi.update(updateData);
            }
            
            // ... (rest of create logic)
             if (newProviderForm.isEnabled) {
                const newProviderId = res.data?.name || newProviderForm.nameId;
                setProviderConfigs(prev => ({
                    ...prev,
                    [newProviderId]: {
                    apiKey: '',
                    baseUrl: newProviderForm.defaultBaseUrl,
                    enabled: true
                    }
                }));
            }

            setShowAddProviderModal(false);
            const newId = res.data?.name || newProviderForm.nameId;
            setSelectedProviderId(newId);
            setNewProviderForm({ nameId: '', label: '', description: '', defaultBaseUrl: '', supportedTypes: [], isEnabled: true });
            fetchProvidersWithModels(newId);
          } else {
            throw new Error(res.message || '创建供应商失败');
          }
      }
    } catch (error: any) {
      console.error('Error saving provider:', error);
      alert((editingProviderId ? '更新' : '添加') + '供应商时发生错误：' + (error.message || '网络错误')); 
    }
  };

  const toggleSupportedType = (typeId: string) => {
      setNewProviderForm(prev => {
          const types = prev.supportedTypes.includes(typeId) 
            ? prev.supportedTypes.filter(t => t !== typeId)
            : [...prev.supportedTypes, typeId];
          return { ...prev, supportedTypes: types };
      });
  };

  const toggleModelFeature = (featureId: string) => {
      setModelForm(prev => {
          const features = prev.features.includes(featureId)
            ? prev.features.filter(f => f !== featureId)
            : [...prev.features, featureId];
          return { ...prev, features };
      });
  };

  // Edit existing model
  const handleEditModel = (model: any) => {
    setEditingModelId(model.id);
    setModelForm({
      id: model.id,
      label: model.name || model.id, // Populate label
      type: model.modelType || 'llm',
      description: model.description || '',
      features: model.rawFeatures || model.features || [],
      contextWindow: model.contextWindow ? parseInt(model.contextWindow.replace('K', '000')) : undefined,
      maxTokens: model.default_max_tokens,
      temperature: model.default_temperature ?? 0.7,
      topP: model.default_top_p ?? 1.0,
      enabled: model.isEnabled !== false
    });
    setShowAddModelModal(true);
  }; 

  // Handle update existing model
  const handleUpdateCustomModel = async () => {
    if (!modelForm.id || !currentProviderMeta?.id || !editingModelId) return;
    
    // Validate model name format
    const modelNameRegex = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;
    if (!modelNameRegex.test(modelForm.id)) {
      alert('模型名称只能包含字母、数字、点号、下划线和横线，且必须以字母或数字开头');
      return;
    }
    
    try {
      const updateData: ModelConfigUpdate = {
        model_config_id: editingModelId, // This should be the actual model config ID
        label: modelForm.label || modelForm.id, // Update the display name
        description: modelForm.description,
        model_type: modelForm.type,
        features: mapModelFeatures(modelForm.features),
        context_window: modelForm.contextWindow,
        default_max_tokens: modelForm.maxTokens,
        default_parameters: {
            temperature: modelForm.temperature,
            top_p: modelForm.topP
        },
        is_enabled: modelForm.enabled
      };
      
      // Find the actual model config ID from the meta map
      const modelMeta = rawModelsForProvider.find((m: any) => m.id === editingModelId);
      if (modelMeta?.modelConfigId) {
        updateData.model_config_id = modelMeta.modelConfigId;
      }
      
      const res = await modelApi.update(updateData);
      if (res.success === true) {
        // Update the custom models state
        setCustomModels(prev => 
          prev.map(m => 
            m.id === editingModelId 
              ? { 
                  ...m, 
                  id: modelForm.id,
                  name: modelForm.label || modelForm.id,
                  description: modelForm.description,
                  supportsThinking: modelForm.features.includes('thinking'),
                  supportsVision: modelForm.features.includes('vision') || modelForm.type.includes('Multimodal'),
                  supportsToolCall: modelForm.features.includes('tool_call'),
                  contextWindow: modelForm.contextWindow.toString() + (modelForm.contextWindow > 1000 ? 'K' : ''),
                } 
              : m
          )
        );
        
        // Update enabled models if needed
        if (modelForm.enabled && !enabledModels.includes(editingModelId)) {
          setEnabledModels(prev => [...prev, editingModelId]);
        } else if (!modelForm.enabled && enabledModels.includes(editingModelId)) {
          setEnabledModels(prev => prev.filter(id => id !== editingModelId));
        }
        
        setShowAddModelModal(false);
        setEditingModelId(null);
        setModelForm({
          id: '',
          label: '',
          type: 'llm',
          description: '',
          features: [],
          contextWindow: undefined,
          maxTokens: undefined,
          temperature: 0.7,
          topP: 1.0,
          enabled: true
        });
        fetchProvidersWithModels(selectedProviderId);
      } else {
        throw new Error(res.message || '更新模型失败');
      }
    } catch (error: any) {
      console.error('Error updating custom model:', error);
      alert('更新模型时发生错误：' + (error.message || '网络错误'));
    }
  };

  return (
    <div className="flex h-screen bg-rice-paper dark:bg-black text-tcm-charcoal dark:text-gray-200 transition-colors duration-500 overflow-hidden font-sans">
      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && deleteTarget && (
        <DeleteConfirmToast
          message={deleteTarget.type === 'provider' 
            ? '确定要删除该提供商及其全部模型吗？' 
            : '确定要删除该模型吗？'}
          onConfirm={handleConfirmDelete}
          onCancel={() => { setShowDeleteConfirm(false); setDeleteTarget(null); }}
          onClose={() => { setShowDeleteConfirm(false); setDeleteTarget(null); }}
          variant="tcm"
        />
      )}

      {/* 1. Sidebar - Provider List */}
      <aside className="w-72 bg-[#f9fafb] dark:bg-black border-r border-gray-200 dark:border-white/5 flex flex-col flex-shrink-0 z-20">
        <div className="p-4 pt-6">
          <div className="flex items-center gap-2 mb-6 cursor-pointer" onClick={() => navigate('/public')}>
            <button className="p-1.5 hover:bg-gray-200 dark:hover:bg-white/10 rounded-lg transition-colors text-gray-500">
              <Icons.ChevronRight className="rotate-180" size={20} />
            </button>
            <h1 className="text-lg font-bold font-serif-sc text-tcm-darkGreen dark:text-tcm-cream">设置</h1>
          </div>
          
          <div className="flex gap-2 mb-4">
            <div className="relative flex-1">
                <Icons.Activity className="absolute left-3 top-2.5 text-gray-400" size={16} />
                <input 
                type="text" 
                placeholder="搜索服务商..." 
                value={searchProvider}
                onChange={(e) => setSearchProvider(e.target.value)}
                className="w-full pl-9 pr-3 py-2 bg-white dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:ring-2 focus:ring-tcm-lightGreen/50 outline-none transition-all"
                />
            </div>
            <button 
                onClick={() => setShowAddProviderModal(true)}
                className="p-2 bg-white dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-500 hover:text-tcm-darkGreen hover:border-tcm-darkGreen transition-all group relative"
                title="添加自定义服务"
            >
                <Icons.Plus size={20} />
                <div className="absolute left-1/2 -translate-x-1/2 -top-8 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">添加自定义服务</div>
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 space-y-1 custom-scrollbar pb-4">
          <div className="px-3 py-2 text-xs font-bold text-gray-400 uppercase tracking-wider">全部</div>
          {allProviders.filter(p => p.name.toLowerCase().includes(searchProvider.toLowerCase())).map(provider => {
            const config = providerConfigs[provider.id];
            const isEnabled = config?.enabled && !!config?.apiKey;
            const isActive = selectedProviderId === provider.id;
            const providerIcon = getProviderIconPath(provider.name);
            const shouldInvert = isDarkInvert(provider.name);

            return (
              <button
                key={provider.id}
                onClick={() => setSelectedProviderId(provider.id)}
                className={`w-full flex items-center justify-between p-3 rounded-xl transition-all group ${
                  isActive 
                    ? 'bg-white dark:bg-[#151515] shadow-sm border border-gray-100 dark:border-white/10' 
                    : 'hover:bg-gray-100 dark:hover:bg-white/5 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-xl bg-gray-100 dark:bg-white/10 w-8 h-8 flex items-center justify-center rounded-lg">
                    {providerIcon ? (
                        <img 
                            src={providerIcon} 
                            alt={provider.name} 
                            className={`w-5 h-5 ${shouldInvert ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} 
                        />
                    ) : provider.icon}
                  </span>
                  <span className={`text-sm font-medium ${isActive ? 'text-tcm-darkGreen dark:text-tcm-lightGreen font-bold' : 'text-gray-700 dark:text-gray-300'}`}>
                    {provider.name}
                  </span>
                </div>
                {isEnabled && <div className="w-1.5 h-1.5 rounded-full bg-tcm-lightGreen shadow-[0_0_8px_rgba(77,140,124,0.6)]"></div>}
              </button>
            );
          })}
        </div>
      </aside>

      {/* 2. Main Content - Settings & Model List */}
      <main className="flex-1 overflow-y-auto bg-rice-paper dark:bg-black p-4 md:p-8 scroll-smooth">
        <div className="max-w-4xl mx-auto space-y-6">
          
          {/* Header */}
          <div className="flex items-center gap-3 mb-6">
             <div className="text-3xl">
                {currentProvider?.name && getProviderIconPath(currentProvider.name) ? (
                    <img 
                        src={getProviderIconPath(currentProvider.name)} 
                        alt={currentProvider.name} 
                        className={`w-8 h-8 ${isDarkInvert(currentProvider.name) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} 
                    />
                ) : (
                    currentProvider?.icon
                )}
             </div>
             <h2 className="text-2xl font-bold font-serif-sc text-tcm-darkGreen dark:text-tcm-cream">{currentProvider?.name}</h2>
             {/* Show badge if custom */}
             {isCustomProvider && (
                 <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-tcm-gold/10 text-tcm-gold border border-tcm-gold/20">CUSTOM</span>
             )}
             
             {/* Provider Master Controls (Right Aligned) */}
             <div className="ml-auto flex items-center gap-3">
                {/* Provider Master Toggle */}
                <button
                    onClick={toggleProviderEnabled}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    currentProviderMeta?.is_enabled ? 'bg-tcm-lightGreen' : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${currentProviderMeta?.is_enabled ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>

                {/* Edit Provider Button (Only for Custom Providers) */}
                {isCustomProvider && (
                    <button 
                        onClick={handleEditProvider}
                        className="p-2 text-gray-400 hover:text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-all"
                        title="编辑供应商"
                    >
                        <Icons.Settings size={18} />
                    </button>
                )}

                {/* Delete Provider Button (Only for Custom Providers) */}
                {isCustomProvider && (
                    <button 
                        onClick={() => requestDeleteCustomProvider(selectedProviderId)}
                        className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-all"
                        title="删除该提供商"
                    >
                        <Icons.Trash2 size={18} />
                    </button>
                )}
             </div>
          </div>

          {/* Card 1: Provider Configuration */}
          <div className="bg-white dark:bg-[#0a0a0a] rounded-2xl border border-gray-200 dark:border-white/5 p-6 shadow-sm">
             <div className="grid gap-6">
                
                {/* API Key */}
                <div className="space-y-2">
                   <div className="flex justify-between">
                      <label className="text-sm font-bold text-gray-700 dark:text-gray-200">API Key</label>
                      {currentProvider?.helpUrl ? (
                        <a
                          href={currentProvider.helpUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-tcm-lightGreen hover:underline flex items-center gap-1"
                        >
                          获取 API Key
                          <Icons.ExternalLink size={12} />
                        </a>
                      ) : (
                        <span className="text-xs text-gray-400">获取 API Key</span>
                      )}
                   </div>
                   <div className="relative group">
                      <input
                        type={showKey ? "text" : "password"}
                        value={currentConfig.apiKey || ''}
                        onChange={(e) => updateConfig('apiKey', e.target.value)}
                        placeholder={`输入 ${currentProvider?.name} API Key`}
                        className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-xl outline-none focus:ring-2 focus:ring-tcm-lightGreen/50 focus:border-tcm-lightGreen transition-all text-sm font-mono"
                      />
                      <button
                        onClick={() => setShowKey(!showKey)}
                        className="absolute right-3 top-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                      >
                        {showKey ? <Icons.Image size={16} /> : <Icons.Zap size={16} />}
                      </button>
                   </div>
                </div>

                {/* Base URL */}
                <div className="space-y-2">
                   <label className="text-sm font-bold text-gray-700 dark:text-gray-200">API 代理地址</label>
                   <input
                      type="text"
                      value={currentConfig.baseUrl || ''}
                      onChange={(e) => updateConfig('baseUrl', e.target.value)}
                      placeholder={ (currentProvider as any)?.defaultBaseUrl || "https://api.openai.com/v1" }
                      className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-xl outline-none focus:ring-2 focus:ring-tcm-lightGreen/50 focus:border-tcm-lightGreen transition-all text-sm font-mono"
                   />
                   <p className="text-xs text-gray-400">默认使用官方地址，如需使用代理请输入完整地址。</p>
                </div>

                {/* Check Connectivity */}
                <div className="pt-2 flex items-center justify-between bg-gray-50 dark:bg-black/10 p-3 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">
                   <div className="flex flex-col">
                      <span className="text-sm font-bold text-gray-700 dark:text-gray-200">连通性检查</span>
                      <span className="text-xs text-gray-400">测试 API Key 与代理地址是否正确填写</span>
                      {verifyMessage && (
                        <span className={`text-xs mt-1 ${verifyStatus === 'success' ? 'text-green-500' : 'text-red-500'}`}>
                          {verifyMessage}
                        </span>
                      )}
                   </div>

                   <div className="flex items-center gap-2">
                       <div className="relative">
                            <select
                                value={checkModelId}
                                onChange={(e) => setCheckModelId(e.target.value)}
                                className="bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-gray-700 rounded-lg text-xs py-2 pl-3 pr-8 appearance-none outline-none focus:ring-1 focus:ring-tcm-lightGreen cursor-pointer min-w-[160px] font-mono text-gray-700 dark:text-gray-300"
                            >
                                {rawModelsForProvider.length === 0 ? <option>无可用模型</option> : rawModelsForProvider.map((m: any) => (
                                    <option key={m.id} value={m.id}>{m.id}</option>
                                ))}
                            </select>
                            <Icons.ChevronDown className="absolute right-2.5 top-2.5 text-gray-400 pointer-events-none" size={14} />
                       </div>

                       <button
                         onClick={handleVerifyApiKey}
                         disabled={isVerifying || !currentConfig.apiKey}
                         className="px-4 py-2 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-xs font-bold hover:bg-gray-50 dark:hover:bg-gray-600 transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                       >
                         {isVerifying ? '验证中...' : verifyStatus === 'success' ? <span className="text-green-500 flex items-center gap-1"><Icons.Check size={14}/> 通行</span> : '检查'}
                       </button>
                   </div>
                </div>

             </div>
          </div>

          {/* Card 2: Model List */}
          <div className="bg-white dark:bg-[#0a0a0a] rounded-2xl border border-gray-200 dark:border-white/5 shadow-sm flex flex-col min-h-[400px]">

             {/* Toolbar */}
             <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex flex-wrap gap-4 items-center justify-between bg-gray-50/50 dark:bg-black/10 rounded-t-2xl">
                <div className="flex items-center gap-2">
                   <span className="font-bold text-gray-700 dark:text-gray-200 text-sm">模型列表</span>
                   <span className="bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400 px-2 py-0.5 rounded-full text-xs font-mono">{displayModels.length}</span>
                </div>

                <div className="flex items-center gap-2 flex-1 justify-end">
                   <div className="relative max-w-[200px] w-full">
                      <Icons.Activity className="absolute left-2.5 top-2 text-gray-400" size={14} />
                      <input
                        type="text"
                        placeholder="搜索模型..."
                        value={searchModel}
                        onChange={e => setSearchModel(e.target.value)}
                        className="w-full pl-8 pr-3 py-1.5 bg-white dark:bg-black/20 border border-gray-200 dark:border-gray-600 rounded-lg text-xs outline-none focus:ring-1 focus:ring-tcm-lightGreen"
                      />
                   </div>
                   <button
                     onClick={() => {
                       setEditingModelId(null);
                       setModelForm({
                        id: '',
                        type: 'llm',
                        label: '',
                        description: '',
                        features: [],
                        contextWindow: 4096,
                         maxTokens: 4096,
                         temperature: 0.7,
                         topP: 1.0,
                         enabled: true
                       });
                       setShowAddModelModal(true);
                     }}
                     className="p-1.5 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg hover:border-tcm-lightGreen hover:text-tcm-lightGreen transition-colors text-gray-500"
                     title="Add Custom Model"
                   >
                     <Icons.Plus size={16} />
                   </button>
                </div>
             </div>

             {/* Model Type Category Tabs */}
             <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 overflow-x-auto">
                <div className="flex gap-2">
                   {MODEL_TYPE_CATEGORIES.map(category => (
                     <button
                       key={category.key}
                       onClick={() => setSelectedModelType(category.key)}
                       className={`px-3 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap transition-all ${
                         selectedModelType === category.key
                           ? 'bg-tcm-darkGreen text-white shadow-md'
                           : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                       }`}
                     >
                       {category.label}
                     </button>
                   ))}
                </div>
             </div>

             {/* List */}
             <div className="divide-y divide-gray-100 dark:divide-gray-800 flex-1 overflow-y-auto">
                {displayModels.length === 0 ? (
                  <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
                    {selectedModelType === 'all' ? '暂无模型' : `暂无 ${MODEL_TYPE_CATEGORIES.find(c => c.key === selectedModelType)?.label} 类型的模型`}
                  </div>
                ) : displayModels.map((model: any) => (
                   <div key={model.id} className="p-4 flex items-center gap-4 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group">
                      <div className="w-10 h-10 rounded-xl bg-gray-100 dark:bg-white/10 flex items-center justify-center text-xl flex-shrink-0 text-gray-600 dark:text-gray-300">
                         {(() => {
                             const modelIcon = getModelIconPath(model.name);
                             if (modelIcon) return <img src={modelIcon} alt={model.name} className={`w-6 h-6 ${isDarkInvert(model.name) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} />;
                             const providerIcon = getProviderIconPath(selectedProviderId);
                             if (providerIcon) return <img src={providerIcon} alt={selectedProviderId} className={`w-6 h-6 ${isDarkInvert(selectedProviderId) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`} />;
                             
                             return selectedProviderId === 'google' ? <Icons.Zap size={20}/> :
                                    selectedProviderId === 'openai' ? <Icons.BrainCircuit size={20}/> :
                                    selectedProviderId === 'anthropic' ? <Icons.Leaf size={20}/> : <Icons.Bot size={20}/>;
                         })()}
                      </div>

                      <div className="flex-1 min-w-0">
                         <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                            <span className="font-bold text-sm text-gray-800 dark:text-gray-200 truncate">{model.name}</span>
                            <span className="bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 px-1.5 py-0.5 rounded text-[10px] font-mono">{model.id}</span>
                            {(model as CustomModel).isCustom && <span className="text-[9px] text-tcm-gold border border-tcm-gold/30 px-1 rounded">Custom</span>}
                         </div>
                         <div className="flex items-center gap-2 text-xs text-gray-400 flex-wrap">
                            <span className="truncate max-w-[200px]">{model.description}</span>
                            {/* Features Tags */}
                            {model.rawFeatures && model.rawFeatures.length > 0 && (
                              <div className="flex gap-1 flex-wrap">
                                {model.rawFeatures.slice(0, 4).map((feature: string) => (
                                  <span
                                    key={feature}
                                    className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${FEATURE_COLORS[feature] || 'bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400'}`}
                                  >
                                    {feature.replace(/_/g, ' ')}
                                  </span>
                                ))}
                                {model.rawFeatures.length > 4 && (
                                  <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400">
                                    +{model.rawFeatures.length - 4}
                                  </span>
                                )}
                              </div>
                            )}
                         </div>
                      </div>

                      {/* Context Window & Features Icons */}
                      <div className="hidden md:flex gap-2 items-center">
                         {model.supportsVision && <span className="p-1 rounded bg-blue-50 dark:bg-blue-900/20 text-blue-500" title="Vision"><Icons.Image size={14}/></span>}
                         {model.supportsThinking && <span className="p-1 rounded bg-indigo-50 dark:bg-indigo-900/20 text-indigo-500" title="Thinking"><Icons.BrainCircuit size={14}/></span>}
                         {model.contextWindow && <span className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500 text-[10px] font-mono">{model.contextWindow}</span>}
                      </div>

                      {/* Toggle & Delete */}
                      <div className="flex items-center gap-3">
                         <button
                            onClick={() => toggleModelEnabled(model.id)}
                            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                              enabledModels.includes(model.id) ? 'bg-tcm-lightGreen' : 'bg-gray-200 dark:bg-gray-700'
                            }`}
                         >
                            <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${enabledModels.includes(model.id) ? 'translate-x-5' : 'translate-x-0.5'}`} />
                         </button>

                         {/* Delete Model Button (Right side of toggle, only for custom models) */}
                         {(model as CustomModel).isCustom && (
                            <>
                              <button onClick={() => handleEditModel(model)} className="p-1.5 text-gray-300 hover:text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-all" title="编辑模型">
                                 <Icons.Settings size={16} />
                              </button>
                              <button onClick={() => requestDeleteCustomModel(model.id)} className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-all" title="删除模型">
                                 <Icons.Trash2 size={16} />
                              </button>
                            </>
                         )}
                      </div>
                   </div>
                ))}
             </div>
          </div>

        </div>
      </main>

      {/* ADD PROVIDER MODAL */}
      {showAddProviderModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white dark:bg-[#1e1e1e] rounded-2xl border border-gray-200 dark:border-gray-700 shadow-2xl w-full max-w-lg overflow-hidden animate-in zoom-in-95">
                <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
                    <h3 className="text-base font-bold text-tcm-darkGreen dark:text-tcm-cream uppercase tracking-widest font-serif-sc">
                        {editingProviderId ? '编辑供应商' : '添加供应商'}
                    </h3>
                    <button onClick={() => setShowAddProviderModal(false)} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors">
                        <Icons.X size={20} />
                    </button>
                </div>
                <div className="p-6 space-y-5">
                    
                    {/* ID */}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">名称标识 (英文)</label>
                        <input 
                            type="text" 
                            placeholder="如: openai, anthropic"
                            value={newProviderForm.nameId}
                            onChange={(e) => setNewProviderForm({...newProviderForm, nameId: e.target.value.toLowerCase().replace(/\s+/g, '-')})}
                            className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen font-mono disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={!!editingProviderId}
                        />
                    </div>

                    {/* Label */}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">供应商展示名称</label>
                        <input 
                            type="text" 
                            placeholder="如: OpenAI, Anthropic AI"
                            value={newProviderForm.label || ''}
                            onChange={(e) => setNewProviderForm({...newProviderForm, label: e.target.value})}
                            className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen"
                        />
                    </div>

                    {/* Description */}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">描述</label>
                        <input 
                            type="text" 
                            placeholder="供应商描述..."
                            value={newProviderForm.description}
                            onChange={(e) => setNewProviderForm({...newProviderForm, description: e.target.value})}
                            className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen"
                        />
                    </div>

                    {/* Default Base URL */}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">默认BASE URL</label>
                        <input 
                            type="text" 
                            placeholder="https://api.example.com/v1"
                            value={newProviderForm.defaultBaseUrl}
                            onChange={(e) => setNewProviderForm({...newProviderForm, defaultBaseUrl: e.target.value})}
                            className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen font-mono"
                        />
                    </div>

                    {/* Model Types */}
                    <div className="space-y-2">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">支持的模型类型</label>
                        <div className="flex flex-wrap gap-2">
                            {SUPPORTED_MODEL_TYPES.map(type => (
                                <button
                                    key={type.id}
                                    onClick={() => toggleSupportedType(type.id)}
                                    className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                                        newProviderForm.supportedTypes.includes(type.id)
                                            ? 'bg-tcm-darkGreen text-white shadow-lg'
                                            : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                                    }`}
                                >
                                    {type.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Enabled Toggle */}
                    <div className="pt-2 flex items-center justify-between">
                        <span className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">启用状态</span>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => setNewProviderForm({...newProviderForm, isEnabled: !newProviderForm.isEnabled})}
                                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                                    newProviderForm.isEnabled ? 'bg-tcm-lightGreen' : 'bg-gray-200 dark:bg-gray-700'
                                }`}
                            >
                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${newProviderForm.isEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
                            </button>
                            <span className="text-xs text-gray-600 dark:text-gray-300">{newProviderForm.isEnabled ? '已启用' : '已禁用'}</span>
                        </div>
                    </div>

                </div>
                <div className="p-6 bg-gray-50/50 dark:bg-black/10 border-t border-gray-100 dark:border-gray-700 flex gap-3">
                    <button 
                        onClick={() => setShowAddProviderModal(false)}
                        className="flex-1 py-3 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-600 dark:text-gray-200 text-xs font-bold rounded-lg transition-colors"
                    >
                        取消
                    </button>
                    <button 
                        onClick={handleSaveProvider}
                        className="flex-1 py-3 bg-tcm-darkGreen hover:bg-tcm-lightGreen text-white text-xs font-bold rounded-lg transition-colors shadow-lg"
                    >
                        {editingProviderId ? '更新' : '创建'}
                    </button>
                </div>
            </div>
        </div>
      )}

      {/* ADD MODEL MODAL (DETAILED) */}
      {showAddModelModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white dark:bg-[#1e1e1e] rounded-2xl border border-gray-200 dark:border-gray-700 shadow-2xl w-full max-w-2xl overflow-hidden animate-in zoom-in-95 flex flex-col max-h-[90vh]">
                <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center shrink-0">
                    <h3 className="text-base font-bold text-tcm-darkGreen dark:text-tcm-cream uppercase tracking-widest font-serif-sc">添加模型配置</h3>
                    <button onClick={() => setShowAddModelModal(false)} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors">
                        <Icons.X size={20} />
                    </button>
                </div>
                <div className="p-6 space-y-5 overflow-y-auto custom-scrollbar">
                    
                    <div className="grid grid-cols-2 gap-4">
                        {/* Model Name */}
                        <div className="space-y-1.5">
                            <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">模型名称 (API调用用)</label>
                            <input 
                                type="text" 
                                placeholder="如: gpt-4o, claude-3-opus"
                                value={modelForm.id}
                                onChange={(e) => setModelForm({...modelForm, id: e.target.value})}
                                className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen font-mono"
                            />
                        </div>

                        {/* Label */}
                        <div className="space-y-1.5">
                            <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">模型展示名称</label>
                            <input 
                                type="text" 
                                placeholder="如: GPT-4o, Claude 3 Opus"
                                value={modelForm.label || ''}
                                onChange={(e) => setModelForm({...modelForm, label: e.target.value})}
                                className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen"
                            />
                        </div>
                    </div>

                    {/* Model Type */}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">模型类型</label>
                        <select 
                            value={modelForm.type}
                            onChange={(e) => setModelForm({...modelForm, type: e.target.value, features: []})}
                            className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen appearance-none cursor-pointer"
                        >
                            {MODEL_CONFIG_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>

                    {/* Description */}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">描述</label>
                        <textarea 
                            placeholder="模型描述..."
                            value={modelForm.description}
                            onChange={(e) => setModelForm({...modelForm, description: e.target.value})}
                            className="w-full p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen min-h-[80px] resize-none"
                        />
                    </div>

                    {/* Features - Dynamic based on type (Ported from AdminPortal) */}
                    <div className="space-y-2">
                        <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest block">模型特性</label>
                        <div className="flex flex-wrap gap-2">
                            {(() => {
                                const featuresByType: Record<string, {key: string, label: string}[]> = {
                                    llm: [
                                        { key: 'structured_output', label: '结构化输出' },
                                        { key: 'tool_call', label: '工具调用' },
                                        { key: 'thinking', label: '思维链' },
                                        { key: 'streaming', label: '流式输出' },
                                    ],
                                    multimodal: [
                                        { key: 'image_input', label: '图像输入' },
                                        { key: 'image_generate', label: '图像生成' },
                                        { key: 'tts', label: '文字转语音' },
                                        { key: 'speech2text', label: '语音转文字' },
                                        { key: 'thinking', label: '思维链' },
                                        { key: 'tool_call', label: '工具调用' },
                                        { key: 'structured_output', label: '结构化输出' },
                                    ],
                                    embedding: [
                                        { key: 'batch', label: '批量处理' },
                                        { key: 'sparse', label: '稀疏向量' },
                                        { key: 'dense', label: '稠密向量' },
                                    ],
                                    rerank: [
                                        { key: 'batch', label: '批量处理' },
                                        { key: 'multilingual', label: '多语言' },
                                    ],
                                    image: [
                                        { key: 'text2img', label: '文生图' },
                                        { key: 'img2img', label: '图生图' },
                                        { key: 'inpainting', label: '图像修复' },
                                        { key: 'upscale', label: '超分辨率' },
                                    ],
                                    audio: [
                                        { key: 'tts', label: '文字转语音' },
                                        { key: 'speech2text', label: '语音转文字' },
                                        { key: 'voice_clone', label: '声音克隆' },
                                        { key: 'music_gen', label: '音乐生成' },
                                    ],
                                    video: [
                                        { key: 'text2video', label: '文生视频' },
                                        { key: 'img2video', label: '图生视频' },
                                        { key: 'video_edit', label: '视频编辑' },
                                    ],
                                    code: [
                                        { key: 'completion', label: '代码补全' },
                                        { key: 'generation', label: '代码生成' },
                                        { key: 'explanation', label: '代码解释' },
                                        { key: 'refactor', label: '代码重构' },
                                        { key: 'debug', label: '调试' },
                                    ],
                                };

                                // Use mapModelType to get the key from the display string
                                const typeKey = mapModelType(modelForm.type);
                                const currentFeatures = featuresByType[typeKey] || [];
                                const selectedFeatures = modelForm.features || [];

                                return currentFeatures.map(feature => (
                                    <button
                                        key={feature.key}
                                        type="button"
                                        onClick={() => {
                                            setModelForm(prev => {
                                                const features = prev.features.includes(feature.key)
                                                    ? prev.features.filter(f => f !== feature.key)
                                                    : [...prev.features, feature.key];
                                                return { ...prev, features };
                                            });
                                        }}
                                        className={`px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all ${
                                            selectedFeatures.includes(feature.key)
                                                ? 'bg-tcm-darkGreen text-white shadow-lg'
                                                : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                                        }`}
                                    >
                                        {feature.label}
                                    </button>
                                ));
                            })()}
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        {/* Context Window */}
                        <div>
                            <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">上下文窗口</label>
                            <input 
                                type="text"
                                inputMode="numeric"
                                placeholder="128000"
                                value={modelForm.contextWindow !== undefined ? modelForm.contextWindow : ''}
                                onChange={(e) => {
                                    const val = e.target.value;
                                    if (val === '' || /^\d+$/.test(val)) {
                                        setModelForm({...modelForm, contextWindow: val === '' ? undefined : parseInt(val)});
                                    }
                                }}
                                className="w-full mt-2 p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen"
                            />
                        </div>
                        {/* Max Tokens */}
                        <div>
                            <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">最大输出TOKEN</label>
                            <input 
                                type="text"
                                inputMode="numeric"
                                placeholder="4096"
                                value={modelForm.maxTokens !== undefined ? modelForm.maxTokens : ''}
                                onChange={(e) => {
                                    const val = e.target.value;
                                    if (val === '' || /^\d+$/.test(val)) {
                                        setModelForm({...modelForm, maxTokens: val === '' ? undefined : parseInt(val)});
                                    }
                                }}
                                className="w-full mt-2 p-3 bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white text-xs focus:outline-none focus:border-tcm-lightGreen"
                            />
                        </div>
                    </div>

                    {/* Temperature */}
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">Temperature (温度)</label>
                            <span className="text-xs font-mono font-bold text-tcm-lightGreen">{modelForm.temperature.toFixed(2)}</span>
                        </div>
                        <input 
                            type="range" 
                            min="0" max="2" step="0.1"
                            value={modelForm.temperature}
                            onChange={(e) => setModelForm({...modelForm, temperature: parseFloat(e.target.value)})}
                            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-lightGreen"
                        />
                        <div className="flex justify-between text-[10px] text-gray-400 font-bold uppercase">
                            <span>精确 0</span>
                            <span>平衡 1</span>
                            <span>创意 2</span>
                        </div>
                    </div>

                    {/* Top P */}
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <label className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">TOP P (核采样)</label>
                            <span className="text-xs font-mono font-bold text-tcm-lightGreen">{modelForm.topP.toFixed(2)}</span>
                        </div>
                        <input 
                            type="range" 
                            min="0" max="1" step="0.05"
                            value={modelForm.topP}
                            onChange={(e) => setModelForm({...modelForm, topP: parseFloat(e.target.value)})}
                            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-lightGreen"
                        />
                        <div className="flex justify-between text-[10px] text-gray-400 font-bold uppercase">
                            <span>聚焦 0</span>
                            <span>0.5</span>
                            <span>全部 1</span>
                        </div>
                    </div>

                    {/* Enabled Toggle */}
                    <div className="pt-2 flex items-center justify-between">
                        <span className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest">启用状态</span>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => setModelForm({...modelForm, enabled: !modelForm.enabled})}
                                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                                    modelForm.enabled ? 'bg-tcm-lightGreen' : 'bg-gray-200 dark:bg-gray-700'
                                }`}
                            >
                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${modelForm.enabled ? 'translate-x-6' : 'translate-x-1'}`} />
                            </button>
                            <span className="text-xs text-gray-600 dark:text-gray-300">{modelForm.enabled ? '已启用' : '已禁用'}</span>
                        </div>
                    </div>

                </div>
                <div className="p-6 bg-gray-50/50 dark:bg-black/10 border-t border-gray-100 dark:border-gray-700 flex gap-3 shrink-0">
                    <button 
                        onClick={() => {
                          setShowAddModelModal(false);
                          setEditingModelId(null);
                          setModelForm({
                            id: '',
                            label: '',
                            type: 'llm',
                            description: '',
                            features: [],
                            contextWindow: undefined,
                            maxTokens: undefined,
                            temperature: 0.7,
                            topP: 1.0,
                            enabled: true
                          });
                        }}
                        className="flex-1 py-3 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-600 dark:text-gray-200 text-xs font-bold rounded-lg transition-colors"
                    >
                        取消
                    </button>
                    <button 
                        onClick={editingModelId ? handleUpdateCustomModel : handleCreateCustomModel}
                        className="flex-1 py-3 bg-tcm-darkGreen hover:bg-tcm-lightGreen text-white text-xs font-bold rounded-lg transition-colors shadow-lg"
                    >
                        {editingModelId ? '更新' : '创建'}
                    </button>
                </div>
            </div>
        </div>
      )}

    </div>
  );
};

export default PublicModelManagementPage;
