import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Icons } from '../../../components/common/Icons';
import { BrandLogo } from '../../../components/common/BrandLogo';
import type { AIModelConfig, CustomModel } from '../../../types';
import { getProviderIconPath, getModelIconPath, isDarkInvert } from '../../../utils/iconMap';

export type RightPanelMode = 'health' | 'settings' | 'history';

export interface ModelSelectorProvider {
  id: string;
  name: string;
  icon: string;
  isBuiltin?: boolean;
}

interface ModelSelectorProps {
  selectedProviderId: string;
  selectedModel: AIModelConfig;
  providers: ModelSelectorProvider[];
  filteredModels: AIModelConfig[];
  rightPanelMode: RightPanelMode;
  showRightSidebar: boolean;
  onProviderChange: (providerId: string) => void;
  onModelSelect: (model: AIModelConfig) => void;
  onOpenModelSettings: (model: AIModelConfig) => void;
  onToggleRightSidebar: () => void;
}

const ModelSelectorInner: React.FC<ModelSelectorProps> = ({
  selectedProviderId,
  selectedModel,
  providers,
  filteredModels,
  rightPanelMode,
  showRightSidebar,
  onProviderChange,
  onModelSelect,
  onOpenModelSettings,
  onToggleRightSidebar,
}) => {
  const navigate = useNavigate();
  const [showProviderSelector, setShowProviderSelector] = useState(false);
  const [showModelSelector, setShowModelSelector] = useState(false);

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div
          onClick={() => setShowProviderSelector(!showProviderSelector)}
          className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-white/5 px-3 py-1.5 rounded-lg transition-colors group"
        >
          <BrandLogo size="sm" showText={true} />
          <Icons.ChevronDown
            size={14}
            className={`text-gray-400 transition-transform duration-300 ${showProviderSelector ? 'rotate-180' : ''}`}
          />
        </div>

        {showProviderSelector && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setShowProviderSelector(false)}
            ></div>
            <div className="absolute top-full left-0 mt-2 w-64 bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 origin-top-left">
              <div className="p-1.5 space-y-0.5 max-h-64 overflow-y-auto custom-scrollbar">
                <div className="px-2 py-1.5 text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider flex justify-between">
                  <span>选择模型提供商</span>
                  <span></span>
                </div>
                {providers.map(p => (
                  <button
                    key={p.id}
                    onClick={() => {
                      onProviderChange(p.id);
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
                        return iconPath ? (
                          <img
                            src={iconPath}
                            alt={p.name}
                            className={`w-full h-full object-contain ${shouldInvert ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`}
                          />
                        ) : (
                          p.icon
                        );
                      })()}
                    </span>
                    <div className="flex flex-col">
                      <span className="text-xs font-bold">{p.name}</span>
                      {!p.isBuiltin && (
                        <span className="text-[9px] text-tcm-darkGreen dark:text-tcm-lightGreen bg-tcm-lightGreen/10 px-1 py-0.5 rounded w-fit mt-0.5">
                          我的服务
                        </span>
                      )}
                    </div>
                    {selectedProviderId === p.id && <Icons.Check size={14} className="ml-auto" />}
                  </button>
                ))}
                {providers.length === 0 && (
                  <div className="p-4 text-center text-xs text-gray-400">暂无提供商</div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="relative">
        <button
          onClick={() => setShowModelSelector(!showModelSelector)}
          className={`flex items-center justify-between gap-2 w-52 bg-white dark:bg-white/5 border px-3 py-1.5 rounded-lg text-xs font-bold text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-white/10 transition-colors shadow-sm ${
            showModelSelector
              ? 'border-tcm-lightGreen ring-2 ring-tcm-lightGreen/20'
              : 'border-gray-200 dark:border-white/10'
          }`}
        >
          <div className="flex items-center gap-2 truncate">
            {(() => {
              const modelIcon = getModelIconPath(selectedModel.id);
              if (modelIcon) {
                return (
                  <img
                    src={modelIcon}
                    alt={selectedModel.name}
                    className={`w-4 h-4 object-contain ${isDarkInvert(selectedModel.id) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`}
                  />
                );
              }
              return <Icons.Zap size={14} className="text-tcm-gold flex-shrink-0" />;
            })()}
            <span className="truncate">{selectedModel.name}</span>
            {!selectedModel.isBuiltin && (
              <span className="text-[9px] text-tcm-darkGreen dark:text-emerald-300 bg-tcm-lightGreen/10 px-1 py-0.5 rounded w-fit flex-shrink-0">
                me
              </span>
            )}
          </div>
          <Icons.ChevronDown
            size={12}
            className={`text-gray-400 flex-shrink-0 transition-transform duration-300 ${showModelSelector ? 'rotate-180' : ''}`}
          />
        </button>

        {showModelSelector && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setShowModelSelector(false)}
            ></div>
            <div className="absolute top-full right-0 mt-2 w-full bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 origin-top-right">
              <div className="p-1.5 space-y-0.5">
                <div className="px-2 py-1.5 text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider">
                  选择已适配模型
                </div>
                {filteredModels.length === 0 ? (
                  <div className="p-4 text-center text-xs text-gray-400">
                    暂无已启用模型。
                    <br />
                    请前往{' '}
                    <span
                      className="text-tcm-lightGreen cursor-pointer"
                      onClick={() => navigate('/public/models')}
                    >
                      模型管理
                    </span>{' '}
                    启用。
                  </div>
                ) : (
                  filteredModels.map(model => (
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
                            onModelSelect(model);
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
                                    <img
                                      src={modelIcon}
                                      alt={model.name}
                                      className={`w-5 h-5 object-contain ${isDarkInvert(model.id) ? 'dark:invert dark:brightness-0 dark:invert-1' : 'dark:brightness-150 dark:contrast-150'}`}
                                    />
                                  </div>
                                );
                              }
                              return (
                                <div
                                  className={`mt-0.5 p-1 rounded-md flex-shrink-0 w-6 h-6 flex items-center justify-center ${
                                    selectedModel.id === model.id
                                      ? 'bg-tcm-lightGreen text-white'
                                      : 'bg-gray-100 dark:bg-white/10 text-gray-500 dark:text-gray-400'
                                  }`}
                                >
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
                                <div
                                  className={`text-xs font-bold truncate ${
                                    selectedModel.id === model.id
                                      ? 'text-tcm-darkGreen dark:text-tcm-lightGreen'
                                      : 'text-gray-700 dark:text-gray-200'
                                  }`}
                                >
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
                          onClick={e => {
                            e.stopPropagation();
                            onOpenModelSettings(model);
                            setShowModelSelector(false);
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
                  ))
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <button
        onClick={onToggleRightSidebar}
        className={`p-2 rounded-full transition-colors ${
          showRightSidebar
            ? 'bg-tcm-lightGreen/10 text-tcm-lightGreen'
            : 'text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5'
        }`}
      >
        <Icons.Activity size={20} />
      </button>
    </div>
  );
};

export const ModelSelector = React.memo(ModelSelectorInner);
ModelSelector.displayName = 'ModelSelector';
