import React from 'react';
import { Icons } from '../../../components/common/Icons';
import type { AIModelConfig, UserPersona } from '../../../types';
import { PERSONA_FIELD_LABELS } from './shared';
import type { RightPanelMode } from './ModelSelector';

interface SettingsPanelProps {
  showRightSidebar: boolean;
  rightPanelMode: RightPanelMode;
  isDarkMode: boolean;
  persona: UserPersona;
  healthScore: number;
  changedFields: string[];
  temperature: number;
  topP: number;
  maxTokens: number;
  selectedModel: AIModelConfig;
  onChangeMode: (mode: RightPanelMode) => void;
  onCloseSidebar: () => void;
  onTemperatureChange: (v: number) => void;
  onTopPChange: (v: number) => void;
  onMaxTokensChange: (v: number) => void;
  onResetDefaults: () => void;
}

const SettingsPanelInner: React.FC<SettingsPanelProps> = ({
  showRightSidebar,
  rightPanelMode,
  isDarkMode,
  persona,
  healthScore,
  changedFields,
  temperature,
  topP,
  maxTokens,
  selectedModel,
  onChangeMode,
  onCloseSidebar,
  onTemperatureChange,
  onTopPChange,
  onMaxTokensChange,
  onResetDefaults,
}) => {
  return (
    <aside
      className={`${
        showRightSidebar ? 'w-80' : 'w-0'
      } flex-shrink-0 flex flex-col ${
        isDarkMode ? 'glass-panel-dark' : 'glass-panel'
      } border-l border-white/50 dark:border-white/10 z-20 shadow-xl transition-all duration-500 overflow-hidden`}
    >
      <div className="h-16 flex-shrink-0 flex items-center justify-between px-6 border-b border-gray-200/50 dark:border-white/10 bg-white/30 dark:bg-black/10">
        <div className="flex items-center">
          {rightPanelMode === 'settings' ? (
            <Icons.Sliders className="text-tcm-darkGreen mr-3" size={20} />
          ) : (
            <Icons.Leaf className="text-tcm-lightGreen mr-3" size={24} />
          )}
          <h1 className="text-lg font-bold text-tcm-darkGreen dark:text-tcm-cream font-serif-sc tracking-wide whitespace-nowrap">
            {rightPanelMode === 'settings' ? '模型参数配置' : '智能健康画像'}
          </h1>
        </div>
        {rightPanelMode === 'settings' && (
          <button
            onClick={() => onChangeMode('health')}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          >
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
                  <span className="text-tcm-darkGreen dark:text-tcm-lightGreen font-mono">
                    {temperature}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={e => onTemperatureChange(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-darkGreen"
                />
                <p className="text-[10px] text-gray-400 mt-1">
                  值越高,回复越具有创造性;值越低,回复越保守准确。
                </p>
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
                  onChange={e => onTopPChange(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-darkGreen"
                />
                <p className="text-[10px] text-gray-400 mt-1">
                  控制模型选择候选词的范围,较低的值会使回复更加专注。
                </p>
              </div>

              <div>
                <div className="flex justify-between text-xs font-bold mb-2">
                  <span className="text-gray-600 dark:text-gray-300">最大Token数</span>
                  <span className="text-tcm-darkGreen dark:text-tcm-lightGreen font-mono">
                    {maxTokens}
                  </span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="8000"
                  step="100"
                  value={maxTokens}
                  onChange={e => onMaxTokensChange(parseInt(e.target.value, 10))}
                  className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-tcm-darkGreen"
                />
                <p className="text-[10px] text-gray-400 mt-1">
                  限制模型单次回复生成的最大长度。
                </p>
              </div>

              <div className="pt-4 border-t border-gray-200 dark:border-white/10">
                <button
                  onClick={onResetDefaults}
                  className="w-full py-2 text-xs text-gray-500 hover:text-tcm-darkGreen dark:text-gray-400 dark:hover:text-tcm-lightGreen transition-colors flex items-center justify-center gap-2"
                >
                  <Icons.RotateCcw size={12} />
                  恢复默认设置
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="bg-white/60 dark:bg-white/5 p-4 rounded-xl border border-white dark:border-white/10 shadow-sm backdrop-blur-sm">
                <div className="flex justify-between items-end mb-2">
                  <span className="text-sm font-bold text-gray-600 dark:text-gray-300">
                    体质健康分
                  </span>
                  <span className="text-3xl font-serif-sc font-bold text-tcm-darkGreen dark:text-tcm-lightGreen">
                    {healthScore}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-tcm-lightGreen to-tcm-gold transition-all duration-1000"
                    style={{ width: `${healthScore}%` }}
                  ></div>
                </div>
              </div>

              {(Object.keys(persona) as Array<keyof UserPersona>).map(key => {
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
                      {PERSONA_FIELD_LABELS[key] ||
                        key.replace(/([A-Z])/g, ' $1').trim()}
                      {changedFields.includes(key) && (
                        <span className="w-2 h-2 bg-tcm-accent rounded-full animate-ping"></span>
                      )}
                    </div>
                    <div className="font-serif-sc text-tcm-darkGreen dark:text-tcm-cream text-sm font-medium leading-relaxed">
                      {(persona[key] as string) || '未录入'}
                    </div>
                  </div>
                );
              })}

              <div className="bg-white/40 dark:bg-white/5 p-3 rounded-lg border border-transparent hover:border-tcm-lightGreen/30 transition-all">
                <div className="text-[10px] text-gray-500 dark:text-gray-400 uppercase mb-2 border-b border-gray-200 dark:border-white/10 pb-1">
                  基础健康画像
                </div>
                <div className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                  {persona.base_profile &&
                  (persona.base_profile.constitution_type ||
                    persona.base_profile.medical_history ||
                    persona.base_profile.allergy_info ||
                    (persona.base_profile.taboo_items &&
                      persona.base_profile.taboo_items.length > 0)) ? (
                    <>
                      {persona.base_profile.constitution_type && (
                        <div>
                          <span className="opacity-70">体质:</span>{' '}
                          {persona.base_profile.constitution_type}
                        </div>
                      )}
                      {persona.base_profile.medical_history && (
                        <div>
                          <span className="opacity-70">病史:</span>{' '}
                          {persona.base_profile.medical_history}
                        </div>
                      )}
                      {persona.base_profile.allergy_info && (
                        <div>
                          <span className="opacity-70">过敏:</span>{' '}
                          {persona.base_profile.allergy_info}
                        </div>
                      )}
                      {persona.base_profile.taboo_items &&
                        persona.base_profile.taboo_items.length > 0 && (
                          <div>
                            <span className="opacity-70">禁忌:</span>{' '}
                            {persona.base_profile.taboo_items.join(', ')}
                          </div>
                        )}
                    </>
                  ) : (
                    <div className="text-gray-400 italic">
                      暂无基础数据,请在个人资料中完善
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </aside>
  );
};

export const SettingsPanel = React.memo(SettingsPanelInner);
SettingsPanel.displayName = 'SettingsPanel';
