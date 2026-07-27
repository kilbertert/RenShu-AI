import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Icons } from '../../../components/common/Icons';
import { BrandLogo } from '../../../components/common/BrandLogo';
import type { User, AIModelConfig } from '../../../types';
import { chatApi } from '../../../api/modules/chat';
import type { ChatSession, Attachment } from './shared';

interface ChatPanelProps {
  user: User;
  currentSession: ChatSession | null;
  selectedModel: AIModelConfig;
  enableThinking: boolean;
  isLoading: boolean;
  showLeftSidebar: boolean;
  globalSendingLockRef: React.MutableRefObject<boolean>;
  onOpenLeftSidebar: () => void;
  onSendMessage: (overrideText?: string, attachments?: Attachment[]) => Promise<boolean>;
  onCancelGeneration: () => void;
  onDeleteMessage: (messageId: string) => void;
  onToggleThinking: () => void;
}

type MessageAttachment = NonNullable<ChatSession['messages'][number]['attachments']>[number];

const SecureAttachmentImage: React.FC<{ attachment: MessageAttachment }> = ({ attachment }) => {
  const [src, setSrc] = useState<string>('');
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl = '';
    setFailed(false);
    chatApi.downloadAttachment(attachment.url)
      .then(blob => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [attachment.url]);

  if (failed) {
    return <div className="text-xs text-red-500 mb-2">图片加载失败或已无权访问</div>;
  }
  if (!src) {
    return <div className="h-32 w-32 rounded-lg bg-gray-100 dark:bg-gray-800 animate-pulse mb-2" />;
  }
  if (attachment.mime_type === 'application/pdf' || attachment.type === 'file') {
    return (
      <a
        href={src}
        target="_blank"
        rel="noreferrer"
        className="mb-2 flex max-w-sm items-center gap-2 rounded-lg border border-gray-200 bg-white/60 px-3 py-2 text-sm text-tcm-darkGreen hover:bg-white dark:border-gray-700 dark:bg-gray-800/60 dark:text-tcm-lightGreen"
      >
        <Icons.FileText size={18} />
        <span className="truncate">{attachment.name || '医疗报告.pdf'}</span>
      </a>
    );
  }
  return <img src={src} alt={attachment.name || '舌像'} className="h-32 rounded-lg border mb-2" />;
};

const TongueAnalysisCard: React.FC<{ analysis: Record<string, any> }> = ({ analysis }) => (
  <div className="mt-3 rounded-xl border border-tcm-lightGreen/30 bg-tcm-lightGreen/5 p-3 text-sm">
    <div className="font-semibold text-tcm-darkGreen dark:text-tcm-lightGreen mb-2">舌像结构化观察</div>
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-700 dark:text-gray-300">
      <span>舌色：{analysis.tongue_color || '未识别'}</span>
      <span>舌形：{analysis.tongue_shape || '未识别'}</span>
      <span>苔色：{analysis.coating_color || '未识别'}</span>
      <span>苔质：{analysis.coating_quality || '未识别'}</span>
      <span>图片质量：{analysis.image_quality || '未知'}</span>
      <span>置信度：{Math.round(Number(analysis.confidence || 0) * 100)}%</span>
    </div>
    {Array.isArray(analysis.syndrome_hints) && analysis.syndrome_hints.length > 0 && (
      <div className="mt-2 text-gray-700 dark:text-gray-300">
        证型提示：{analysis.syndrome_hints.join('、')}
      </div>
    )}
    <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
      {analysis.warning || '舌象只能作为四诊合参的一部分，不能单独作为诊断或用药依据。'}
    </div>
  </div>
);

const REPORT_FLAG_LABELS: Record<string, string> = {
  high: '偏高',
  low: '偏低',
  abnormal: '异常',
  normal: '正常',
  positive: '阳性',
  negative: '阴性',
  unknown: '未标记',
};

const ReportAnalysisCard: React.FC<{ analysis: Record<string, any> }> = ({ analysis }) => {
  const abnormalMetrics = Array.isArray(analysis.metrics)
    ? analysis.metrics.filter((item: any) =>
        ['high', 'low', 'abnormal', 'positive'].includes(item?.abnormal_flag)
      )
    : [];
  return (
    <div className="mt-3 rounded-xl border border-blue-300/40 bg-blue-50/60 p-3 text-sm dark:border-blue-700/40 dark:bg-blue-950/20">
      <div className="mb-2 font-semibold text-blue-800 dark:text-blue-300">
        医疗报告结构化解读
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-700 dark:text-gray-300">
        <span>类型：{analysis.report_type || '未识别'}</span>
        <span>日期：{analysis.report_date || '未识别'}</span>
        <span>解析方式：{analysis.extraction_method || '未知'}</span>
        <span>页码：{Array.isArray(analysis.analyzed_pages) ? analysis.analyzed_pages.join('、') : '1'}</span>
      </div>
      {analysis.summary && <div className="mt-2 text-gray-700 dark:text-gray-300">{analysis.summary}</div>}
      {abnormalMetrics.length > 0 && (
        <div className="mt-2 space-y-1">
          <div className="font-medium text-amber-800 dark:text-amber-300">报告标记的异常项</div>
          {abnormalMetrics.slice(0, 8).map((item: any, index: number) => (
            <div key={`${item.name}-${index}`} className="text-gray-700 dark:text-gray-300">
              {item.name || '未命名指标'}：{item.value || '未识别'}{item.unit || ''}
              {item.reference_range ? `（参考 ${item.reference_range}）` : ''}
              <span className="ml-1 text-amber-700 dark:text-amber-300">
                {REPORT_FLAG_LABELS[item.abnormal_flag] || '异常'}
              </span>
            </div>
          ))}
        </div>
      )}
      {Array.isArray(analysis.medical_attention_advice) && analysis.medical_attention_advice.length > 0 && (
        <div className="mt-2 text-gray-700 dark:text-gray-300">
          建议：{analysis.medical_attention_advice.join('；')}
        </div>
      )}
      <div className={`mt-2 text-xs ${analysis.urgent_warning ? 'text-red-600 dark:text-red-300' : 'text-amber-700 dark:text-amber-300'}`}>
        {analysis.warning || '报告解读仅供辅助理解，不能替代医生诊断。'}
      </div>
    </div>
  );
};

const GRAPH_ROLE_LABELS: Record<string, string> = {
  main: '主症',
  supplement: '兼症',
  tongue: '舌象',
  pulse: '脉象',
};

const DiagnosisEvidenceCard: React.FC<{ diagnosis: Record<string, any> }> = ({ diagnosis }) => {
  const citations = Array.isArray(diagnosis.citations)
    ? diagnosis.citations.filter((item: any) => item?.source_type === 'graph_path')
    : [];
  if (!diagnosis.syndrome && citations.length === 0) return null;
  return (
    <div className="mt-3 rounded-xl border border-emerald-300/40 bg-emerald-50/60 p-3 text-sm dark:border-emerald-700/40 dark:bg-emerald-950/20">
      <div className="mb-2 font-semibold text-emerald-800 dark:text-emerald-300">
        结构化辨证与图谱证据
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-gray-700 dark:text-gray-300">
        <span>主证：{diagnosis.syndrome || '未明确'}</span>
        <span>置信度：{Math.round(Number(diagnosis.confidence || 0) * 100)}%</span>
        {diagnosis.syndrome_id && <span>知识库 ID：{diagnosis.syndrome_id}</span>}
      </div>
      {citations.length > 0 && (
        <div className="mt-3 space-y-2">
          <div className="font-medium text-emerald-800 dark:text-emerald-300">
            Neo4j 可追溯证据
          </div>
          {citations.slice(0, 4).map((citation: any, index: number) => (
            <div
              key={citation.citation_id || `${citation.title}-${index}`}
              className="rounded-lg border border-emerald-200/60 bg-white/60 px-3 py-2 text-xs text-gray-700 dark:border-emerald-800/60 dark:bg-gray-900/30 dark:text-gray-300"
            >
              <div className="font-medium text-gray-800 dark:text-gray-200">
                {citation.title || `图谱证据 ${index + 1}`}
              </div>
              {citation.evidence && <div className="mt-1">{citation.evidence}</div>}
              <div className="mt-1 text-emerald-700 dark:text-emerald-300">
                来源：{citation.source || '未知'}
                {citation.symptom_role
                  ? ` · ${GRAPH_ROLE_LABELS[citation.symptom_role] || citation.symptom_role}`
                  : ''}
                {citation.evidence_weight != null
                  ? ` · 权重 ${Number(citation.evidence_weight).toFixed(1)}`
                  : ''}
              </div>
              {Array.isArray(citation.relationship_path) && citation.relationship_path.length > 0 && (
                <div className="mt-1 break-all text-gray-500 dark:text-gray-400">
                  路径：{citation.relationship_path.join(' ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
        图谱证据用于辅助辨证，不代表临床确诊或处方依据。
      </div>
    </div>
  );
};

const ChatPanelInner: React.FC<ChatPanelProps> = ({
  user,
  currentSession,
  selectedModel,
  enableThinking,
  isLoading,
  showLeftSidebar,
  globalSendingLockRef,
  onOpenLeftSidebar,
  onSendMessage,
  onCancelGeneration,
  onDeleteMessage,
  onToggleThinking,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [attachmentKind, setAttachmentKind] = useState<Attachment['kind']>('tongue_image');
  const [attachmentError, setAttachmentError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isNearBottomRef = useRef(true);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (
      !selectedModel.supportsVision
      && (attachmentKind === 'tongue_image' || file.type !== 'application/pdf')
    ) {
      setAttachmentError('当前模型未声明图片输入能力，请先选择支持视觉的模型。');
      return;
    }
    const allowedTypes = attachmentKind === 'medical_report'
      ? ['image/jpeg', 'image/png', 'image/webp', 'application/pdf']
      : ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setAttachmentError(
        attachmentKind === 'medical_report'
          ? '医疗报告仅支持 JPEG、PNG、WebP 或 PDF。'
          : '舌像仅支持 JPEG、PNG 或 WebP 图片。'
      );
      return;
    }
    if (file.size <= 0 || file.size > 8 * 1024 * 1024) {
      setAttachmentError('附件不能为空且不能超过 8MB。');
      return;
    }
    attachments.forEach(item => URL.revokeObjectURL(item.previewUrl));
    setAttachmentError('');
    setAttachments([{ file, previewUrl: URL.createObjectURL(file), kind: attachmentKind }]);
  };

  const removeAttachment = (index: number) => {
    const target = attachments[index];
    if (target) URL.revokeObjectURL(target.previewUrl);
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const submitMessage = async (text: string) => {
    if (!text.trim() && attachments.length === 0) return;
    const pending = [...attachments];
    const accepted = await onSendMessage(text, pending);
    if (accepted) {
      pending.forEach(item => URL.revokeObjectURL(item.previewUrl));
      setAttachments([]);
      setAttachmentError('');
      setInputValue('');
    }
  };

  const handleVoiceInput = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'zh-CN';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onstart = () => setInputValue(prev => prev + ' (正在倾听...)');
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(prev => prev.replace(' (正在倾听...)', '') + transcript);
      };
      recognition.onerror = () => setInputValue(prev => prev.replace(' (正在倾听...)', ''));
      recognition.start();
    } else {
      alert('您的浏览器不支持语音识别功能。');
    }
  };

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const onScroll = () => {
      const distanceFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      isNearBottomRef.current = distanceFromBottom < 120;
    };
    container.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    return () => container.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (!isNearBottomRef.current) return;
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages]);

  return (
    <main className="flex-1 flex flex-col relative z-10 transition-colors duration-500 min-w-0 min-h-0 bg-transparent">
      <header className="h-16 flex items-center justify-between px-6 z-10 transition-colors bg-white/80 dark:bg-[#131314]/80 backdrop-blur-sm border-b border-tcm-lightGreen/5">
        <div className="flex items-center gap-3">
          {!showLeftSidebar && (
            <button
              onClick={onOpenLeftSidebar}
              className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-white/5 rounded-full transition-colors"
            >
              <Icons.Menu size={20} />
            </button>
          )}
        </div>
      </header>

      <div
        ref={messagesContainerRef}
        className="flex-1 min-h-0 overflow-y-auto p-4 md:p-8 space-y-8 scroll-smooth"
      >
        <div className="max-w-4xl mx-auto w-full space-y-8">
          {currentSession ? (
            currentSession.messages.map(msg => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in-up group w-full relative`}
              >
                {msg.role === 'model' && (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-tcm-lightGreen to-tcm-darkGreen text-white flex items-center justify-center shadow-lg mr-4 flex-shrink-0 mt-1 overflow-hidden">
                    <Icons.Zap size={14} />
                  </div>
                )}
                <div className={`space-y-2 max-w-[85%] relative group-hover:z-10`}>
                  {msg.role === 'user' ? (
                    <div className="px-5 py-3 text-gray-800 dark:text-white leading-relaxed whitespace-pre-wrap">
                      {msg.attachments?.map((att, idx) => (
                        <SecureAttachmentImage key={att.id || `${att.url}-${idx}`} attachment={att} />
                      ))}
                      {msg.text}
                      {msg.tongueAnalysis && <TongueAnalysisCard analysis={msg.tongueAnalysis} />}
                      {msg.reportAnalysis && <ReportAnalysisCard analysis={msg.reportAnalysis} />}
                    </div>
                  ) : (
                    <div className="text-gray-800 dark:text-gray-200 leading-relaxed">
                      {msg.agentSteps && msg.agentSteps.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mb-3">
                          {msg.agentSteps.map((step, idx) => {
                            const isLast = idx === msg.agentSteps!.length - 1;
                            const isActive = isLast && !msg.text;
                            return (
                              <span
                                key={idx}
                                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                                  ${
                                    isActive
                                      ? 'bg-tcm-lightGreen/20 text-tcm-darkGreen dark:text-tcm-lightGreen animate-pulse'
                                      : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
                                  }`}
                              >
                                {isActive ? '⟳' : '✓'} {step}
                              </span>
                            );
                          })}
                        </div>
                      )}
                      {msg.text ? (
                        <div
                          className="prose prose-sm md:prose-base max-w-none dark:prose-invert
                            prose-p:my-2 prose-p:leading-relaxed
                            prose-headings:text-tcm-darkGreen dark:prose-headings:text-tcm-lightGreen
                            prose-strong:text-tcm-darkGreen dark:prose-strong:text-tcm-lightGreen
                            prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
                            prose-pre:bg-gray-900 prose-pre:text-gray-100
                            prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                            prose-blockquote:border-tcm-lightGreen prose-blockquote:bg-tcm-lightGreen/5
                            prose-a:text-tcm-darkGreen hover:prose-a:text-tcm-lightGreen"
                        >
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 py-1">
                          <div className="w-2 h-2 bg-tcm-lightGreen rounded-full animate-bounce"></div>
                          <div
                            className="w-2 h-2 bg-tcm-gold rounded-full animate-bounce"
                            style={{ animationDelay: '0.1s' }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-tcm-darkGreen rounded-full animate-bounce"
                            style={{ animationDelay: '0.2s' }}
                          ></div>
                        </div>
                      )}
                      {msg.tongueAnalysis && <TongueAnalysisCard analysis={msg.tongueAnalysis} />}
                      {msg.reportAnalysis && <ReportAnalysisCard analysis={msg.reportAnalysis} />}
                      {msg.diagnosisResult && (
                        <DiagnosisEvidenceCard diagnosis={msg.diagnosisResult} />
                      )}
                    </div>
                  )}

                  <div
                    className={`absolute top-0 ${
                      msg.role === 'user' ? '-left-8' : '-right-8'
                    } opacity-0 group-hover:opacity-100 transition-opacity flex flex-col gap-1`}
                  >
                    <button
                      onClick={() => onDeleteMessage(msg.id)}
                      className="p-1 text-gray-400 hover:text-red-500 bg-white dark:bg-black/20 rounded shadow-sm"
                      title="删除消息"
                    >
                      <Icons.Trash2 size={12} />
                    </button>
                  </div>
                </div>
                {msg.role === 'user' && (
                  <img
                    src={user.avatar}
                    className="w-8 h-8 rounded-full border border-gray-200 dark:border-gray-600 ml-3 shadow-sm object-cover flex-shrink-0 mt-1"
                    alt="Me"
                  />
                )}
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center min-h-[50vh] animate-in fade-in slide-in-from-bottom-4">
              <div className="mb-8 p-4 bg-white/50 dark:bg-white/5 rounded-full shadow-sm backdrop-blur-sm">
                <BrandLogo size="lg" showText={false} />
              </div>
              <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-8 font-serif-sc">
                有什么我能帮你的吗?
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full max-w-3xl">
                {[
                  { icon: <Icons.Thermometer size={18} />, text: '我最近感觉有点上火,喉咙痛怎么办?' },
                  { icon: <Icons.Activity size={18} />, text: '分析一下我的体质健康状况' },
                  { icon: <Icons.BookOpen size={18} />, text: "解释一下'气虚'是什么意思?" },
                  { icon: <Icons.Coffee size={18} />, text: '推荐一些适合春季的养生茶饮' },
                  { icon: <Icons.Moon size={18} />, text: '最近失眠多梦,有什么调理建议?' },
                  { icon: <Icons.FileText size={18} />, text: '帮我解读一下这个体检报告' },
                ].map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      if (globalSendingLockRef.current) {
                        console.log('⚠️ [Landing按钮] 请求进行中,忽略点击');
                        return;
                      }
                      onSendMessage(item.text);
                    }}
                    className="flex items-center gap-3 p-4 bg-white dark:bg-white/5 hover:bg-gray-50 dark:hover:bg-white/10 border border-gray-100 dark:border-white/5 rounded-2xl shadow-sm hover:shadow-md transition-all text-left group"
                  >
                    <div className="p-2 bg-tcm-lightGreen/10 text-tcm-darkGreen dark:text-tcm-lightGreen rounded-lg group-hover:bg-tcm-lightGreen/20 transition-colors">
                      {item.icon}
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white transition-colors">
                      {item.text}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="p-2 z-20">
        <div className="max-w-4xl mx-auto">
          {attachments.length > 0 && (
            <div className="flex gap-3 px-4 py-2 overflow-x-auto custom-scrollbar">
              {attachments.map((att, idx) => (
                <div key={idx} className="relative group flex-shrink-0">
                  {att.file.type === 'application/pdf' ? (
                    <div className="flex h-12 max-w-48 items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 text-xs text-gray-700 shadow-sm dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200">
                      <Icons.FileText size={16} />
                      <span className="truncate">{att.file.name}</span>
                    </div>
                  ) : (
                    <img
                      src={att.previewUrl}
                      alt="preview"
                      className="h-12 w-12 rounded-lg object-cover border border-gray-200 dark:border-gray-600 shadow-sm"
                    />
                  )}
                  <button
                    onClick={() => removeAttachment(idx)}
                    className="absolute -top-1 -right-1 bg-gray-800 text-white rounded-full p-0.5 shadow-md opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Icons.X size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {attachmentError && (
            <div className="px-4 pb-2 text-xs text-red-500">{attachmentError}</div>
          )}

          <div className="bg-[#f0f4f9] dark:bg-[#1e1e1e] rounded-full p-3 flex items-center gap-2 transition-all duration-300 shadow-sm border border-gray-200 dark:border-gray-700 ring-1 ring-transparent focus-within:ring-tcm-lightGreen/30 focus-within:bg-white dark:focus-within:bg-[#252525]">
            <select
              value={attachmentKind}
              onChange={event => {
                attachments.forEach(item => URL.revokeObjectURL(item.previewUrl));
                setAttachments([]);
                setAttachmentError('');
                setAttachmentKind(event.target.value as Attachment['kind']);
              }}
              className="max-w-24 rounded-full border-0 bg-white/70 px-2 py-1 text-xs text-gray-600 focus:ring-1 focus:ring-tcm-lightGreen dark:bg-gray-800 dark:text-gray-300"
              title="附件类型"
            >
              <option value="tongue_image">舌像</option>
              <option value="medical_report">医疗报告</option>
            </select>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-1.5 text-gray-400 hover:text-tcm-darkGreen dark:text-gray-500 dark:hover:text-tcm-lightGreen transition-colors rounded-full hover:bg-gray-200/50 dark:hover:bg-white/10"
              title="Upload"
            >
              <input
                type="file"
                accept={
                  attachmentKind === 'medical_report'
                    ? 'image/jpeg,image/png,image/webp,application/pdf'
                    : 'image/jpeg,image/png,image/webp'
                }
                ref={fileInputRef}
                className="hidden"
                onChange={handleFileUpload}
              />
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
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!e.nativeEvent.isComposing) void submitMessage(inputValue);
                }
              }}
              placeholder={selectedModel.id ? '输入健康咨询问题...' : '请先选择你要使用的模型...'}
              className="flex-1 bg-transparent border-none focus:ring-0 focus:outline-none text-gray-800 dark:text-gray-100 placeholder-gray-400 resize-none py-1.5 text-sm max-h-32"
              rows={1}
              style={{ minHeight: '32px' }}
            />

            <div className="flex items-center gap-1 border-l border-gray-300 dark:border-gray-700 pl-2">
              {selectedModel.supportsThinking && (
                <button
                  onClick={onToggleThinking}
                  className={`p-1.5 rounded-full transition-all ${
                    enableThinking
                      ? 'text-tcm-darkGreen bg-tcm-lightGreen/20 dark:text-tcm-lightGreen'
                      : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
                  }`}
                  title={enableThinking ? '关闭深度思考' : '开启深度思考'}
                >
                  <Icons.BrainCircuit size={18} />
                </button>
              )}
            </div>

            {isLoading ? (
              <button
                onClick={onCancelGeneration}
                className="p-1.5 rounded-full transition-all duration-300 bg-red-500 text-white shadow-md hover:bg-red-600 transform hover:scale-105"
                title="取消生成"
              >
                <Icons.X size={18} />
              </button>
            ) : (
              <button
                onClick={() => void submitMessage(inputValue)}
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
  );
};

export const ChatPanel = React.memo(ChatPanelInner);
ChatPanel.displayName = 'ChatPanel';
