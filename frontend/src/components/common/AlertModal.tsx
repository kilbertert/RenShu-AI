import React from 'react';
import { Icons } from './Icons';

interface AlertModalProps {
  isOpen: boolean;
  onConfirm: () => void;
  title: string;
  description: string;
  variant?: 'public' | 'professional' | 'admin';
  confirmText?: string;
  onCancel?: () => void;
  cancelText?: string;
  icon?: keyof typeof Icons;
}

export const AlertModal: React.FC<AlertModalProps> = ({ 
  isOpen, 
  onConfirm, 
  onCancel,
  title,
  description,
  variant = 'public',
  confirmText = '确定',
  cancelText = '取消',
  icon = 'AlertTriangle'
}) => {
  if (!isOpen) return null;

  // Reuse the themes from LogoutConfirmModal for consistency
  const themes = {
    public: {
      overlay: 'bg-black/20 backdrop-blur-[2px]',
      container: 'bg-[#fcfbf7] rounded-[32px] border border-white shadow-2xl',
      title: 'text-tcm-darkGreen font-serif-sc',
      description: 'text-gray-500 font-sans',
      iconBg: 'bg-orange-50 text-orange-500',
      confirmBtn: 'bg-tcm-darkGreen text-white hover:bg-tcm-lightGreen shadow-lg',
      cancelBtn: 'bg-gray-100 text-gray-500 hover:bg-gray-200',
      decoration: 'bg-tcm-lightGreen/20'
    },
    professional: {
      overlay: 'bg-slate-900/40 backdrop-blur-sm',
      container: 'bg-white rounded-2xl border border-slate-100 shadow-2xl',
      title: 'text-tcm-darkGreen font-serif-sc',
      description: 'text-slate-500 font-sans',
      iconBg: 'bg-tcm-gold/10 text-tcm-gold',
      confirmBtn: 'bg-tcm-darkGreen text-white hover:bg-tcm-lightGreen shadow-lg',
      cancelBtn: 'bg-slate-50 text-slate-400 border border-slate-100 hover:bg-slate-100',
      decoration: 'bg-tcm-gold/30'
    },
    admin: {
      overlay: 'bg-black/70 backdrop-blur-md',
      container: 'bg-slate-900 rounded-2xl border border-slate-700 shadow-[0_0_50px_rgba(0,0,0,0.5)]',
      title: 'text-white font-sans uppercase tracking-widest',
      description: 'text-slate-400 font-sans',
      iconBg: 'bg-red-500/10 text-red-500',
      confirmBtn: 'bg-red-600 text-white hover:bg-red-500 shadow-[0_0_20px_rgba(220,38,38,0.2)]',
      cancelBtn: 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700',
      decoration: 'bg-red-600/50'
    }
  }[variant];

  const IconComponent = Icons[icon] || Icons.AlertTriangle;

  return (
    <div className="fixed inset-0 z-[999] flex items-center justify-center p-4 animate-in fade-in duration-300">
      {/* Overlay */}
      <div 
        className={`absolute inset-0 ${themes.overlay}`} 
        onClick={onCancel || onConfirm}
      ></div>

      {/* Modal Body */}
      <div className={`relative w-full max-w-[420px] overflow-hidden animate-in zoom-in-95 duration-200 ${themes.container}`}>
        <div className="p-8 md:p-10">
          <div className="flex items-start gap-5">
            {/* Icon */}
            <div className={`p-4 rounded-full shrink-0 ${themes.iconBg}`}>
              <IconComponent size={32} strokeWidth={2.5} />
            </div>
            
            {/* Content */}
            <div className="space-y-3">
              <h3 className={`text-2xl font-bold leading-tight ${themes.title}`}>
                {title}
              </h3>
              <p className={`text-sm leading-relaxed ${themes.description}`}>
                {description}
              </p>
            </div>
          </div>

          {/* Buttons */}
          <div className="mt-10 flex items-center justify-end gap-3">
            {onCancel && (
              <button 
                onClick={onCancel}
                className={`px-8 py-3 text-sm font-bold rounded-xl transition-all active:scale-95 ${themes.cancelBtn}`}
              >
                {cancelText}
              </button>
            )}
            <button 
              onClick={onConfirm}
              className={`px-8 py-3 text-sm font-bold rounded-xl transition-all active:scale-95 ${themes.confirmBtn}`}
            >
              {confirmText}
            </button>
          </div>
        </div>
        
        {/* Decoration */}
        <div className={`h-1.5 w-full opacity-50 ${themes.decoration}`}></div>
      </div>
    </div>
  );
};
