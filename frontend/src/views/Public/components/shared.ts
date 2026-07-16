import type { ChatMessage, UserPersona, BaseProfile } from '../../../types';

export interface Attachment {
  file: File;
  previewUrl: string;
  base64: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  lastModified: Date;
  persona?: UserPersona;
  healthScore?: number;
  threadId?: string;
  isInterrupted?: boolean;
}

export const PERSONA_FIELD_LABELS: Record<string, string> = {
  age: '年龄',
  gender: '性别',
  chief_complaint: '主诉症状',
  suspected_diagnosis: '疑似诊断',
  recommended_treatment: '调理建议',
};

export const groupSessionsByDate = (sessions: ChatSession[]) => {
  const groups: { [key: string]: ChatSession[] } = {
    '今天': [],
    '昨天': [],
    '最近7天': [],
    '更早': [],
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

export const buildDefaultPersona = (baseProfile?: BaseProfile): UserPersona => ({
  age: '',
  gender: '',
  chief_complaint: '待分析...',
  suspected_diagnosis: '分析待定',
  recommended_treatment: ' wellness 建议',
  health_score: 100,
  base_profile: baseProfile,
});
