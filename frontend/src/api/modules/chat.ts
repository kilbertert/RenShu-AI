import request, { streamRequest } from '../request';
import type { ApiResponse } from '../types';
import type { ChatRequest, PersonaAnalysisRequest, ChatResumeRequest } from '../types/chat.types';

export interface AttachmentRecord {
    id: string;
    conversation_id: string;
    kind: 'generic_image' | 'tongue_image' | 'medical_report';
    original_filename: string;
    mime_type: string;
    size_bytes: number;
    sha256: string;
    status: 'uploaded' | 'attached' | 'analyzed' | 'analysis_failed';
    download_url: string;
    analysis_result?: Record<string, any> | null;
}

export const chatApi = {
    uploadAttachment: (
        conversationId: string,
        file: File,
        kind: AttachmentRecord['kind'] = 'tongue_image'
    ): Promise<ApiResponse<AttachmentRecord>> => {
        const form = new FormData();
        form.append('conversation_id', conversationId);
        form.append('kind', kind);
        form.append('file', file);
        return request.post('/api/v1/attachments', form, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },

    downloadAttachment: (url: string): Promise<Blob> => {
        return request.get(url, { responseType: 'blob' }) as unknown as Promise<Blob>;
    },

    // 普通请求（使用统一的 axios 封装）
    generate: (data: ChatRequest): Promise<ApiResponse<any>> => {
        return request.post('/api/v1/chat/generate', data);
    },
    
    // 流式请求（使用企业级 SSE 封装）
    generateStream: async (
        data: ChatRequest,
        onMessage: (data: { type: string; content?: string; data?: Record<string, any>; query_type?: string; steps?: string[] }) => void,
        onError?: (error: Error) => void,
        onComplete?: () => void
    ): Promise<() => void> => {
        return streamRequest({
            url: '/api/v1/chat/generate',
            method: 'POST',
            data,
            onMessage: (parsed) => {
                if (parsed.type === 'error') {
                    onError?.(new Error(parsed.content || 'Unknown error'));
                } else {
                    onMessage(parsed);
                }
            },
            onError,
            onComplete
        });
    },
    
    // 用户画像分析（使用统一的 axios 封装）
    analyzePersona: (data: PersonaAnalysisRequest): Promise<any> => {
        return request.post('/api/v1/chat/analyze_persona', data);
    },

    // 恢复被 interrupt 暂停的流式请求
    resumeStream: async (
        data: ChatResumeRequest,
        onMessage: (data: { type: string; content?: string; data?: Record<string, any>; question?: string; thread_id?: string; query_type?: string; steps?: string[] }) => void,
        onError?: (error: Error) => void,
        onComplete?: () => void
    ): Promise<() => void> => {
        return streamRequest({
            url: '/api/v1/chat/resume',
            method: 'POST',
            data,
            onMessage: (parsed) => {
                if (parsed.type === 'error') {
                    onError?.(new Error(parsed.content || 'Unknown error'));
                } else {
                    onMessage(parsed);
                }
            },
            onError,
            onComplete
        });
    }
}
