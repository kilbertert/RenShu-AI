import request, { streamRequest } from '../request';
import type { ApiResponse } from '../types';
import type { ChatRequest, PersonaAnalysisRequest, ChatResumeRequest } from '../types/chat.types';

export const chatApi = {
    // 普通请求（使用统一的 axios 封装）
    generate: (data: ChatRequest): Promise<ApiResponse<any>> => {
        return request.post('/api/v1/chat/generate', data);
    },
    
    // 流式请求（使用企业级 SSE 封装）
    generateStream: async (
        data: ChatRequest,
        onMessage: (data: { type: string; content?: string; query_type?: string; steps?: string[] }) => void,
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
        onMessage: (data: { type: string; content?: string; question?: string; thread_id?: string; query_type?: string; steps?: string[] }) => void,
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
