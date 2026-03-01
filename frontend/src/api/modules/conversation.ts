import request from '../request';
import type { ApiResponse } from '../types';

export const conversationApi = {
    // Get current user's conversation list
    getConversations: (): Promise<ApiResponse<any[]>> => {
        return request.get('/api/v1/conversations/me');
    },

    // Get messages for a specific conversation
    getMessages: (conversationId: string): Promise<ApiResponse<any[]>> => {
        return request.post('/api/v1/conversations/messages', { conversation_id: conversationId });
    },

    // Delete a conversation
    deleteConversation: (conversationId: string): Promise<ApiResponse<any>> => {
        return request.post('/api/v1/conversations/delete', { conversation_id: conversationId });
    },

    // Delete a message
    deleteMessage: (messageId: string): Promise<ApiResponse<any>> => {
        return request.post('/api/v1/conversations/messages/delete', { message_id: messageId });
    }
};
