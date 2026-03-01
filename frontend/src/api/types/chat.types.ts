
export interface ModelConfiguration {
    provider_id: string;
    model_id: string;
    model_name: string;
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
}

export interface ChatRequest {
    user_id: string;
    conversation_id: string;
    query: string;
    model_configuration: ModelConfiguration;
    stream?: boolean;
    enable_thinking?: boolean;  // 是否启用思考过程展示
}

export interface PersonaAnalysisRequest {
    user_id: string;
    text: string;
    current_persona?: Record<string, any>;
    conversation_id?: string; // Optional for backward compatibility, but recommended for persistence
    model_configuration: ModelConfiguration;
}

export interface ChatResumeRequest {
    conversation_id: string;
    thread_id: string;
    query: string;
    model_configuration: ModelConfiguration;
}