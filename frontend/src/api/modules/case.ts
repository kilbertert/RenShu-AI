import request from '../request';
import type { ApiResponse } from '../types';

export interface CaseSummary {
    id: string;
    conversation_id: string;
    thread_id: string | null;
    chief_complaint: string;
    complexity_level: 'simple' | 'moderate' | 'complex' | null;
    syndrome_id: string | null;
    syndrome_name: string | null;
    syndrome_confidence: number | null;
    created_at: string | null;
}

export interface CaseDetail {
    case: CaseSummary;
    diagnosis_text: string | null;
    symptoms: string[];
    syndromes: Array<{ name: string; confidence: number | null; is_primary: boolean }>;
    prescriptions: Array<{
        name: string;
        composition: string | null;
        usage: string | null;
        source: string | null;
        rank: number;
    }>;
}

export interface HealthProfile {
    user_id: string;
    constitution: string | null;
    chronic_conditions: string[];
    allergies: string[];
    total_cases: number;
    last_case_at: string | null;
    most_common_syndrome: string | null;
    updated_at: string | null;
}

export const caseApi = {
    list: (limit = 50, offset = 0): Promise<ApiResponse<{ total: number; items: CaseSummary[] }>> =>
        request.get('/api/v1/cases', { params: { limit, offset } }),

    detail: (caseId: string): Promise<ApiResponse<CaseDetail | null>> =>
        request.get(`/api/v1/cases/${caseId}`),

    profile: (): Promise<ApiResponse<HealthProfile | null>> =>
        request.get('/api/v1/cases/profile'),
};
