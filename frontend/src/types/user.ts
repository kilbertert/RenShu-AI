// 用户角色枚举
export enum UserRole {
  PUBLIC = 'PUBLIC',
  PROFESSIONAL = 'PROFESSIONAL',
  ADMIN = 'ADMIN'
}


export interface BaseProfile {
  constitution_type?: string;// 体质类型
  taboo_items?: string[];// 禁忌项
  medical_history?: string;// 病史
  family_history?: string;// 家族病史
  allergy_info?: string;// 过敏信息
  merged_diseases?: string;// 合并症
}
// 用户画像
export interface UserPersona {
  age: string;// 年龄
  gender: string;// 性别
  health_score?: number;// 健康评分
  chief_complaint: string;// 主诉
  suspected_diagnosis: string;// 疑似诊断
  recommended_treatment: string; // 推荐治疗
  base_profile?: BaseProfile ; // 基础健康画像
}

// 用户信息
export interface User {
  id: string;
  name: string;
  role: UserRole;
  avatar: string;
  base_profile?: BaseProfile ;// 用户画像
}
