# 诊断子图架构设计

> 版本: v1.0
> 日期: 2024-02
> 状态: 设计阶段

## 1. 概述

诊断子图（Diagnose Subgraph）负责处理用户的病情问诊请求，通过多轮追问收集信息，结合多模态输入（舌像、检验报告），根据病情复杂度选择不同的辨证策略。

### 1.1 设计目标

- **信息完整性**：通过多轮追问确保收集到足够的辨证信息
- **多模态融合**：支持舌像分析、检验报告解读等多模态输入
- **分级处理**：根据病情复杂度选择合适的辨证策略
- **可追溯性**：记录完整的问诊过程和推理链路

### 1.2 核心流程

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     诊断子图 (Diagnose Subgraph)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              信息收集循环 (Collection Loop)          │   │
│  │                                                     │   │
│  │   [collect_info] ←──────────────────────┐          │   │
│  │        │                                │          │   │
│  │        ▼                                │          │   │
│  │   [analyze_and_follow_up]               │          │   │
│  │        │                                │          │   │
│  │   ┌────┴────┬─────────┬────────┐       │          │   │
│  │   ▼         ▼         ▼        ▼       │          │   │
│  │ [问症状] [请求舌像] [请求报告] [完成] ──┘          │   │
│  │   │         │         │                           │   │
│  │   │    [tongue_analysis]                          │   │
│  │   │         │    [report_analysis]                │   │
│  │   └─────────┴─────────┘                           │   │
│  │              │                                     │   │
│  └──────────────┼─────────────────────────────────────┘   │
│                 ▼                                         │
│        [assess_complexity] ─── 复杂度评估                  │
│                 │                                         │
│      ┌──────────┼──────────┐                             │
│      ▼          ▼          ▼                             │
│  [simple]   [moderate]  [complex]                        │
│      │          │          │                             │
│      ▼          ▼          ▼                             │
│  [LLM直接   [RAG +      [DeepSearch                     │
│   辨证]    预定义Cypher]  Agent]                         │
│      │          │          │                             │
│      └──────────┴──────────┘                             │
│                 │                                         │
│                 ▼                                         │
│                END                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 状态定义

### 2.1 输入状态 (DiagnoseInputState)

```python
class DiagnoseInputState(TypedDict):
    """诊断子图输入状态"""
    query: str                          # 用户当前输入
    messages: List[BaseMessage]         # 对话历史
    user_profile: Dict[str, Any]        # 用户画像（体质、既往史等）
    llm_config: Optional[LLMConfig]     # LLM 配置
    extracted_entities: Dict[str, Any]  # 意图识别提取的实体
```

### 2.2 内部状态 (DiagnoseOverallState)

```python
class DiagnoseOverallState(TypedDict):
    """诊断子图内部状态"""
    # === 输入继承 ===
    query: str
    messages: List[BaseMessage]
    user_profile: Dict[str, Any]
    llm_config: Optional[LLMConfig]

    # === 信息收集 ===
    collected_info: CollectedDiagnoseInfo      # 已收集的诊断信息
    collection_history: Annotated[List[CollectionRecord], add]  # 收集历史
    follow_up_count: int                       # 追问轮数

    # === 多模态 ===
    tongue_analysis: Optional[TongueAnalysisResult]   # 舌像分析结果
    report_analysis: Optional[ReportAnalysisResult]   # 报告解读结果

    # === 复杂度评估 ===
    complexity: Optional[ComplexityAssessment]  # 复杂度评估结果

    # === 辨证结果 ===
    diagnosis_result: Optional[DiagnosisResult] # 辨证结果

    # === 流程控制 ===
    next_action: str                           # 路由信号
    steps: Annotated[List[str], add]           # 执行步骤记录
```

### 2.3 输出状态 (DiagnoseOutputState)

```python
class DiagnoseOutputState(TypedDict):
    """诊断子图输出状态"""
    answer: str                               # 回复内容
    diagnosis_result: Optional[DiagnosisResult]  # 辨证结果
    steps: List[str]                          # 执行步骤
    follow_up_question: Optional[str]         # 追问问题（如果需要）
```

---

## 3. 核心数据模型

### 3.1 已收集信息 (CollectedDiagnoseInfo)

```python
class CollectedDiagnoseInfo(BaseModel):
    """已收集的诊断信息 - 基于中医十问"""

    # === 主诉 ===
    chief_complaint: Optional[str] = None       # 主诉
    onset_time: Optional[str] = None            # 发病时间
    duration: Optional[str] = None              # 病程

    # === 十问信息 ===
    cold_heat: Optional[str] = None             # 寒热：恶寒、发热、寒热往来
    sweat: Optional[str] = None                 # 汗出：有汗、无汗、盗汗、自汗
    head_body: Optional[str] = None             # 头身：头痛、头晕、身痛、乏力
    urine_stool: Optional[str] = None           # 二便：大便、小便情况
    diet: Optional[str] = None                  # 饮食：食欲、口渴、口苦、口淡
    sleep: Optional[str] = None                 # 睡眠：失眠、多梦、嗜睡
    emotion: Optional[str] = None               # 情志：烦躁、抑郁、焦虑

    # === 望诊（多模态）===
    tongue: Optional[Dict[str, str]] = None     # 舌象：舌色、舌形、苔色、苔质
    complexion: Optional[str] = None            # 面色

    # === 既往史 ===
    medical_history: Optional[List[str]] = None  # 既往病史
    current_medications: Optional[List[str]] = None  # 当前用药
    allergies: Optional[List[str]] = None       # 过敏史

    # === 女性专属 ===
    menstruation: Optional[str] = None          # 月经情况（女性）

    def get_missing_categories(self) -> List[str]:
        """获取缺失的必要信息类别"""
        required = {
            "cold_heat": "寒热",
            "sweat": "汗出",
            "head_body": "头身",
            "urine_stool": "二便",
            "diet": "饮食",
            "sleep": "睡眠",
        }
        missing = []
        for field, name in required.items():
            if getattr(self, field) is None:
                missing.append(name)
        return missing

    def is_sufficient(self, min_categories: int = 4) -> bool:
        """判断信息是否足够（至少收集到 N 类信息）"""
        missing = self.get_missing_categories()
        return len(missing) <= (6 - min_categories)
```

### 3.2 复杂度评估 (ComplexityAssessment)

```python
class ComplexityLevel(str, Enum):
    SIMPLE = "simple"       # 简单：LLM 直接辨证
    MODERATE = "moderate"   # 中等：RAG + 预定义 Cypher
    COMPLEX = "complex"     # 复杂：DeepSearch Agent

class ComplexityAssessment(BaseModel):
    """复杂度评估结果"""
    level: ComplexityLevel
    score: int                          # 0-10 分
    factors: Dict[str, int]             # 各因素得分
    reasoning: str                      # 评估理由

    # === 评估因素 ===
    # symptom_count: 症状数量 (1-3: 0分, 4-5: 1分, >5: 2分)
    # organ_systems: 涉及脏腑 (1: 0分, 2: 1分, >2: 2分)
    # duration: 病程 (<2周: 0分, 2周-3月: 1分, >3月: 2分)
    # contradiction: 症状矛盾 (无: 0分, 有: 2分)
    # chronic_conditions: 既往慢性病 (0: 0分, 1-2: 1分, >2: 2分)
    # tongue_abnormal: 舌象异常程度 (轻: 0分, 中: 1分, 重: 2分)
```

### 3.3 辨证结果 (DiagnosisResult)

```python
class DiagnosisResult(BaseModel):
    """辨证结果"""
    # === 八纲辨证 ===
    ba_gang: Dict[str, str] = Field(default_factory=dict)
    # 示例: {"阴阳": "阳证", "表里": "表证", "寒热": "热证", "虚实": "实证"}

    # === 证型 ===
    syndrome: str                       # 主要证型
    syndrome_secondary: Optional[List[str]] = None  # 兼证

    # === 病因病机 ===
    etiology: Optional[str] = None      # 病因
    pathogenesis: Optional[str] = None  # 病机

    # === 治则治法 ===
    treatment_principle: Optional[str] = None  # 治则
    treatment_method: Optional[str] = None     # 治法

    # === 建议 ===
    recommendations: Optional[List[str]] = None  # 调理建议
    warnings: Optional[List[str]] = None         # 注意事项
    should_seek_doctor: bool = False             # 是否建议就医

    # === 置信度 ===
    confidence: float = 0.0            # 辨证置信度 0-1
    reasoning_chain: List[str] = Field(default_factory=list)  # 推理链

    # === 参考来源 ===
    references: Optional[List[Dict[str, Any]]] = None  # 参考医案/文献
```

---

## 4. 节点定义

### 4.1 信息收集节点 (collect_info)

```python
async def collect_info(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    收集用户输入的信息，更新 collected_info

    功能：
    1. 解析用户最新输入
    2. 提取症状、时间、程度等信息
    3. 映射到 CollectedDiagnoseInfo 的相应字段
    4. 检测是否有图片（舌像）或文件（报告）
    """
```

### 4.2 分析与追问节点 (analyze_and_follow_up)

```python
async def analyze_and_follow_up(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    分析已收集信息，决定下一步行动

    输出 next_action:
    - "ask_symptom": 追问症状
    - "request_tongue": 请求上传舌像
    - "request_report": 请求上传检验报告
    - "assess_complexity": 信息足够，进入复杂度评估
    - "intent_switch": 检测到意图切换，退出子图

    判断逻辑：
    1. 检查必要信息是否收集完整
    2. 如果涉及脾胃/湿热/气血问题且无舌像 → 建议上传
    3. 如果涉及慢性病且无报告 → 建议上传
    4. 如果追问轮数 > MAX_ROUNDS → 强制进入评估
    5. 检测用户是否切换了意图
    """
```

### 4.3 舌像分析节点 (tongue_analysis)

```python
async def tongue_analysis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    分析用户上传的舌像

    输出：
    - tongue_analysis: TongueAnalysisResult
    - collected_info.tongue: 更新舌象信息

    分析内容：
    - 舌色：淡白、淡红、红、绛红、紫暗
    - 舌形：胖大、瘦薄、齿痕、裂纹
    - 苔色：白、黄、灰黑
    - 苔质：薄、厚、腻、燥、剥
    """
```

### 4.4 报告解读节点 (report_analysis)

```python
async def report_analysis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    解读用户上传的检验报告

    支持类型：
    - 血常规
    - 肝肾功能
    - 血糖血脂
    - 甲状腺功能
    - 其他常规检查

    输出：
    - report_analysis: ReportAnalysisResult
    - 关键异常指标及其中医意义
    """
```

### 4.5 复杂度评估节点 (assess_complexity)

```python
async def assess_complexity(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    评估病情复杂度，决定辨证策略

    评估因素（总分 10 分）：
    ┌────────────────┬─────────────────────────────────┬───────┐
    │ 因素           │ 评分标准                         │ 分值  │
    ├────────────────┼─────────────────────────────────┼───────┤
    │ 症状数量       │ 1-3个:0 / 4-5个:1 / >5个:2       │ 0-2   │
    │ 涉及脏腑       │ 1个:0 / 2个:1 / >2个:2           │ 0-2   │
    │ 病程           │ <2周:0 / 2周-3月:1 / >3月:2      │ 0-2   │
    │ 症状矛盾       │ 无:0 / 有:2                      │ 0-2   │
    │ 既往慢性病     │ 0个:0 / 1-2个:1 / >2个:2         │ 0-2   │
    └────────────────┴─────────────────────────────────┴───────┘

    复杂度分级：
    - 0-3 分: SIMPLE   → 简单辨证
    - 4-6 分: MODERATE → RAG 辅助
    - 7-10分: COMPLEX  → DeepSearch

    输出：
    - complexity: ComplexityAssessment
    - next_action: "simple" | "moderate" | "complex"
    """
```

### 4.6 简单辨证节点 (simple_diagnosis)

```python
async def simple_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    简单病情的直接辨证

    方法：
    - LLM 根据收集的信息直接进行八纲辨证
    - 结合望闻问切四诊信息
    - 给出证型、治则、建议

    适用场景：
    - 单一证型（如普通感冒）
    - 症状明确，指向清晰
    - 无复杂既往史

    Prompt 要点：
    - 系统提示包含八纲辨证框架
    - 输入已收集的四诊信息
    - 要求结构化输出
    """
```

### 4.7 中等辨证节点 (moderate_diagnosis)

```python
async def moderate_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    中等复杂度的 RAG 辅助辨证

    方法：
    1. 根据症状提取关键词
    2. 使用预定义 Cypher 查询知识图谱
       - 查询相似证型
       - 查询相关医案
       - 查询常用方剂
    3. 将检索结果作为上下文
    4. LLM 结合上下文生成辨证结果

    预定义 Cypher 示例：
    - match_syndrome_by_symptoms: 根据症状匹配证型
    - find_similar_cases: 查找相似医案
    - get_treatment_by_syndrome: 根据证型查询治法

    适用场景：
    - 多个症状需要综合判断
    - 可能存在兼证
    - 需要医案参考
    """
```

### 4.8 复杂辨证节点 (complex_diagnosis / DeepSearch)

```python
async def complex_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    复杂病情的 DeepSearch Agent 辨证

    方法：
    1. 任务分解：将复杂问题拆分为子问题
    2. 多源检索：
       - 知识图谱（证型、方剂、药材关系）
       - 向量数据库（相似医案）
       - 中医文献库（经典论述）
    3. 推理链构建：
       - 分析主症与兼症
       - 判断病因病机
       - 鉴别诊断
    4. 综合判断：
       - 多轮迭代优化
       - 交叉验证结果

    DeepSearch Agent 架构：
    ┌─────────────────────────────────────────┐
    │           DeepSearch Agent              │
    ├─────────────────────────────────────────┤
    │                                         │
    │  [Planner] ─→ 任务分解                  │
    │      │                                  │
    │      ▼                                  │
    │  [Multi-Tool Router]                    │
    │      │                                  │
    │  ┌───┴───┬───────┬───────┐             │
    │  ▼       ▼       ▼       ▼             │
    │ [KG]  [Vector] [Text] [Web]            │
    │  │       │       │       │             │
    │  └───────┴───────┴───────┘             │
    │           │                             │
    │           ▼                             │
    │     [Synthesizer] ─→ 综合分析           │
    │           │                             │
    │           ▼                             │
    │     [Validator] ─→ 验证与迭代           │
    │                                         │
    └─────────────────────────────────────────┘

    适用场景：
    - 虚实夹杂、寒热错杂
    - 多脏腑联动
    - 久病复杂
    - 罕见病症
    """
```

---

## 5. 路由逻辑

### 5.1 追问循环路由 (route_collection)

```python
def route_collection(state: DiagnoseOverallState) -> str:
    """信息收集阶段的路由"""
    next_action = state.get("next_action", "")

    match next_action:
        case "ask_symptom":
            return "collect_info"  # 继续收集
        case "request_tongue":
            return "wait_for_tongue"  # 等待舌像
        case "request_report":
            return "wait_for_report"  # 等待报告
        case "assess_complexity":
            return "assess_complexity"  # 进入评估
        case "intent_switch":
            return END  # 退出子图
        case _:
            return "assess_complexity"
```

### 5.2 复杂度路由 (route_by_complexity)

```python
def route_by_complexity(state: DiagnoseOverallState) -> str:
    """根据复杂度路由到不同的辨证节点"""
    complexity = state.get("complexity")

    if not complexity:
        return "simple_diagnosis"

    match complexity.level:
        case ComplexityLevel.SIMPLE:
            return "simple_diagnosis"
        case ComplexityLevel.MODERATE:
            return "moderate_diagnosis"
        case ComplexityLevel.COMPLEX:
            return "complex_diagnosis"
        case _:
            return "simple_diagnosis"
```

---

## 6. 配置参数

```python
class DiagnoseConfig:
    """诊断子图配置"""

    # === 追问控制 ===
    MAX_FOLLOW_UP_ROUNDS: int = 5          # 最大追问轮数
    MIN_REQUIRED_CATEGORIES: int = 4       # 最少需要的信息类别数

    # === 复杂度阈值 ===
    SIMPLE_THRESHOLD: int = 3              # 简单阈值（≤3分）
    MODERATE_THRESHOLD: int = 6            # 中等阈值（≤6分）
    # > 6分为复杂

    # === 多模态 ===
    ENABLE_TONGUE_REQUEST: bool = True     # 是否启用舌像请求
    ENABLE_REPORT_REQUEST: bool = True     # 是否启用报告请求
    TONGUE_REQUEST_PROBABILITY: float = 0.7  # 请求舌像的概率（满足条件时）

    # === LLM 参数 ===
    COLLECTION_TEMPERATURE: float = 0.7    # 信息收集阶段温度
    DIAGNOSIS_TEMPERATURE: float = 0.3     # 辨证阶段温度（更确定性）

    # === DeepSearch ===
    DEEPSEARCH_MAX_ITERATIONS: int = 3     # DeepSearch 最大迭代次数
    DEEPSEARCH_SOURCES: List[str] = [      # DeepSearch 数据源
        "knowledge_graph",
        "vector_store",
        "tcm_classics"
    ]
```

---

## 7. 错误处理

### 7.1 降级策略

```python
class DiagnoseFallbackStrategy:
    """降级策略"""

    # 场景1: RAG 检索失败
    # → 降级到 simple_diagnosis

    # 场景2: DeepSearch 超时
    # → 降级到 moderate_diagnosis

    # 场景3: 舌像分析失败
    # → 跳过舌像，继续问诊

    # 场景4: 所有辨证方法失败
    # → 返回通用健康建议 + 建议就医
```

### 7.2 异常响应

```python
DIAGNOSE_ERROR_RESPONSES = {
    "collection_failed": "抱歉，我没有完全理解您的描述。能否再详细说明一下您的症状？",
    "tongue_analysis_failed": "舌像分析暂时出现问题，我们继续通过问诊了解您的情况。",
    "diagnosis_failed": "抱歉，根据目前的信息，我无法给出确切的判断。建议您前往医院进行详细检查。",
    "timeout": "分析时间较长，请稍候。如果长时间没有响应，请刷新重试。",
}
```

---

## 8. 文件结构

```
backend/app/src/agent/components/diagnose/
├── __init__.py
├── ARCHITECTURE.md          # 本文档
├── states.py                # 状态定义
├── models.py                # 数据模型
├── config.py                # 配置参数
├── builder.py               # 子图构建器
├── router.py                # 路由逻辑
│
├── nodes/
│   ├── __init__.py
│   ├── collect_info.py      # 信息收集节点
│   ├── analyze_follow_up.py # 分析追问节点
│   ├── tongue_analysis.py   # 舌像分析节点
│   ├── report_analysis.py   # 报告解读节点
│   ├── assess_complexity.py # 复杂度评估节点
│   ├── simple_diagnosis.py  # 简单辨证节点
│   ├── moderate_diagnosis.py # 中等辨证节点
│   ├── complex_diagnosis/   # DeepSearch Agent
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── tools.py
│   │   ├── synthesizer.py
│   │   └── validator.py
│   └── generate_result.py   # 结果生成节点
│
├── prompts/
│   ├── __init__.py
│   ├── collection_prompts.py
│   ├── diagnosis_prompts.py
│   └── deepsearch_prompts.py
│
├── utils/
│   ├── __init__.py
│   ├── symptom_extractor.py  # 症状提取器
│   ├── complexity_scorer.py  # 复杂度评分器
│   └── cypher_templates.py   # 预定义 Cypher
│
└── tests/
    ├── __init__.py
    ├── test_collection.py
    ├── test_complexity.py
    └── test_diagnosis.py
```

---

## 9. 示例流程

### 9.1 简单病情示例

```
用户: "我感冒了，有点流鼻涕"

[collect_info] → 提取: 主诉=感冒, 症状=流鼻涕
[analyze_and_follow_up] → 信息不足，追问

AI: "请问您怕冷还是怕热？有没有发烧？出汗吗？"

用户: "怕冷，不发烧，没有汗"

[collect_info] → 更新: 寒热=恶寒, 汗出=无汗
[analyze_and_follow_up] → 继续追问

AI: "鼻涕是清的还是黄的？喉咙痛吗？"

用户: "清鼻涕，喉咙不痛"

[collect_info] → 更新: 补充症状
[analyze_and_follow_up] → 信息足够

[assess_complexity] → 评分: 2分 (SIMPLE)
  - 症状数量: 3个 (0分)
  - 涉及脏腑: 1个/肺 (0分)
  - 病程: <2周 (0分)
  - 无矛盾症状 (0分)
  - 无既往史 (0分)
  - 舌象: 未提供 (+2分待定)

[simple_diagnosis] → LLM 直接辨证生成结果
  - 证型: 风寒束表
  - 八纲: 表证、寒证、实证
  - 治法: 辛温解表
  - 建议: 生姜红糖水、注意保暖
  - 输出: Markdown 格式报告 → END
```

### 9.2 复杂病情示例

```
用户: "我头晕乏力好几个月了，有时候心慌，睡眠也不好"

[多轮信息收集...]

[assess_complexity] → 评分: 8分 (COMPLEX)
  - 症状数量: >5个 (2分)
  - 涉及脏腑: 心、脾、肝 (2分)
  - 病程: >3月 (2分)
  - 症状: 虚实不明 (2分)

[complex_diagnosis / DeepSearch] → 综合辨证生成结果
  ├── [Planner] 分解任务:
  │   - 分析头晕的可能证型
  │   - 分析乏力的脏腑归属
  │   - 分析心慌与失眠的关系
  │   - 综合判断虚实
  │
  ├── [Multi-Tool] 多源检索:
  │   - KG: 匹配到气血两虚、心脾两虚等证型
  │   - Vector: 找到3个相似医案
  │   - Classics: 《景岳全书》相关论述
  │
  ├── [Synthesizer] 综合分析:
  │   - 头晕 + 乏力 + 心慌 → 气血不足
  │   - 失眠 + 多梦 → 心血不足，心神失养
  │   - 综合: 心脾两虚证
  │
  └── [Validator] 验证:
      - 与医案对比验证
      - 确认证型合理性

→ 生成详细回复 + 建议就医 → END
```

---

## 10. 待定事项

- [ ] DeepSearch Agent 的具体实现细节
- [ ] 预定义 Cypher 模板的完整列表
- [ ] 舌像分析模型的选择（本地 vs API）
- [ ] 与主图的多模态消息传递机制
- [ ] 意图切换检测的具体实现
- [ ] 测试用例的设计

---

## 11. 参考

- Teacher 电商多轮追问架构 (`teacher/lg_agent/kg_sub_graph/`)
- 现有 wellness 子图实现 (`components/wellness/`)
- LangGraph 官方文档
- 中医十问歌
