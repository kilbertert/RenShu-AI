// RenShu-AI 端到端验收用最小中医知识图谱。
// 使用 MERGE 保证脚本可重复执行。

MERGE (sy:Syndrome {id: 'E2E-SY-001'})
SET sy.name_zh = '脾气虚证',
    sy.name_en = 'Spleen Qi Deficiency',
    sy.definition = '长期乏力、食欲不振、腹胀便溏、睡眠不佳，可伴气短与面色少华',
    sy.source_db = 'renshu_e2e';

MERGE (sy2:Syndrome {id: 'E2E-SY-002'})
SET sy2.name_zh = '心脾两虚证',
    sy2.name_en = 'Heart and Spleen Deficiency',
    sy2.definition = '乏力、心悸、气短、头晕、多梦易醒，常见于心血不足兼脾气虚弱',
    sy2.source_db = 'renshu_e2e';

MERGE (ts1:TCMSymptom {id: 'E2E-TS-001'})
SET ts1.name_zh = '乏力', ts1.name_en = 'fatigue', ts1.source_db = 'renshu_e2e';
MERGE (ts2:TCMSymptom {id: 'E2E-TS-002'})
SET ts2.name_zh = '食欲不振', ts2.name_en = 'poor appetite', ts2.source_db = 'renshu_e2e';
MERGE (ts3:TCMSymptom {id: 'E2E-TS-003'})
SET ts3.name_zh = '腹胀', ts3.name_en = 'abdominal distension', ts3.source_db = 'renshu_e2e';

MERGE (f:Formula {id: 'E2E-F-001'})
SET f.name_zh = '四君子汤',
    f.effect_zh = '益气健脾，改善乏力、食欲不振与气短',
    f.indications_zh = '脾胃气虚所致面色萎白、语声低微、腹胀便溏',
    f.source = '《太平惠民和剂局方》',
    f.source_db = 'renshu_e2e';

MERGE (f2:Formula {id: 'E2E-F-002'})
SET f2.name_zh = '参苓白术散',
    f2.effect_zh = '益气健脾，渗湿止泻，改善乏力、食少与便溏',
    f2.indications_zh = '脾胃气虚夹湿所致食欲不振、腹胀、便溏',
    f2.source = '《太平惠民和剂局方》',
    f2.source_db = 'renshu_e2e';

MERGE (f3:Formula {id: 'E2E-F-003'})
SET f3.name_zh = '归脾汤',
    f3.effect_zh = '益气补血，健脾养心，改善乏力、心悸、气短与多梦',
    f3.indications_zh = '心脾气血两虚所致头晕、心悸、多梦易醒、体倦食少',
    f3.source = '《正体类要》',
    f3.source_db = 'renshu_e2e';

MATCH (seeded_syndrome:Syndrome {id: 'E2E-SY-001'}),
      (seeded_formula:Formula {id: 'E2E-F-001'})
RETURN seeded_syndrome.name_zh AS syndrome,
       seeded_formula.name_zh AS formula;
