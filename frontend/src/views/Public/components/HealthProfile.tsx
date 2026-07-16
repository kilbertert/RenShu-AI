import { useEffect, useState } from 'react';
import { ArrowLeft, FileText, Loader2 } from 'lucide-react';
import { caseApi, type CaseSummary, type HealthProfile as HealthProfileData } from '../../../api/modules/case';

interface Props {
    onBack: () => void;
}

const COMPLEXITY_COLOR: Record<string, string> = {
    simple: 'bg-green-100 text-green-700',
    moderate: 'bg-amber-100 text-amber-700',
    complex: 'bg-rose-100 text-rose-700',
};

const COMPLEXITY_LABEL: Record<string, string> = {
    simple: '简单',
    moderate: '中等',
    complex: '复杂',
};

function formatTime(iso: string | null): string {
    if (!iso) return '-';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('zh-CN', { hour12: false });
}

export default function HealthProfile({ onBack }: Props) {
    const [profile, setProfile] = useState<HealthProfileData | null>(null);
    const [cases, setCases] = useState<CaseSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            setLoading(true);
            setError(null);
            try {
                const [p, c] = await Promise.all([caseApi.profile(), caseApi.list(50, 0)]);
                if (cancelled) return;
                setProfile(p.data ?? null);
                setCases(c.data?.items ?? []);
            } catch (e) {
                if (!cancelled) setError(e instanceof Error ? e.message : '加载失败');
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => {
            cancelled = true;
        };
    }, []);

    if (loading) {
        return (
            <div className="flex h-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-stone-400" />
            </div>
        );
    }

    return (
        <div className="flex h-full flex-col bg-stone-50">
            <header className="flex items-center gap-3 border-b border-stone-200 bg-white px-6 py-4">
                <button
                    onClick={onBack}
                    className="rounded-full p-2 transition-colors hover:bg-stone-100"
                    aria-label="返回"
                >
                    <ArrowLeft className="h-5 w-5 text-stone-600" />
                </button>
                <div>
                    <h1 className="text-lg font-semibold text-stone-800">健康档案</h1>
                    <p className="text-xs text-stone-500">跨会话问诊记录与体质聚合</p>
                </div>
            </header>

            {error && (
                <div className="mx-6 mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                    加载失败：{error}
                </div>
            )}

            <div className="flex-1 overflow-y-auto px-6 py-6">
                <section className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <StatCard label="累计问诊" value={profile?.total_cases ?? cases.length} />
                    <StatCard
                        label="最近问诊"
                        value={profile?.last_case_at ? formatTime(profile.last_case_at) : '暂无'}
                    />
                    <StatCard
                        label="高频证型"
                        value={profile?.most_common_syndrome ?? '暂无'}
                    />
                </section>

                {profile && (profile.chronic_conditions?.length || profile.allergies?.length) && (
                    <section className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
                        {profile.chronic_conditions?.length ? (
                            <TagCard title="慢病标签" tags={profile.chronic_conditions} />
                        ) : null}
                        {profile.allergies?.length ? (
                            <TagCard title="过敏标签" tags={profile.allergies} />
                        ) : null}
                    </section>
                )}

                <section>
                    <h2 className="mb-3 text-sm font-semibold text-stone-700">历次问诊</h2>
                    {cases.length === 0 ? (
                        <div className="rounded-lg border border-dashed border-stone-300 bg-white px-6 py-12 text-center text-sm text-stone-400">
                            <FileText className="mx-auto mb-2 h-8 w-8 text-stone-300" />
                            暂无问诊记录
                        </div>
                    ) : (
                        <ol className="space-y-3">
                            {cases.map((c) => (
                                <li
                                    key={c.id}
                                    className="rounded-lg border border-stone-200 bg-white p-4 shadow-sm transition-shadow hover:shadow"
                                >
                                    <div className="mb-2 flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            {c.complexity_level && (
                                                <span
                                                    className={`rounded-full px-2 py-0.5 text-xs ${COMPLEXITY_COLOR[c.complexity_level]}`}
                                                >
                                                    {COMPLEXITY_LABEL[c.complexity_level]}
                                                </span>
                                            )}
                                            {c.syndrome_name && (
                                                <span className="text-sm font-medium text-stone-800">
                                                    {c.syndrome_name}
                                                </span>
                                            )}
                                        </div>
                                        <span className="text-xs text-stone-400">
                                            {formatTime(c.created_at)}
                                        </span>
                                    </div>
                                    <p className="line-clamp-2 text-sm text-stone-600">
                                        主诉：{c.chief_complaint}
                                    </p>
                                </li>
                            ))}
                        </ol>
                    )}
                </section>
            </div>
        </div>
    );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
    return (
        <div className="rounded-lg border border-stone-200 bg-white p-4">
            <div className="text-xs text-stone-500">{label}</div>
            <div className="mt-1 truncate text-lg font-semibold text-stone-800">
                {value}
            </div>
        </div>
    );
}

function TagCard({ title, tags }: { title: string; tags: string[] }) {
    return (
        <div className="rounded-lg border border-stone-200 bg-white p-4">
            <div className="mb-2 text-xs text-stone-500">{title}</div>
            <div className="flex flex-wrap gap-1.5">
                {tags.map((t) => (
                    <span
                        key={t}
                        className="rounded-full bg-stone-100 px-2.5 py-0.5 text-xs text-stone-700"
                    >
                        {t}
                    </span>
                ))}
            </div>
        </div>
    );
}
