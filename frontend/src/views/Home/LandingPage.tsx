import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Icons } from '../../components/common/Icons';
import { BrandLogo } from '../../components/common/BrandLogo';

type EntryRole = 'public' | 'professional';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeRole, setActiveRole] = useState<EntryRole | null>(null);

  const enterPortal = (role: EntryRole) => {
    navigate(`/login/${role}`);
  };

  return (
    <main className="landing-page" aria-labelledby="landing-title">
      <div className="landing-page__texture" aria-hidden="true" />
      <div className="landing-page__rings" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>

      <header className="landing-header">
        <div className="landing-header__eyebrow">
          <span className="landing-header__dot" aria-hidden="true" />
          <span>REN SHU AI MEDICAL</span>
        </div>
        <div className="landing-header__symbol" aria-hidden="true">☯</div>
        <span className="landing-header__edition">传承 · 创新 · 仁心</span>
      </header>

      <section className="landing-stage">
        <div className="landing-stage__copy">
          <p className="landing-stage__label">中医智慧 · 人工智能</p>
          <h1 id="landing-title">以阴阳之道，<br />观身心之衡</h1>
          <p className="landing-stage__description">
            让传统辨证与现代智能相遇，<br className="hidden sm:block" />
            从一次真实的倾听开始。
          </p>
        </div>

        <div className={`landing-orbit ${activeRole ? `is-${activeRole}` : ''}`} aria-hidden="true">
          <div className="landing-orbit__halo" />
          <div className="landing-orbit__ring" />
          <div className="landing-yinyang">
            <span className="landing-yinyang__dot landing-yinyang__dot--dark" />
            <span className="landing-yinyang__dot landing-yinyang__dot--light" />
          </div>
          <div className="landing-orbit__spark landing-orbit__spark--top" />
          <div className="landing-orbit__spark landing-orbit__spark--bottom" />
        </div>

        <div className="landing-brand-lockup">
          <BrandLogo size="md" variant="dark" showText />
          <span className="landing-brand-lockup__line" />
          <span className="landing-brand-lockup__caption">INTELLIGENT DIAGNOSIS</span>
        </div>
      </section>

      <section className="landing-entries" aria-label="选择入口">
        <button
          type="button"
          className={`landing-entry landing-entry--patient ${activeRole === 'public' ? 'is-active' : ''}`}
          onMouseEnter={() => setActiveRole('public')}
          onMouseLeave={() => setActiveRole(null)}
          onFocus={() => setActiveRole('public')}
          onBlur={() => setActiveRole(null)}
          onClick={() => enterPortal('public')}
        >
          <span className="landing-entry__icon" aria-hidden="true"><Icons.UserCircle2 size={24} /></span>
          <span className="landing-entry__body">
            <span className="landing-entry__title">患者入口</span>
            <span className="landing-entry__subtitle">个人健康与中医问诊</span>
          </span>
          <span className="landing-entry__action">进入 <Icons.ChevronRight size={16} /></span>
        </button>

        <button
          type="button"
          className={`landing-entry landing-entry--professional ${activeRole === 'professional' ? 'is-active' : ''}`}
          onMouseEnter={() => setActiveRole('professional')}
          onMouseLeave={() => setActiveRole(null)}
          onFocus={() => setActiveRole('professional')}
          onBlur={() => setActiveRole(null)}
          onClick={() => enterPortal('professional')}
        >
          <span className="landing-entry__icon" aria-hidden="true"><Icons.Stethoscope size={24} /></span>
          <span className="landing-entry__body">
            <span className="landing-entry__title">医生与研究入口</span>
            <span className="landing-entry__subtitle">临床决策支持与知识图谱</span>
          </span>
          <span className="landing-entry__action">进入 <Icons.ChevronRight size={16} /></span>
        </button>
      </section>

      <footer className="landing-footer">
        <span>仁术 AI</span>
        <span className="landing-footer__separator" aria-hidden="true" />
        <span>知常达变 · 和而不同</span>
      </footer>
    </main>
  );
};

export default LandingPage;
