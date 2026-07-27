# RenShu AI 正式服务部署

服务以 `claude` 用户的 user systemd units 运行，不使用 root supervisor。仓库中的 unit 是可审计模板，安装后由 `systemctl --user` 管理。

## 前置条件

- `/home/claude/Projects/RenShu-AI/.env` 已配置，权限为 `600`
- PostgreSQL、Redis、Neo4j、Qdrant 容器已启动并健康
- 后端依赖已安装在 `backend/.venv`
- 前端已执行 `npm ci` 和 `npm run build`
- `8091` 由其他系统使用；RenShu 后端使用 `8094`，前端使用 `3002`
- 生产前端默认通过 `3002/api` 同源代理访问后端，浏览器不需要直连 `8094`
- `.env` 中的 `JWT_SECRET_KEY`、`ENCRYPTION_KEY`、数据库密码和模型密钥必须使用部署环境自己的值；这些值不进入 Git
- 轮换 `ENCRYPTION_KEY` 时必须先迁移 `user_provider_configs.api_key` 的 Fernet 密文，再重启后端

## 安装和启动

```bash
mkdir -p /home/claude/.config/systemd/user
install -m 0644 deploy/systemd/renshu-backend.service /home/claude/.config/systemd/user/
install -m 0644 deploy/systemd/renshu-frontend.service /home/claude/.config/systemd/user/
install -m 0644 deploy/systemd/renshu-celery.service /home/claude/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now renshu-backend.service renshu-frontend.service renshu-celery.service
```

## 验证

```bash
systemctl --user status renshu-backend.service renshu-frontend.service renshu-celery.service
curl --fail http://127.0.0.1:8094/health
curl --fail http://127.0.0.1:3002/
```

部署单元继承当前用户的 service manager；更新代码后重新构建前端，并执行 `systemctl --user restart renshu-backend.service renshu-frontend.service renshu-celery.service`。回滚时恢复上一个 Git 提交、重新构建前端，再重启同样的三个单元即可。
