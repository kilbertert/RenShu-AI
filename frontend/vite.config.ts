import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const apiTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8094';
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        // Dev only: 允许任意 Host 头，便于 ngrok / localhost.run / Cloudflared 等内网穿透
        // 生产环境不要这样写
        allowedHosts: true,
        proxy: {
          '/api': {
            target: apiTarget,
            changeOrigin: true,
          },
        },
      },
      preview: {
        host: '0.0.0.0',
        port: 3002,
        proxy: {
          '/api': {
            target: apiTarget,
            changeOrigin: true,
            ws: true,
          },
        },
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
