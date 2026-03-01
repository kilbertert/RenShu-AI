import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

const container = document.getElementById('root');
const root = createRoot(container!);

// 🔧 暂时禁用 StrictMode 来排查重复请求问题
// 如果禁用后问题消失，说明问题与 StrictMode 的双重渲染有关
root.render(
  // <React.StrictMode>
    <App />
  // </React.StrictMode>
);
