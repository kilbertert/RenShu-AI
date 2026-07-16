import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: (error: Error, reset: () => void) => React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * 顶层 Error Boundary，捕获 React 19 已知 text node 调和 bug（"removeChild on Node" / "<Text> component"）
 * 以及任何子组件渲染时抛出的同步错误，防止整页变白。
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[ErrorBoundary] 捕获到渲染错误:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.handleReset);
      }
      return (
        <div className="h-screen w-full flex items-center justify-center bg-tcm-cream p-6">
          <div className="max-w-md w-full bg-white dark:bg-tcm-charcoal rounded-2xl shadow-2xl p-8 text-center border border-tcm-lightGreen/20">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-50 dark:bg-red-900/20 flex items-center justify-center">
              <span className="text-3xl">⚠️</span>
            </div>
            <h2 className="text-xl font-bold text-tcm-darkGreen dark:text-tcm-cream mb-2 font-serif-sc">
              页面渲染出错
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              组件遇到一个意外错误。常见原因是 HMR 重载或路由懒加载切换时的临时问题，刷新即可恢复。
            </p>
            <details className="text-left text-xs text-gray-400 dark:text-gray-500 bg-gray-50 dark:bg-black/20 rounded-lg p-3 mb-4 max-h-32 overflow-auto">
              <summary className="cursor-pointer font-mono">错误详情</summary>
              <pre className="mt-2 whitespace-pre-wrap break-all font-mono">{this.state.error.message}</pre>
            </details>
            <div className="flex gap-2 justify-center">
              <button
                onClick={this.handleReset}
                className="px-5 py-2 text-sm bg-tcm-lightGreen/20 text-tcm-darkGreen dark:text-tcm-lightGreen rounded-lg hover:bg-tcm-lightGreen/30 transition-colors"
              >
                重试
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-5 py-2 text-sm bg-tcm-darkGreen text-white rounded-lg hover:bg-tcm-lightGreen transition-colors"
              >
                刷新页面
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
