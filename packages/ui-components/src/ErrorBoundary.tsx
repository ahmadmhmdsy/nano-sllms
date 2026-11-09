import { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Extra class for the fallback container. */
  className?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/** Shared error boundary (extracted from flow/) that catches React errors
 * and shows a fallback UI, preventing the entire app from crashing when
 * a component throws. Standard class component: getDerivedStateFromError
 * flips the state, componentDidCatch logs for debugging. The fallback
 * carries role="alert", the error details in a <details>, and a Reload
 * button; its colors come from the --bg / --ink / --line CSS variables
 * (viz defines them; the fallbacks reproduce flow's light palette). */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Log the error to the console for debugging
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div
          className={"error-boundary" + (this.props.className ? " " + this.props.className : "")}
          role="alert"
          style={{
            background: "var(--bg, #fff)",
            color: "var(--ink, #333)",
            border: "1px solid var(--line, #ccc)",
            borderRadius: 6,
            padding: "16px",
          }}
        >
          <h2>Something went wrong</h2>
          <p>The application encountered an unexpected error.</p>
          <details>
            <summary>Error details</summary>
            <pre>{this.state.error?.message}</pre>
            <pre>{this.state.error?.stack}</pre>
          </details>
          <button onClick={this.handleReload}>Reload</button>
        </div>
      );
    }

    return this.props.children;
  }
}
