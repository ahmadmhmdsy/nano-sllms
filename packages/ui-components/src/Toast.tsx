/** Shared presentational toast (extracted from flow's inline
 * store.error toast, FlowCanvas.tsx). The parent owns visibility and
 * positioning (flow renders it absolutely positioned over the canvas via
 * the .toast class); this component only paints the message. Renders
 * nothing for an empty message so callers can mount it unconditionally.
 * role="alert" by default (errors demand attention); pass role="status"
 * for passive notifications. Colors come from the --out / --on-out CSS
 * variables (viz defines --out; the fallbacks reproduce flow's toast). */
export interface ToastProps {
  message: string;
  className?: string;
  role?: "alert" | "status";
}

export default function Toast({ message, className, role = "alert" }: ToastProps) {
  if (message === "") return null;
  return (
    <div
      className={"toast" + (className ? " " + className : "")}
      role={role}
      style={{
        background: "var(--out, #c62828)",
        color: "var(--on-out, #fff)",
      }}
    >
      {message}
    </div>
  );
}
