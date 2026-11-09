import { useEffect } from "react";

/** Shared confirmation dialog for destructive actions (extracted from
 * flow/, Task M2.4; Clear canvas / Delete flow). Purely presentational:
 * the parent owns the open/closed state and the confirm effect. Renders
 * null when closed; Escape and backdrop click both dismiss via onDismiss,
 * matching the Save modal's dismissal semantics (the file modals dismiss
 * through the Toolbar's capture-phase Escape handler; this dialog uses
 * its own useEffect keydown listener, InferHandoffDialog-style). Reuses the
 * existing modal CSS classes (.modal-backdrop / .modal / .modal-actions);
 * the inline styles only carry the themed colors (CSS variables with
 * fallbacks that reproduce flow's light palette), so app stylesheets keep
 * owning layout. The className prop lands on the dialog box itself. */
export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  /** The destructive effect — run only after the user confirms. */
  onConfirm: () => void;
  /** Cancel / Escape / backdrop click. */
  onDismiss: () => void;
  /** Extra class for the dialog box (the .modal element). */
  className?: string;
}

export default function ConfirmDialog({ open, title, message, onConfirm, onDismiss, className }: ConfirmDialogProps) {
  /* Escape dismisses the dialog. Scoped to the open state, window level
   * so it fires regardless of focus target (same pattern as
   * InferHandoffDialog). */
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onDismiss]);

  if (!open) return null;
  return (
    <div
      className="modal-backdrop"
      onClick={onDismiss}
      style={{ background: "var(--backdrop, rgba(0, 0, 0, 0.15))" }}
    >
      <div
        className={"modal" + (className ? " " + className : "")}
        role="dialog"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--panel, #fff)",
          color: "var(--ink, #333)",
          borderColor: "var(--line, #999)",
        }}
      >
        <p className="modal-title">{title}</p>
        <p className="modal-note">{message}</p>
        <div className="modal-actions">
          <button onClick={onDismiss}>Cancel</button>
          <button onClick={onConfirm}>Confirm</button>
        </div>
      </div>
    </div>
  );
}
