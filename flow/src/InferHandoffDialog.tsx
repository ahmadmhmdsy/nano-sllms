import { useEffect } from "react";
import { useFlowStore } from "./store";
import { attemptWebuiOpen, inferHandoffDialog, WEBUI_URL } from "./inferHandoff";

/** Task 11 infer-handoff dialog. Rendered by App whenever the store's
 * inferHandoffNodeId is non-null (set by PipelineNode's "Open in webui
 * Chat" click). ALWAYS shown - the window.open attempt to WEBUI_URL is
 * best-effort: a blocked popup or a not-running webui still leaves the
 * user with the manual launch instructions below (honest MVP; there is
 * no server handoff endpoint by design). */
export default function InferHandoffDialog() {
  const nodeId = useFlowStore((s) => s.inferHandoffNodeId);
  const dismiss = useFlowStore((s) => s.dismissInferHandoff);
  /* Attempt the popup only when the dialog actually opens (the node id
   * going non-null), one attempt per open - the component stays mounted
   * for the whole app, so an unconditional effect would fire on load. */
  useEffect(() => {
    if (nodeId !== null) attemptWebuiOpen();
  }, [nodeId]);
  /* Escape dismisses the dialog (drill finding: it was Escape-less).
   * Scoped to the open state, window level so it fires regardless of
   * focus target, like the App Escape handler does for the file modals. */
  useEffect(() => {
    if (nodeId === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") dismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nodeId, dismiss]);
  if (nodeId === null) return null;
  const info = inferHandoffDialog();
  return (
    <div className="modal-backdrop" onClick={dismiss}>
      <div
        className="modal"
        role="dialog"
        aria-label="Open inference in webui Chat"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="modal-title">Open in webui Chat (node {nodeId})</p>
        <p className="modal-note">
          The flow editor does not run the webui itself. The checkpoint
          produced by this pipeline is {info.checkpointNote} - start the
          webui manually, open its {info.chatTab} tab there and load the
          checkpoint the connected train node wrote.
        </p>
        <p className="modal-note">
          Manual launch:
          <code className="infer-handoff-cmd">{info.launchCommand}</code>
          then open{" "}
          <a href={info.url} target="_blank" rel="noreferrer">
            {WEBUI_URL}
          </a>
        </p>
        <p className="honesty-label">
          A new tab to {WEBUI_URL} was attempted just now - popups may be
          blocked and the webui may not be running (a browser error page in
          that tab is the expected, honest MVP behavior).
        </p>
        <div className="modal-actions">
          <button onClick={dismiss}>Close</button>
        </div>
      </div>
    </div>
  );
}
