/** Generic loading spinner (shared UI library; extracted from flow/,
 * Task M2.5). Pure CSS rotation driven by app CSS (.spinner + the
 * spinner-rotate keyframes) - no JS timers, so it is cheap to mount and
 * safe to drop into any in-flight state. role="status" +
 * aria-label="Loading" announce it to screen readers; the className prop
 * lets callers size/place it per surface. The color follows the
 * --color-primary variable when an app defines it and otherwise inherits
 * the surrounding text color (currentColor), so the spinner themes
 * automatically in both flow (light) and viz (dark). */
export default function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={className === undefined ? "spinner" : "spinner " + className}
      role="status"
      aria-label="Loading"
      style={{ color: "var(--color-primary, currentColor)" }}
    />
  );
}
