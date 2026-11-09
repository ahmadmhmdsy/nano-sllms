import type { CSSProperties, KeyboardEvent } from "react";

/** Shared search input (pattern from flow/'s Task M2.2 toolbar search).
 * Proper labeling out of the box: pass label for a visible <label>
 * associated via htmlFor/id, or ariaLabel for an aria-label when no
 * visible label is wanted (defaults to "Search"). The magnifier glyph is
 * decorative (aria-hidden) and inherits the wrapper color via
 * currentColor. onChange receives the string value directly so callers
 * can wire it straight into a store; onKeyDown is forwarded for
 * Enter/Escape shortcuts. Colors come from the --bg / --ink / --dim /
 * --line CSS variables (viz defines them; the fallbacks reproduce flow's
 * light palette). */
export interface SearchInputProps {
  id: string;
  value: string;
  onChange: (value: string) => void;
  onKeyDown?: (e: KeyboardEvent<HTMLInputElement>) => void;
  placeholder?: string;
  /** Visible label rendered with htmlFor. */
  label?: string;
  /** Accessible name when no visible label is given (default "Search"). */
  ariaLabel?: string;
  className?: string;
}

const searchInputStyle: CSSProperties = {
  border: 0,
  outline: "none",
  font: "inherit",
  background: "transparent",
  color: "var(--ink, #333)",
};

export default function SearchInput({ id, value, onChange, onKeyDown, placeholder, label, ariaLabel, className }: SearchInputProps) {
  return (
    <div
      className={"ui-search" + (className ? " " + className : "")}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        border: "1px solid var(--line, #ccc)",
        borderRadius: 4,
        padding: "2px 6px",
        background: "var(--bg, #fff)",
        color: "var(--dim, #888)",
      }}
    >
      {label !== undefined && (
        <label htmlFor={id} className="ui-search-label">
          {label}
        </label>
      )}
      <svg className="ui-search-icon" width="13" height="13" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <line x1="11" y1="11" x2="14.5" y2="14.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <input
        id={id}
        type="search"
        value={value}
        placeholder={placeholder}
        aria-label={label === undefined ? ariaLabel ?? "Search" : undefined}
        style={searchInputStyle}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
      />
    </div>
  );
}
