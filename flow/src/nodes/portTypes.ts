/**
 * Port-type compatibility rule (named ONCE, per the registry-promotion
 * plan). Moved verbatim out of flow/src/store.ts (T3); store.ts
 * re-exports it so the existing import surface stays green.
 */

/** Port-type compatibility: exact match or one side is a prefixed
 * description of the other ("raw-dir" vs "raw-dir: dataset name/rows").
 * The dataset node's out port carries a descriptive suffix, so strict
 * string equality would refuse the primary dataset -> prepare edge. */
export function portTypesCompatible(a: string, b: string): boolean {
  const base = (t: string) => t.split(":")[0].trim();
  return base(a) === base(b);
}
