// YAML parse wrapper (kept tiny so the registry stays testable).
import { parse } from "yaml";
export function yaml(raw: string): unknown {
  return parse(raw) as unknown;
}
