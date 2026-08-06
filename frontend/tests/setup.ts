import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

Element.prototype.scrollIntoView = vi.fn();

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});
