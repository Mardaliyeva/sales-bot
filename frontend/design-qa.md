# Design QA — Satış köməkçisi frontend-i

## Evidence

- Source visual truth: `C:\Users\aytan.mardaliyeva\Documents\sales-bot\frontend\design-reference.png`
- Source pixels: `1920 × 903`
- Intended implementation viewport: `1920 × 903` CSS px, device scale factor `1`
- Implementation URL: `http://127.0.0.1:3000/`
- Implementation screenshot: unavailable
- State: light-theme empty chat screen; desktop reference state
- Density normalization: not applicable yet because the implementation capture could not be produced

## Findings

- [P0] Browser-rendered implementation evidence is unavailable.
  - Location: in-app Browser verification and screenshot capture.
  - Evidence: the reference image opened successfully, the local frontend returned HTTP `200`, but the required in-app Browser control failed before a tab could be inspected or captured (`failed to write kernel assets: path not found`).
  - Impact: fonts, spacing, colors, image fidelity, copy, responsive layout, interaction states and console errors cannot be judged from a same-state visual comparison.
  - Fix: restore the in-app Browser connection, capture the empty screen at `1920 × 903`, then capture the medium and approximately `390px` mobile states and repeat this QA.

## Required Fidelity Surfaces

- Fonts and typography: blocked pending implementation screenshot.
- Spacing and layout rhythm: blocked pending implementation screenshot.
- Colors and visual tokens: blocked pending implementation screenshot.
- Image quality and asset fidelity: source robot assets are present in the implementation, but rendered scale and transparency cannot be accepted without a capture.
- Copy and content: covered by automated component tests; visual wrapping and hierarchy remain blocked.
- Responsiveness and accessibility: responsive CSS and semantic controls exist; rendered desktop/mobile evidence and keyboard/browser verification remain blocked.

## Full-view Comparison Evidence

The source full view was opened at `1920 × 903`. A same-viewport implementation image could not be captured, so no visual comparison was performed.

## Focused Region Comparison Evidence

Not performed. The full-view implementation capture is missing, so evaluating the sidebar brand area, hero, suggestions, composer and robot asset in isolation would not be evidence-backed.

## Automated and Runtime Evidence

- `npm test`: `16/16` passed.
- `npm run lint`: passed with zero warnings.
- `npm run build`: passed.
- Local frontend HTTP request: `200`.
- Next.js proxy session creation: passed.
- Next.js proxy chat with the Azure text deployment: passed (`finish_reason=completed`).
- Product query with Azure tool calling: passed (`used_tools=product_search`).
- API error, expired-session, loading, parallel-send, suggestion, Enter/Shift+Enter, storage corruption and storage-limit states: covered by tests.
- Browser console errors: not checked because Browser control could not connect.

## Comparison History

- Iteration 1: source opened; implementation screenshot and browser interaction capture blocked before comparison. No visual fixes were claimed from code-only inspection.

## Implementation Checklist

1. Reconnect the in-app Browser and capture the empty state at `1920 × 903`.
2. Compare the source and implementation in the same visual input; fix every P0/P1/P2 mismatch.
3. Verify drawer, suggestion, input, loading, error, recent-chat and new-chat interactions.
4. Capture medium and approximately `390px` mobile states and check console errors.

## Follow-up Polish

No P3 polish is classified until the first evidence-backed visual comparison is available.

final result: blocked
