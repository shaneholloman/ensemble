# Request Lifecycle and Retry Behavior

Ensemble now treats retries and terminal failure handling as an outer request-lifecycle concern instead of hiding them inside provider retry wrappers. That means one `ensembleRequest(...)` call has one authoritative outer status flow, even if multiple provider attempts happen underneath it.

## Authoritative Lifecycle

The outer request lifecycle is surfaced through `operation_status` events:

- `started` means the outer request has begun
- `retrying` means a recoverable failure happened and Ensemble is scheduling another attempt
- `completed` is the only authoritative success outcome
- `failed` is the only authoritative terminal failure outcome

The outer lifecycle keeps the same `request_id` across `started`, `retrying`, `completed`, and `failed`. Individual provider rounds and `agent_done` events can use different inner request IDs.

Typical recoverable sequence:

- `operation_status.started`
- `error` with `recoverable: true`
- `operation_status.retrying`
- later `operation_status.completed` or `operation_status.failed`

Typical terminal sequence:

- `operation_status.started`
- `error` with `recoverable: false`
- `operation_status.failed`

Nested verifier requests do not emit their own authoritative outer terminal statuses into the parent stream. The parent request still emits exactly one outer `completed` or `failed`.

## Error Event Contract

`error` events are still useful, but they are not the authoritative final outcome by themselves. They now carry more structured failure information:

```ts
type ErrorEvent = {
    type: 'error';
    error: string;
    code?: string;
    details?: unknown;
    recoverable?: boolean;
};
```

Use these fields as follows:

- `error`: human-readable failure message
- `code`: provider/network/app error code when available
- `details`: raw provider details when available
- `recoverable`: whether the outer lifecycle may retry this failure

Recoverable errors can appear before a final success. If you need the final outcome, prefer terminal `operation_status` events or `ensembleResult(...)`.

## Recoverable vs Terminal Failures

Recoverability is normalized in the outer request lifecycle so providers and callers share the same semantics.

Recoverable by default:

- network transport failures like `ECONNRESET`, `ETIMEDOUT`, `ENOTFOUND`, `ECONNREFUSED`
- retryable status codes like `408`, `429`, `500`, `502`, `503`, `504`, `522`, `524`
- matching transient messages like `fetch failed`, `network error`, or `Incomplete JSON segment`
- any custom codes/statuses added through `retryOptions.additionalRetryableErrors` or `retryOptions.additionalRetryableStatusCodes`

Terminal by default:

- aborts and whole-request timeouts
- request setup failures such as model/provider resolution errors
- malformed provider tool calls
- truncated/stopped provider responses that are not classified as transient
- strict structured-output schema validation failures
- verification failures after the configured verification attempts
- tool call limit and tool call round limit exhaustion
- errors that happen after terminal output has already been emitted

That last rule matters: if a provider already emitted terminal content and then drops the transport afterward, Ensemble does not retry and the final failure becomes terminal to avoid duplicating completed output.

## Retry Configuration

`retryOptions` configures outer retries only:

```typescript
const agent = {
    model: 'gpt-4',
    retryOptions: {
        maxRetries: 2,
        initialDelay: 1000,
        maxDelay: 5000,
        backoffMultiplier: 2,
        onRetry: (error, attempt) => {
            console.log(`Retry ${attempt}:`, error.message ?? error);
        },
    },
};
```

- `maxRetries`: maximum retries after the initial request. Default: `4`
- `initialDelay`: delay before the first retry. Default: `1000ms`
- `maxDelay`: upper bound for retry delay growth. Default: `30000ms`
- `backoffMultiplier`: exponential backoff multiplier. Default: `2`
- `additionalRetryableErrors`: extra transient error codes
- `additionalRetryableStatusCodes`: extra transient status codes
- `onRetry(error, attempt)`: callback when the outer lifecycle schedules retry `attempt`

Set `maxRetries: 0` to disable retries entirely.

Ensemble calls providers through `createResponseStream(...)` directly. The older provider-local `createResponseStreamWithRetry(...)` path remains only as a deprecated compatibility wrapper for downstream providers extending the public base class.

## Whole-Request Timeout and Abort Behavior

`modelSettings.timeout_ms` is a whole outer-request timeout budget, not a per-attempt timeout. The same budget applies across:

- the initial provider attempt
- any retries
- waiting for started tools to finish
- bounded failure finalization

If the budget expires, the outer request fails terminally and the provider abort signal is triggered. A user-supplied `abortSignal` is treated the same way: it propagates into the provider request and active tool execution, and the final failure is terminal.

## Tool Behavior During Failure Handling

Tool execution now participates directly in request failure handling:

- Started tools are allowed to finish before a recoverable retry begins
- On terminal request failure, Ensemble switches into bounded tool finalization
- Abort-aware tools receive an abort signal so they can stop quickly
- Queued sequential tools are not started once the request is already terminal
- If a non-abortable tool never settles, Ensemble can stop waiting and leave it tracked as still running instead of pretending it completed

This makes failures visible instead of hiding them behind fallbacks or synthetic success states.

## Structured Output and Verification

Prefer `modelSettings.json_schema` as the authoritative structured-output contract. The older `jsonSchema` agent property is still accepted as a compatibility alias and mapped onto `modelSettings.json_schema`.

When `json_schema.strict === true`:

- the final text response is validated by the outer request lifecycle
- validation failures are terminal with reason `structured_output_validation_failed`
- schema constraints like `exclusiveMinimum` and `exclusiveMaximum` are enforced

Verifier behavior also changed:

- verifier calls are forced onto a strict JSON schema of `{ status: "pass" | "fail", reason?: string }`
- invalid verifier JSON/schema output is treated as verifier failure, not ignored
- if verification fails and retries are allowed, Ensemble reruns the main request with verifier feedback
- if verification still fails after the configured attempts, the outer request fails terminally with reason `verification_failed`

## `ensembleResult(...)` Semantics

`ensembleResult(...)` now follows the outer lifecycle instead of treating the first error as final.

It exposes:

- `requestStatus`: outer request status (`started`, `retrying`, `failed`, `completed`)
- `failure`: final failure details `{ error, reason?, recoverable? }`
- `completed`: true only when the authoritative outer outcome is success

Behavioral rules:

- a terminal outer `completed` status clears earlier recoverable errors
- a terminal outer `failed` status is the final failure outcome
- if the stream ends without an authoritative terminal status, `ensembleResult(...)` falls back to the raw stream outcome
- `failFast: true` still waits for the authoritative failed request status before stopping, so callers receive the final normalized failure reason

## Example

```typescript
import { ensembleRequest, ensembleResult } from '@just-every/ensemble';

const messages = [{ type: 'message', role: 'user', content: 'Hello!' }];
const agent = {
    model: 'claude-3-5-haiku-latest',
    modelSettings: {
        timeout_ms: 15000,
    },
    retryOptions: {
        maxRetries: 1,
        onRetry: (error, attempt) => {
            console.log(`Retry ${attempt}`, error.code ?? error.message);
        },
    },
};

for await (const event of ensembleRequest(messages, agent)) {
    if (event.type === 'operation_status') {
        console.log('status', event.status, event.reason, event.attempt, event.max_attempts);
    }

    if (event.type === 'error') {
        console.log('error', event.recoverable ? 'recoverable' : 'terminal', event.code, event.error);
    }
}

const result = await ensembleResult(ensembleRequest(messages, agent), { failFast: true });
if (!result.completed) {
    console.error(result.requestStatus, result.failure?.reason, result.failure?.error);
}
```
