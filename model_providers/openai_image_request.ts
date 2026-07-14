export type OpenAIImageFailureReason =
    | 'aborted'
    | 'connection'
    | 'content_policy_violation'
    | 'empty_response'
    | 'http'
    | 'timeout';

export interface OpenAIImageErrorDetails {
    reason: OpenAIImageFailureReason;
    retryable: boolean;
    status?: number;
    providerRequestId?: string;
    providerCode?: string;
    providerType?: string;
    providerParam?: string;
    cause?: unknown;
}

export class OpenAIImageRequestError extends Error {
    readonly reason: OpenAIImageFailureReason;
    readonly retryable: boolean;
    readonly recoverable: boolean;
    readonly status?: number;
    readonly requestID?: string;
    readonly request_id?: string;
    readonly code?: string;
    readonly type?: string;
    readonly param?: string;
    readonly providerCode?: string;
    readonly providerType?: string;
    readonly providerParam?: string;
    readonly cause?: unknown;

    constructor(message: string, details: OpenAIImageErrorDetails) {
        super(message);
        this.name = 'OpenAIImageRequestError';
        this.reason = details.reason;
        this.retryable = details.retryable;
        this.recoverable = details.retryable;
        this.status = details.status;
        this.requestID = details.providerRequestId;
        this.request_id = details.providerRequestId;
        this.code = details.providerCode;
        this.type = details.providerType;
        this.param = details.providerParam;
        this.providerCode = details.providerCode;
        this.providerType = details.providerType;
        this.providerParam = details.providerParam;
        this.cause = details.cause;
    }
}

export const DEFAULT_OPENAI_IMAGE_TIMEOUT_MS = 5 * 60 * 1000;

type ProviderErrorShape = {
    name?: unknown;
    message?: unknown;
    status?: unknown;
    code?: unknown;
    type?: unknown;
    param?: unknown;
    requestID?: unknown;
    request_id?: unknown;
    error?: unknown;
    headers?: unknown;
};

const normalizedString = (value: unknown): string | undefined => {
    if (typeof value !== 'string') return undefined;
    const normalized = value.trim();
    return normalized.length > 0 ? normalized : undefined;
};

const nestedProviderError = (error: ProviderErrorShape): ProviderErrorShape =>
    error.error && typeof error.error === 'object' ? (error.error as ProviderErrorShape) : {};

const providerRequestId = (error: ProviderErrorShape): string | undefined => {
    const direct = normalizedString(error.requestID) ?? normalizedString(error.request_id);
    if (direct) return direct;
    return error.headers instanceof Headers ? normalizedString(error.headers.get('x-request-id')) : undefined;
};

const isPolicyFailure = (values: Array<string | undefined>): boolean => {
    const description = values.filter(Boolean).join(' ').toLowerCase();
    return (
        description.includes('content_policy') ||
        description.includes('content policy') ||
        description.includes('moderation_blocked') ||
        description.includes('moderation blocked') ||
        description.includes('safety violation')
    );
};

export function normalizeOpenAIImageError(error: unknown): OpenAIImageRequestError {
    if (error instanceof OpenAIImageRequestError) return error;

    const raw = error && typeof error === 'object' ? (error as ProviderErrorShape) : {};
    const nested = nestedProviderError(raw);
    const name = normalizedString(raw.name) ?? (error instanceof Error ? error.name : undefined);
    const message =
        normalizedString(raw.message) ??
        normalizedString(nested.message) ??
        (typeof error === 'string' ? error : undefined) ??
        'OpenAI image request failed.';
    const status = typeof raw.status === 'number' && Number.isFinite(raw.status) ? raw.status : undefined;
    const providerCode = normalizedString(raw.code) ?? normalizedString(nested.code);
    const providerType = normalizedString(raw.type) ?? normalizedString(nested.type);
    const providerParam = normalizedString(raw.param) ?? normalizedString(nested.param);
    const requestId = providerRequestId(raw);
    const policyFailure = isPolicyFailure([providerCode, providerType, message]);
    const timeout = name === 'APIConnectionTimeoutError' || name === 'TimeoutError' || providerCode === 'ETIMEDOUT';
    const aborted = name === 'AbortError' || providerCode === 'ABORT_ERR';
    const connection = name === 'APIConnectionError';
    const retryableStatus =
        status === 408 || status === 409 || status === 429 || (status !== undefined && status >= 500);

    const reason: OpenAIImageFailureReason = policyFailure
        ? 'content_policy_violation'
        : timeout
          ? 'timeout'
          : aborted
            ? 'aborted'
            : connection
              ? 'connection'
              : 'http';

    return new OpenAIImageRequestError(message, {
        reason,
        retryable: timeout || connection || retryableStatus,
        status,
        providerRequestId: requestId,
        providerCode,
        providerType,
        providerParam,
        cause: error,
    });
}

const timeoutError = (operationName: string, timeoutMs: number): OpenAIImageRequestError =>
    new OpenAIImageRequestError(`${operationName} timed out after ${timeoutMs}ms`, {
        reason: 'timeout',
        retryable: true,
        providerCode: 'ETIMEDOUT',
    });

const abortError = (operationName: string, cause?: unknown): OpenAIImageRequestError =>
    new OpenAIImageRequestError(`${operationName} aborted`, {
        reason: 'aborted',
        retryable: false,
        providerCode: 'ABORT_ERR',
        cause,
    });

export async function runOpenAIImageRequest<T>(options: {
    operationName: string;
    timeoutMs: number;
    abortSignal?: AbortSignal;
    execute: (signal: AbortSignal) => Promise<T>;
}): Promise<T> {
    if (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0) {
        throw new RangeError('OpenAI image timeoutMs must be a positive finite number.');
    }

    const controller = new AbortController();
    let abortListener: (() => void) | undefined;
    let rejectGuard: ((error: OpenAIImageRequestError) => void) | undefined;
    const guardPromise = new Promise<never>((_, reject) => {
        rejectGuard = reject;
    });

    const abortWith = (error: OpenAIImageRequestError) => {
        if (!controller.signal.aborted) controller.abort(error);
        rejectGuard?.(error);
    };

    if (options.abortSignal?.aborted) {
        abortWith(abortError(options.operationName, options.abortSignal.reason));
    } else if (options.abortSignal) {
        abortListener = () => abortWith(abortError(options.operationName, options.abortSignal?.reason));
        options.abortSignal.addEventListener('abort', abortListener, { once: true });
    }

    const timeoutId = setTimeout(() => {
        abortWith(timeoutError(options.operationName, options.timeoutMs));
    }, options.timeoutMs);
    const operationPromise = Promise.resolve().then(() => options.execute(controller.signal));

    try {
        return await Promise.race([operationPromise, guardPromise]);
    } catch (error) {
        operationPromise.catch(() => {
            // The provider may reject after the hard guard has already settled the caller.
        });
        throw normalizeOpenAIImageError(error);
    } finally {
        clearTimeout(timeoutId);
        if (options.abortSignal && abortListener) {
            options.abortSignal.removeEventListener('abort', abortListener);
        }
    }
}
