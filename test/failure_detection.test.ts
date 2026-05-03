import { describe, expect, it } from 'vitest';
import { createProviderErrorEvent, normalizeFailure } from '../utils/failure_detection.js';

describe('failure_detection', () => {
    it('accepts custom retryable status codes when classifying failures', () => {
        const failure = normalizeFailure(Object.assign(new Error('teapot overload'), { status: 418 }), {
            retryableStatusCodes: [418],
        });

        expect(failure.recoverable).toBe(true);
        expect(failure.terminal).toBe(false);
    });

    it('propagates custom retryable status codes through provider error events', () => {
        const errorEvent = createProviderErrorEvent(Object.assign(new Error('teapot overload'), { status: 418 }), {
            prefix: 'Provider error: ',
            retryableStatusCodes: [418],
        });

        expect(errorEvent.error).toContain('teapot overload');
        expect(errorEvent.recoverable).toBe(true);
    });

    it('normalizes object-shaped provider errors without hiding the original error', () => {
        const failure = normalizeFailure({
            error: {
                message: 'OpenRouter rejected response_format',
                type: 'invalid_request_error',
                code: 'bad_request',
            },
            status: 400,
        });

        expect(failure.error).toContain('OpenRouter rejected response_format');
        expect(failure.error).toContain('invalid_request_error');
        expect(failure.recoverable).toBe(false);
    });

    it('still classifies retryable snippets from object-shaped provider errors', () => {
        const failure = normalizeFailure({
            error: {
                message: 'fetch failed while contacting provider',
            },
        });

        expect(failure.error).toContain('fetch failed');
        expect(failure.recoverable).toBe(true);
    });
});
