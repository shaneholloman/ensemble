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
});
