import { describe, expect, it } from 'vitest';
import type { ProviderStreamEvent } from '../types/types.js';
import { ensembleResult } from '../utils/ensemble_result.js';

async function* createStream(events: ProviderStreamEvent[]): AsyncGenerator<ProviderStreamEvent> {
    for (const event of events) {
        yield event;
    }
}

describe('ensembleResult', () => {
    it('treats a completed request status as authoritative over recoverable error events', async () => {
        const result = await ensembleResult(
            createStream([
                {
                    type: 'operation_status',
                    operation: 'request',
                    status: 'started',
                    terminal: false,
                    will_continue: true,
                } as ProviderStreamEvent,
                {
                    type: 'error',
                    error: 'temporary provider failure',
                    recoverable: true,
                } as ProviderStreamEvent,
                {
                    type: 'operation_status',
                    operation: 'request',
                    status: 'completed',
                    terminal: true,
                    recoverable: false,
                    will_continue: false,
                } as ProviderStreamEvent,
                {
                    type: 'stream_end',
                } as ProviderStreamEvent,
            ])
        );

        expect(result.completed).toBe(true);
        expect(result.requestStatus).toBe('completed');
        expect(result.error).toBeUndefined();
        expect(result.failure).toBeUndefined();
    });

    it('treats a failed request status as the authoritative terminal outcome', async () => {
        const result = await ensembleResult(
            createStream([
                {
                    type: 'operation_status',
                    operation: 'request',
                    status: 'started',
                    terminal: false,
                    will_continue: true,
                } as ProviderStreamEvent,
                {
                    type: 'error',
                    error: 'terminal provider failure',
                    recoverable: false,
                } as ProviderStreamEvent,
                {
                    type: 'operation_status',
                    operation: 'request',
                    status: 'failed',
                    terminal: true,
                    recoverable: false,
                    will_continue: false,
                    error: 'terminal provider failure',
                    reason: 'provider_failed',
                } as ProviderStreamEvent,
                {
                    type: 'stream_end',
                } as ProviderStreamEvent,
            ])
        );

        expect(result.completed).toBe(false);
        expect(result.requestStatus).toBe('failed');
        expect(result.error).toBe('terminal provider failure');
        expect(result.failure).toEqual({
            error: 'terminal provider failure',
            reason: 'provider_failed',
            recoverable: false,
        });
    });

    it('preserves recoverable errors when no outer terminal status is emitted', async () => {
        const result = await ensembleResult(
            createStream([
                {
                    type: 'error',
                    error: 'temporary provider failure',
                    recoverable: true,
                } as ProviderStreamEvent,
                {
                    type: 'stream_end',
                } as ProviderStreamEvent,
            ])
        );

        expect(result.completed).toBe(false);
        expect(result.error).toBe('temporary provider failure');
        expect(result.failure).toEqual({
            error: 'temporary provider failure',
            recoverable: true,
        });
    });

    it('keeps reading through a terminal error until the authoritative failed status arrives in failFast mode', async () => {
        const result = await ensembleResult(
            createStream([
                {
                    type: 'error',
                    error: 'terminal provider failure',
                    recoverable: false,
                } as ProviderStreamEvent,
                {
                    type: 'operation_status',
                    operation: 'request',
                    status: 'failed',
                    terminal: true,
                    recoverable: false,
                    will_continue: false,
                    error: 'terminal provider failure',
                    reason: 'request_setup_failed',
                } as ProviderStreamEvent,
                {
                    type: 'stream_end',
                } as ProviderStreamEvent,
            ]),
            { failFast: true }
        );

        expect(result.completed).toBe(false);
        expect(result.requestStatus).toBe('failed');
        expect(result.failure).toEqual({
            error: 'terminal provider failure',
            reason: 'request_setup_failed',
            recoverable: false,
        });
    });
});
