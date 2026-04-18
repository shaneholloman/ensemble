import { describe, expect, it } from 'vitest';
import { BaseModelProvider } from '../model_providers/base_provider.js';
import type { AgentDefinition, ProviderStreamEvent, ResponseInput } from '../types/types.js';

class TestBaseProvider extends BaseModelProvider {
    attempts = 0;

    constructor() {
        super('openai');
    }

    async *createResponseStream(
        _messages: ResponseInput,
        _model: string,
        _agent: AgentDefinition
    ): AsyncGenerator<ProviderStreamEvent> {
        this.attempts += 1;

        if (this.attempts === 1) {
            throw Object.assign(new Error('fetch failed: ECONNRESET'), {
                code: 'ECONNRESET',
            });
        }

        yield {
            type: 'message_complete',
            message_id: 'provider-retry-success',
            content: 'Recovered response',
        } as ProviderStreamEvent;
    }
}

describe('BaseModelProvider', () => {
    it('keeps the deprecated retry helper available for downstream providers', async () => {
        const provider = new TestBaseProvider();
        const events: ProviderStreamEvent[] = [];

        for await (const event of provider.createResponseStreamWithRetry(
            [{ type: 'message', role: 'user', content: 'Hello' } as any],
            'test-model',
            {
                retryOptions: {
                    maxRetries: 1,
                },
            }
        )) {
            events.push(event);
        }

        expect(provider.attempts).toBe(2);
        expect(events).toEqual([
            {
                type: 'message_complete',
                message_id: 'provider-retry-success',
                content: 'Recovered response',
            },
        ]);
    });
});
