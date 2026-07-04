import { describe, expect, it } from 'vitest';
import { ClaudeProvider } from '../model_providers/claude.js';

function streamOf(events: unknown[]) {
    return {
        async *[Symbol.asyncIterator]() {
            for (const event of events) {
                yield event;
            }
        },
    };
}

function providerWith(events: unknown[]): ClaudeProvider {
    const provider = new ClaudeProvider('sk-ant-test');
    (provider as any)._client = {
        messages: {
            create: async () => streamOf(events),
        },
    };
    return provider;
}

async function collect(provider: ClaudeProvider): Promise<any[]> {
    const collected: any[] = [];
    for await (const event of provider.createResponseStream(
        [{ type: 'message', role: 'user', content: 'hello' }] as any,
        'claude-fable-5',
        { agent_id: 'test-claude-stop-reason' } as any
    )) {
        collected.push(event);
    }
    return collected;
}

describe('Claude stop_reason handling', () => {
    it('surfaces a refusal as a typed non-recoverable error event, not a silent empty stream', async () => {
        const events = await collect(
            providerWith([
                { type: 'message_start', message: { usage: { input_tokens: 1200, output_tokens: 1 } } },
                {
                    type: 'message_delta',
                    delta: {
                        stop_reason: 'refusal',
                        stop_details: { type: 'refusal', category: 'bio' },
                    },
                    usage: { output_tokens: 3 },
                },
                { type: 'message_stop' },
            ])
        );

        const errors = events.filter(event => event.type === 'error');
        expect(errors).toHaveLength(1);
        expect(errors[0].code).toBe('refusal');
        expect(errors[0].recoverable).toBe(false);
        expect(errors[0].error).toContain('stop_reason: refusal');
        expect(errors[0].error).toContain('category: bio');
        expect(errors[0].details).toEqual({
            stop_reason: 'refusal',
            stop_details: { type: 'refusal', category: 'bio' },
        });
        expect(events.filter(event => event.type === 'message_complete')).toHaveLength(0);
    });

    it('surfaces a refusal without stop_details', async () => {
        const events = await collect(
            providerWith([
                { type: 'message_start', message: { usage: { input_tokens: 100 } } },
                { type: 'message_delta', delta: { stop_reason: 'refusal' }, usage: { output_tokens: 1 } },
                { type: 'message_stop' },
            ])
        );

        const errors = events.filter(event => event.type === 'error');
        expect(errors).toHaveLength(1);
        expect(errors[0].code).toBe('refusal');
        expect(events.filter(event => event.type === 'message_complete')).toHaveLength(0);
    });

    it('surfaces a thinking-only max_tokens truncation as a typed error with a budget hint', async () => {
        const events = await collect(
            providerWith([
                { type: 'message_start', message: { usage: { input_tokens: 500 } } },
                { type: 'content_block_start', index: 0, content_block: { type: 'thinking' } },
                {
                    type: 'content_block_delta',
                    index: 0,
                    delta: { type: 'signature_delta', signature: 'sig' },
                },
                { type: 'content_block_stop', index: 0 },
                { type: 'message_delta', delta: { stop_reason: 'max_tokens' }, usage: { output_tokens: 256 } },
                { type: 'message_stop' },
            ])
        );

        const errors = events.filter(event => event.type === 'error');
        expect(errors).toHaveLength(1);
        expect(errors[0].code).toBe('max_tokens_no_output');
        expect(errors[0].recoverable).toBe(false);
        expect(errors[0].error).toContain('stop_reason: max_tokens');
        expect(errors[0].error).toContain('increase max_tokens');
        expect(events.filter(event => event.type === 'message_complete')).toHaveLength(0);
    });

    it('stamps stop_reason on message_complete for normal turns', async () => {
        const events = await collect(
            providerWith([
                { type: 'message_start', message: { usage: { input_tokens: 10 } } },
                { type: 'content_block_start', index: 0, content_block: { type: 'text' } },
                { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ready' } },
                { type: 'content_block_stop', index: 0 },
                { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } },
                { type: 'message_stop' },
            ])
        );

        expect(events.filter(event => event.type === 'error')).toHaveLength(0);
        const completes = events.filter(event => event.type === 'message_complete');
        expect(completes).toHaveLength(1);
        expect(completes[0].content).toBe('ready');
        expect(completes[0].stop_reason).toBe('end_turn');
    });

    it('does not emit an empty-response error for tool-only turns', async () => {
        const events = await collect(
            providerWith([
                { type: 'message_start', message: { usage: { input_tokens: 10 } } },
                {
                    type: 'content_block_start',
                    index: 0,
                    content_block: { type: 'tool_use', id: 'tool_1', name: 'do_thing', input: {} },
                },
                { type: 'content_block_stop', index: 0, content_block: { type: 'tool_use' } },
                { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 20 } },
                { type: 'message_stop' },
            ])
        );

        expect(events.filter(event => event.type === 'error')).toHaveLength(0);
        expect(events.filter(event => event.type === 'tool_start')).toHaveLength(1);
    });
});
