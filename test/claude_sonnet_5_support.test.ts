import { describe, expect, it, vi } from 'vitest';
import { MODEL_CLASSES, findModel } from '../data/model_data.js';
import { ClaudeProvider } from '../model_providers/claude.js';
import { getModelFromAgent, getProviderFromModel } from '../model_providers/model_provider.js';

async function drain(stream: AsyncIterable<unknown>): Promise<void> {
    for await (const _event of stream) {
        // Intentionally empty: these tests only need the provider to build the request.
    }
}

function emptyStream() {
    return {
        async *[Symbol.asyncIterator]() {
            // No-op stream.
        },
    };
}

describe('Claude Sonnet 5 support', () => {
    it('registers Sonnet 5 with current aliases, pricing, and metadata', () => {
        const model = findModel('claude-sonnet-5');
        const latestAlias = findModel('claude-sonnet-latest');
        const prefixlessAlias = findModel('sonnet-5');
        const previous = findModel('claude-sonnet-4.6');

        expect(model?.id).toBe('claude-sonnet-5');
        expect(latestAlias?.id).toBe('claude-sonnet-5');
        expect(prefixlessAlias?.id).toBe('claude-sonnet-5');
        expect(previous?.id).toBe('claude-sonnet-4-6');
        expect(getProviderFromModel('claude-sonnet-5')).toBe('anthropic');

        expect(model?.cost?.input_per_million).toBe(2.0);
        expect(model?.cost?.cached_input_per_million).toBe(0.2);
        expect(model?.cost?.output_per_million).toBe(10.0);
        expect(model?.features?.context_length).toBe(1_000_000);
        expect(model?.features?.max_output_tokens).toBe(128000);
        expect(model?.features?.input_modality).toEqual(['text', 'image']);
        expect(model?.features?.output_modality).toEqual(['text']);
        expect(model?.features?.tool_use).toBe(true);
        expect(model?.features?.streaming).toBe(true);
        expect(model?.features?.json_output).toBe(true);
        expect(model?.features?.reasoning_output).toBe(true);
    });

    it('uses Sonnet 5 for broad Anthropic model classes while Fable remains highest tier', () => {
        expect(MODEL_CLASSES.standard.models).toContain('claude-sonnet-5');
        expect(MODEL_CLASSES.reasoning.models).toContain('claude-sonnet-5');
        expect(MODEL_CLASSES.reasoning_mini.models).toContain('claude-sonnet-5');
        expect(MODEL_CLASSES.monologue.models).toContain('claude-sonnet-5');
        expect(MODEL_CLASSES.writing.models).toContain('claude-sonnet-5');
        expect(MODEL_CLASSES.reasoning_high.models).toContain('claude-fable-5');
    });

    it('normalizes Sonnet 5 aliases while preserving effort suffixes', async () => {
        const latest = await getModelFromAgent({
            agent_id: 'test-claude-sonnet-latest',
            model: 'claude-sonnet-latest',
        } as any);
        const xhigh = await getModelFromAgent({
            agent_id: 'test-claude-sonnet-5-xhigh',
            model: 'claude-sonnet-latest-xhigh',
        } as any);
        const prefixless = await getModelFromAgent({
            agent_id: 'test-sonnet-5-alias',
            model: 'sonnet-5',
        } as any);

        expect(latest).toBe('claude-sonnet-5');
        expect(xhigh).toBe('claude-sonnet-5-xhigh');
        expect(prefixless).toBe('claude-sonnet-5');
    });

    it('uses implicit adaptive thinking controls and omits sampling params for Sonnet 5 requests', async () => {
        const provider = new ClaudeProvider('sk-ant-test');
        const create = vi.fn().mockResolvedValue(emptyStream());
        (provider as any)._client = {
            messages: {
                create,
            },
        };

        await drain(
            provider.createResponseStream(
                [{ type: 'message', role: 'user', content: 'Review this carefully' }] as any,
                'claude-sonnet-latest-xhigh',
                {
                    agent_id: 'test-claude-sonnet-5-request',
                    modelSettings: {
                        temperature: 0.3,
                    },
                } as any
            )
        );

        const requestParams = create.mock.calls.at(0)?.[0];
        expect(requestParams?.model).toBe('claude-sonnet-5');
        expect(requestParams?.thinking).toBeUndefined();
        expect(requestParams?.output_config).toEqual({
            effort: 'xhigh',
        });
        expect(requestParams?.temperature).toBeUndefined();
    });
});
