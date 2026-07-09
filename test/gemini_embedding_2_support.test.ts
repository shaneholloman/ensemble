import { describe, expect, it, vi } from 'vitest';
import { MODEL_CLASSES, findModel } from '../data/model_data.js';
import { GeminiProvider } from '../model_providers/gemini.js';

describe('Gemini Embedding 2 support', () => {
    it('registers Gemini Embedding 2 with its current text embedding metadata', () => {
        const model = findModel('gemini-embedding-2');

        expect(model).toMatchObject({
            id: 'gemini-embedding-2',
            provider: 'google',
            cost: {
                input_per_million: 0.2,
                output_per_million: 0,
            },
            features: {
                input_modality: ['text'],
                output_modality: ['embedding'],
                input_token_limit: 8192,
            },
            embedding: true,
            dim: 3072,
            class: 'embedding',
        });
        expect(MODEL_CLASSES.embedding.models).toContain('gemini-embedding-2');
    });

    it('wraps chunked input and omits the unsupported taskType parameter', async () => {
        const provider = new GeminiProvider('test-key');
        const embedContent = vi.fn().mockResolvedValue({
            embeddings: [{ values: [0.1, 0.2] }, { values: [0.3, 0.4] }],
        });
        (provider as any)._client = { models: { embedContent } };

        await expect(
            provider.createEmbedding(['first chunk', 'second chunk'], 'gemini-embedding-2', {
                agent_id: 'test-gemini-embedding-2',
            } as any)
        ).resolves.toEqual([
            [0.1, 0.2],
            [0.3, 0.4],
        ]);

        expect(embedContent).toHaveBeenCalledWith({
            model: 'gemini-embedding-2',
            contents: [{ parts: [{ text: 'first chunk' }] }, { parts: [{ text: 'second chunk' }] }],
            config: {},
        });
    });

    it('rejects taskType instead of sending an invalid Embedding 2 request', async () => {
        const provider = new GeminiProvider('test-key');
        const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);

        await expect(
            provider.createEmbedding('query', 'gemini-embedding-2', { agent_id: 'test-gemini-embedding-2' } as any, {
                taskType: 'RETRIEVAL_QUERY',
            })
        ).rejects.toThrow('does not support taskType');

        consoleError.mockRestore();
    });
});
